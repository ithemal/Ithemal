#include <stddef.h> /* for offsetof */
#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */
#include <stdlib.h>

#include "common.h"
#include "code_embedding.h"
#include "sql_dump.h"
#include "client.h"
#include "mmap.h"

client_arg_t client_args;
config_t config;
mmap_file_t filenames_file;
uint32_t num_threads = 0; //sqlite can handle only one thread
void * mutex;

#define BEGIN_CONTROL(C,ST,EN)						   \
  if(client_args.op_mode == SNOOP){					   \
    while(!__sync_bool_compare_and_swap(&C,ST,EN));			   \
  }									   \
 
#define END_CONTROL(C,ST,EN)						\
  if(client_args.op_mode == SNOOP){					\
    DR_ASSERT(__sync_bool_compare_and_swap(&C,ST,EN));			\
    while(C != IDLE);							\
  }									\


void post_cleancall();

//thread init event
static void 
thread_init(void * drcontext){

  mmap_file_t * data = (mmap_file_t *)dr_thread_alloc(drcontext, sizeof(mmap_file_t));    
  dr_set_tls_field(drcontext,data);
  
  get_perthread_filename(drcontext,data->filename,MAX_MODULE_SIZE);
  data->offs = data->filled = 0;
  
  if(client_args.op_mode == SNOOP){ //create filenames file for snooping process to know where we are dumping data
    dr_mutex_lock(mutex);
    volatile thread_files_t * files = filenames_file.data;
    DR_ASSERT(files); //should be memory mapped 

    BEGIN_CONTROL(files->control,IDLE,DR_CONTROL)
    strcpy(files->modules[files->num_modules],data->filename);
    files->num_modules++;
    END_CONTROL(files->control,DR_CONTROL,DUMP_ONE);

    dr_mutex_unlock(mutex);  
  }

  //create the file and memory map it  
  data->file = dr_open_file(data->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
  create_memory_map_file(data,TOTAL_SIZE);
  memset(data->data,0,TOTAL_SIZE);

  //if mode is raw sql dumping 
  if(client_args.op_mode == RAW_SQL){

    //processor model information
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    bk->arch = proc_get_model();    
    bk->static_file = dr_thread_alloc(drcontext, sizeof(mmap_file_t));
    create_raw_file(drcontext,client_args.data_folder,"static",bk->static_file);
  
    //insert the config string (query)
    query_t * query = (query_t *)(data->data + START_QUERY);
    int sz = insert_config(query, &config, client_args.op_mode);
    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    sz = complete_query(query,sz);
    write_to_file(bk->static_file,query,sz);
  }


}

bool dump_bb(void * drcontext, code_embedding_t embedding_func, code_info_t * cinfo, instrlist_t * bb, bool insert, bookkeep_t * bk, query_t * query){

  int sz = -1;

  //get the embedding
  embedding_func(drcontext, cinfo, bb);

  if(cinfo->code_size == -1){
    return false;
  }

  if(client_args.op_mode != SNOOP){

    if(insert)
      sz = insert_code(query, cinfo, client_args.op_mode);
    else
      sz = update_code(query, cinfo, client_args.op_mode);

    if(sz == -1){
      return false;
    }

    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    sz = complete_query(query,sz);
    write_to_file(bk->static_file,query,sz);

  }

  return true;


} 

//bb analysis routines
bool populate_bb_info(void * drcontext, volatile code_info_t * cinfo, instrlist_t * bb, bookkeep_t * bk, query_t * query){

  instr_t * first = instrlist_first(bb);
  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = first_pc - md->start;
  
  strcpy(cinfo->module,dr_module_preferred_name(md));
  cinfo->module_start = md->start;
  cinfo->rel_addr = rel_addr;

  uint32_t inserted = false;


  if( (client_args.dump_mode & DUMP_TOKEN) == DUMP_TOKEN ){

    cinfo->code_type = CODE_TOKEN;

    if(client_args.insert_or_update == INSERT_CODE && !inserted){
      if(!dump_bb(drcontext, token_text_embedding, cinfo, bb, true, bk, query)){
	return false;
      }
      inserted = true;
    }
    else{
      if(!dump_bb(drcontext, token_text_embedding, cinfo, bb, false, bk, query)){
	return false;
      }
    }

  }

  if( (client_args.dump_mode & DUMP_INTEL) == DUMP_INTEL ){

    cinfo->code_type = CODE_INTEL;

    disassemble_set_syntax(DR_DISASM_INTEL);
    if(client_args.insert_or_update == INSERT_CODE && !inserted){
      if(!dump_bb(drcontext, textual_embedding, cinfo, bb, true, bk, query)){
	return false;
      }
      inserted = true;
    }
    else{
      if(!dump_bb(drcontext, textual_embedding, cinfo, bb, false, bk, query)){
	return false;
      }
    }

  }


  if( (client_args.dump_mode & DUMP_ATT) == DUMP_ATT ){
    
    cinfo->code_type = CODE_ATT;

    disassemble_set_syntax(DR_DISASM_ATT);
    if(client_args.insert_or_update == INSERT_CODE && !inserted){
      if(!dump_bb(drcontext, textual_embedding_with_size, cinfo, bb, true, bk, query)){
	return false;
      }
      inserted = true;
    }
    else{
      if(!dump_bb(drcontext, textual_embedding_with_size, cinfo, bb, false, bk, query)){
	return false;
      }
    }

  }
  
  return true;

}

static dr_emit_flags_t
bb_creation_event(void * drcontext, void * tag, instrlist_t * bb, bool for_trace, bool translating){

  instr_t * first;
  instr_t * last;

  first = instrlist_first(bb);
  last = instrlist_last(bb);

  mmap_file_t * file = dr_get_tls_field(drcontext);
  if(file->data == NULL){
    return DR_EMIT_DEFAULT;    
  }

  volatile query_t * query = (query_t *)(file->data + START_QUERY);
  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);
  volatile code_info_t * cinfo = (code_info_t *)(file->data + START_CODE_DATA);

  //not filtering based on executable for static data collection


  //bb analysis 
  BEGIN_CONTROL(cinfo->control,IDLE,DR_CONTROL);


  if(!populate_bb_info(drcontext,cinfo,bb,bk,query)){
    return DR_EMIT_DEFAULT;
  }
 
  END_CONTROL(cinfo->control,DUMP_ONE,DR_CONTROL);

  bk->num_bbs++;

  return DR_EMIT_DEFAULT;
 
}

  /* thread exit event */
static void
thread_exit(void * drcontext){

  mmap_file_t * file = dr_get_tls_field(drcontext);
  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);
    
  if(client_args.op_mode == RAW_SQL){
    close_raw_file(bk->static_file);
  }
  close_memory_map_file(file,TOTAL_SIZE);
 
}

static void
event_exit(void)
{
  dr_mutex_destroy(mutex);
    
  if(client_args.op_mode == SNOOP){
    volatile thread_files_t * files = filenames_file.data;
    files->control = EXIT;
    while(files->control != IDLE);
    close_memory_map_file(&filenames_file,sizeof(thread_files_t));
  }

}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{

  //dr_printf("static client starting...\n");

  
    dr_register_thread_init_event(thread_init);    
    dr_register_bb_event(bb_creation_event);
    dr_register_thread_exit_event(thread_exit);
    dr_register_exit_event(event_exit);
    
    DR_ASSERT(argc == 7);
    client_args.op_mode = atoi(argv[1]);
    client_args.dump_mode = atoi(argv[2]);
    client_args.insert_or_update = atoi(argv[3]);
    strncpy(config.compiler,argv[4], MAX_STRING_SIZE);
    strncpy(config.flags,argv[5], MAX_STRING_SIZE);
    strncpy(client_args.data_folder,argv[6], MAX_STRING_SIZE);
    config.arch = proc_get_model();

    DR_ASSERT(client_args.op_mode != SQLITE);

    mutex = dr_mutex_create();
    num_threads = 0;

    //dr_printf("mode - %d\n",client_args.mode);
    
    if(client_args.op_mode == SNOOP){
      strcpy(filenames_file.filename,FILENAMES_FILE);
      filenames_file.filled = 0;
      filenames_file.offs = 0;
      filenames_file.file = dr_open_file(filenames_file.filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
      create_memory_map_file(&filenames_file,sizeof(thread_files_t));
      memset(filenames_file.data,0,sizeof(thread_files_t));
    }
  
  
}
