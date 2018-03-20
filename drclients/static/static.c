#include <stddef.h> /* for offsetof */
#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */
#include <stdlib.h>

#include "dr_api.h"
#include "timing_mmap.h"
#include "timing_logic.h"
#include "timing_dump.h"
#include "sqlite3_impl.h"
#include "common.h"

//client arguments
typedef struct {
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
  uint32_t mode;
  uint32_t code_format;
  code_embedding_t embedding_func;
} client_arg_t;

client_arg_t client_args;
mmap_file_t filenames_file;
uint32_t num_threads = 0; //sqlite can handle only one thread
void * mutex;

#define BEGIN_CONTROL(C,ST,EN)						   \
  if(client_args.mode == SNOOP){					   \
    while(!__sync_bool_compare_and_swap(&C,ST,EN));			   \
  }									   \
 
#define END_CONTROL(C,ST,EN)						\
  if(client_args.mode == SNOOP){					\
    DR_ASSERT(__sync_bool_compare_and_swap(&C,ST,EN));			\
    while(C != IDLE);							\
  }									\


void post_cleancall();

//thread init event
static void 
thread_init(void * drcontext){

  mmap_file_t * data = (mmap_file_t *)dr_thread_alloc(drcontext, sizeof(mmap_file_t));    
  dr_set_tls_field(drcontext,data);
  
  get_filename(drcontext,data->filename,MAX_MODULE_SIZE);
  data->offs = data->filled = 0;
  
  if(client_args.mode == SNOOP){ //create filenames file for snooping process to know where we are dumping data
    dr_mutex_lock(mutex);
    volatile thread_files_t * files = filenames_file.data;
    DR_ASSERT(files); //should be memory mapped 

    BEGIN_CONTROL(files->control,IDLE,DR_CONTROL)
    strcpy(files->modules[files->num_modules],data->filename);
    files->num_modules++;
    END_CONTROL(files->control,DR_CONTROL,DUMP_ONE);

    dr_mutex_unlock(mutex);  
  }
  else if (client_args.mode == SQLITE){ //keep track of data in this mode
    dr_mutex_lock(mutex);
    num_threads++;
    dr_mutex_unlock(mutex);
  }

  //create the file and memory map it and if sqlite mode then only dump data from thread 1
  if(client_args.mode != SQLITE || num_threads <= 1){
    data->file = dr_open_file(data->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
    create_memory_map_file(data,TOTAL_SIZE);
    memset(data->data,0,TOTAL_SIZE);
  
    //processor model information
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    bk->arch = proc_get_model();
  }
  else{
    data->data = NULL;
  }

  //if mode is raw sql dumping 
  if(client_args.mode == RAW_SQL){
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    bk->mmap_raw_file = dr_thread_alloc(drcontext, sizeof(mmap_file_t));
    create_raw_file(drcontext,bk->mmap_raw_file);
  }

  //insert the config string (query)
  if(client_args.mode != SNOOP){
    query_t * query = (query_t *)(data->data + START_QUERY);
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    int sz = insert_config(query, client_args.compiler, client_args.flags, client_args.mode);
    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    //dr_printf("%s\n",query);
    if(client_args.mode == SQLITE){
      query_db(query);
    }
    else if(client_args.mode == RAW_SQL){
      sz = complete_query(query,sz);
      write_to_file(bk->mmap_raw_file,query,sz);
    }
  }

  //dr_printf("thread %d initialized..\n",dr_get_thread_id(drcontext));

}

//bb analysis routines
void populate_bb_info(void * drcontext, volatile code_info_t * cinfo, instrlist_t * bb, code_embedding_t code_embedding){

  instr_t * first = instrlist_first(bb);
  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = first_pc - md->start;
  
  strcpy(cinfo->module,dr_module_preferred_name(md));
  cinfo->module_start = md->start;
  cinfo->rel_addr = rel_addr;

  code_embedding(drcontext, cinfo, bb);

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

  populate_bb_info(drcontext,cinfo,&tinfo[bk->num_bbs],bb,client_args.embedding_func);
  if(client_args.mode != SNOOP){
    int sz = insert_code(query,cinfo->module,cinfo->rel_addr,cinfo->code, client_args.mode, cinfo->code_size);
    //dr_printf("%s\n",query);
    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    if(client_args.mode == SQLITE){
      query_db(query);
    }
    else if(client_args.mode == RAW_SQL){
      sz = complete_query(query,sz);
      write_to_file(bk->mmap_raw_file,query,sz);
    }
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
    
  if(client_args.mode == RAW_SQL){
    close_raw_file(bk->mmap_raw_file);
  }
  close_memory_map_file(file,TOTAL_SIZE);
 
}

static void
event_exit(void)
{
  dr_mutex_destroy(mutex);
    
  if(client_args.mode == SNOOP){
    volatile thread_files_t * files = filenames_file.data;
    files->control = EXIT;
    while(files->control != IDLE);
    close_memory_map_file(&filenames_file,sizeof(thread_files_t));
  }
  if(client_args.mode == SQLITE){
    connection_close();
  }

}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{

  //dr_printf("timing client starting...\n");
    
    /* register events */
    dr_register_thread_init_event(thread_init);    
    dr_register_bb_event(bb_creation_event);
    dr_register_thread_exit_event(thread_exit);
    dr_register_exit_event(event_exit);
    
    disassemble_set_syntax(DR_DISASM_INTEL);

    /* client arguments */
    DR_ASSERT(argc == 5);
    client_args.mode = atoi(argv[1]);
    client_args.code_format = atoi(argv[2]);
    strncpy(client_args.compiler,argv[3], MAX_STRING_SIZE);
    strncpy(client_args.flags,argv[4], MAX_STRING_SIZE);


    if(client_args.code_format == TEXT){
      client_args.embedding_func = textual_embedding;
    }
    else if(client_args.code_format == TOKEN){
      client_args.embedding_func = token_text_embedding;
    }

    mutex = dr_mutex_create();
    num_threads = 0;

    //dr_printf("mode - %d\n",client_args.mode);
    
    if(client_args.mode == SNOOP){
      /* open filenames_file and mmap it */
      strcpy(filenames_file.filename,FILENAMES_FILE);
      filenames_file.filled = 0;
      filenames_file.offs = 0;
      filenames_file.file = dr_open_file(filenames_file.filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
      create_memory_map_file(&filenames_file,sizeof(thread_files_t));
      memset(filenames_file.data,0,sizeof(thread_files_t));
    }
    else if(client_args.mode == SQLITE){
      connection_init();
    }
  
}
