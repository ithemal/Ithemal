#include <stddef.h> /* for offsetof */
#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */
#include <stdlib.h>

#include "common.h"
#include "code_embedding.h"
#include "sql_dump_v2.h"
#include "client.h"
#include "mmap.h"
#include "dr_utility.h"

client_arg_t client_args;
config_t config;
mmap_file_t filenames_file;
void * mutex;


void post_cleancall();

//thread init event
static void 
thread_init(void * drcontext){

  mmap_file_t * data = (mmap_file_t *)dr_thread_alloc(drcontext, sizeof(mmap_file_t));      
  dr_set_tls_field(drcontext,data);
  
  get_perthread_filename(drcontext,data->filename,MAX_MODULE_SIZE);
  data->offs = data->filled = 0;
  
  //create the file and memory map it  
  data->file = dr_open_file(data->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
  create_memory_map_file(data,TOTAL_SIZE);
  memset(data->data,0,TOTAL_SIZE);

  //processor model information
  bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
  bk->exited = 0;
  bk->arch = proc_get_model();    
  bk->static_file = dr_thread_alloc(drcontext, sizeof(mmap_file_t));
  create_raw_file(drcontext,client_args.data_folder,"static",bk->static_file);
  
  //insert the config string (query)
  query_t * query = (query_t *)(data->data + START_QUERY);
  
  int sz = insert_architecture(query, &config);
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  write_to_file(bk->static_file,query,sz);
  sz = insert_config(query, &config);
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  write_to_file(bk->static_file,query,sz);
  
}

bool dump_bb(void * drcontext, code_embedding_t embedding_func, code_info_t * cinfo, instrlist_t * bb, bookkeep_t * bk, query_t * query){

  int sz = -1;

  //get the embedding
  embedding_func(drcontext, cinfo, bb);

  if(cinfo->code_size == -1)
    return false;
  
  sz = insert_code(query, cinfo);
  if(sz == -1)
    return false;
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  write_to_file(bk->static_file,query,sz);

  sz = insert_code_metadata(query, cinfo);
  if(sz == -1)
    return false;  
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  write_to_file(bk->static_file,query,sz);


  return true;

} 

bool dump_disasm(void * drcontext, code_embedding_t embedding_func, code_info_t * cinfo, instrlist_t * bb, bookkeep_t * bk, query_t * query, uint32_t type){
  
  int sz = -1;

  embedding_func(drcontext, cinfo, bb);

  if(cinfo->code_size == -1)
    return false;
  
  sz = insert_disassembly(query, cinfo, type);
  if(sz == -1)
    return false;
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  write_to_file(bk->static_file, query, sz);

  return true;
  


}

//bb analysis routines
bool populate_bb_info(void * drcontext, code_info_t * cinfo, instrlist_t * bb, bookkeep_t * bk, query_t * query){

  instr_t * first = instrlist_first(bb);
  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);

  
  if(md){
    strcpy(cinfo->module,dr_module_preferred_name(md));
    cinfo->module_start = md->start;
    if(client_args.abs_addr == 0)
      cinfo->rel_addr = first_pc - md->start;
    else
      cinfo->rel_addr = first_pc;
  }
  else{
    strcpy(cinfo->module,"generated");
    cinfo->module_start = 0;
    cinfo->rel_addr = 0;
  }
  
  if(!dump_bb(drcontext, raw_bits, cinfo, bb, bk, query)){
    return false;
  }  

  disassemble_set_syntax(DR_DISASM_INTEL);
  if(!dump_disasm(drcontext, text_intel, cinfo, bb, bk, query, CODE_INTEL)){
    return false;
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

  query_t * query = (query_t *)(file->data + START_QUERY);
  bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);
  code_info_t * cinfo = (code_info_t *)(file->data + START_CODE_DATA);

  if(!populate_bb_info(drcontext,cinfo,bb,bk,query)){
    return DR_EMIT_DEFAULT;
  }
 
 
  bk->num_bbs++;
  return DR_EMIT_DEFAULT;
 
}

  /* thread exit event */
static void
thread_exit(void * drcontext){
  
  mmap_file_t * file = dr_get_tls_field(drcontext);
  bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);  
  
  if(bk->exited) 
    return;
  bk->exited = 1;

  close_raw_file(bk->static_file);
  close_memory_map_file(file,TOTAL_SIZE);
 
}

static void
event_exit(void)
{
  dr_mutex_destroy(mutex);
}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{

  dr_register_thread_init_event(thread_init);    
  dr_register_bb_event(bb_creation_event);
  dr_register_thread_exit_event(thread_exit);
  dr_register_exit_event(event_exit);
  
  DR_ASSERT(argc == 7);
  strncpy(config.compiler,argv[1], MAX_STRING_SIZE);
  strncpy(config.flags,argv[2], MAX_STRING_SIZE);
  strncpy(config.name, argv[3], MAX_STRING_SIZE);
  strncpy(config.vendor, argv[4], MAX_STRING_SIZE);
  strncpy(client_args.data_folder,argv[5], MAX_STRING_SIZE);
  client_args.abs_addr = (uint32_t)atoi(argv[6]);

  mutex = dr_mutex_create();
  
}
