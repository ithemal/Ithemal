#include <stddef.h> /* for offsetof */
#include "dr_api.h"

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */

#include "mmap.h"
#include "utility.h"

void post_cleancall();

#define MAX_STRING_SIZE 100

//client arguments
typedef struct {
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
} client_arg_t;

typedef struct {
  char filename[MAX_MODULE_SIZE];
  file_t file;
  void * data;
} mmap_file_t;

client_arg_t client_args;
void * mutex;
mmap_file_t filenames_file;


static void
event_exit(void);
static void thread_exit(void * drcontext);

void create_memory_map_file(mmap_file_t * file_map, size_t size){
  
    file_map->file = dr_open_file(file_map->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
    DR_ASSERT(file_map->file);

    DR_ASSERT(dr_file_seek(file_map->file,size-1,DR_SEEK_SET));
    DR_ASSERT(dr_write_file(file_map->file,"",1));

    file_map->data = dr_map_file(file_map->file, &size, 0, NULL, DR_MEMPROT_READ | DR_MEMPROT_WRITE, 0);
    DR_ASSERT(file_map->data);
 
}

void destroy_memory_map_file(mmap_file_t * file_map){
  
} 

    /* thread init event */
static void 
thread_init(void * drcontext){

  mmap_file_t * data = (mmap_file_t *)dr_thread_alloc(drcontext, sizeof(mmap_file_t));    
  dr_set_tls_field(drcontext,data);
  
  get_filename(drcontext,data->filename,MAX_MODULE_SIZE);
  
  dr_printf("%s\n",data->filename);

  //write to filenames file
  dr_mutex_lock(mutex);
  
  volatile thread_files_t * files = filenames_file.data;
  DR_ASSERT(files); //should be memory mapped 
  while(!__sync_bool_compare_and_swap(&files->control,IDLE,DR_CONTROL)){} //dr should be in control

  strcpy(files->modules[files->num_modules],data->filename);
  files->num_modules++;
 
  DR_ASSERT(__sync_bool_compare_and_swap(&files->control,DR_CONTROL,DUMP_ONE));

  while(files->control != IDLE){}

  dr_mutex_unlock(mutex);  
 
  //create the file and memory map it
  create_memory_map_file(data,TOTAL_SIZE);
  memset(data->data,0,TOTAL_SIZE);
  
}


static dr_emit_flags_t
bb_creation_event(void * drcontext, void * tag, instrlist_t * bb, bool for_trace, bool translating){
  return DR_EMIT_DEFAULT;
 
}

void post_cleancall(uint32_t num_bbs){


}

  /* thread exit event */
static void
thread_exit(void * drcontext){

  mmap_file_t * data = dr_get_tls_field(drcontext);
  dr_unmap_file(data->data,TOTAL_SIZE);
  dr_close_file(data->file);
}

static void
event_exit(void)
{

  dr_mutex_destroy(mutex);
  volatile thread_files_t * files = filenames_file.data;
  files->control = EXIT;
  while(files->control != IDLE);
  dr_unmap_file(filenames_file.data,sizeof(thread_files_t));
  dr_close_file(filenames_file.file);
}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    
    /* register events */
    
    dr_register_thread_init_event(thread_init);
    
    dr_register_bb_event(bb_creation_event);
    
    dr_register_thread_exit_event(thread_exit);
    dr_register_exit_event(event_exit);
    

    disassemble_set_syntax(DR_DISASM_INTEL);

    /* client arguments */
    DR_ASSERT(argc == 3);
    strncpy(client_args.compiler,argv[1], MAX_STRING_SIZE);
    strncpy(client_args.flags,argv[2], MAX_STRING_SIZE);

    /* open filenames_file and mmap it */
    strcpy(filenames_file.filename,FILENAMES_FILE);
    create_memory_map_file(&filenames_file,sizeof(thread_files_t));
    memset(filenames_file.data,0,sizeof(thread_files_t));
    
    mutex = dr_mutex_create();
  
}
