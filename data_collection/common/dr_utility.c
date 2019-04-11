#include "dr_utility.h"
#include <string.h>

//format - application_threadid_hour_min
int get_perthread_filename(void * drcontext, char * filename, size_t max_size){

    thread_id_t id = dr_get_thread_id(drcontext);
    dr_time_t time;
    dr_get_time(&time);
    return dr_snprintf(filename, max_size, "/tmp/%s_%d_%d_%d.txt", dr_get_application_name(), id, time.hour, time.minute);

}


//creating memory mapped files
void create_memory_map_file(mmap_file_t * file_map, size_t size){
  
  //file_map->file = dr_open_file(file_map->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
    DR_ASSERT(file_map->file);

    DR_ASSERT(dr_file_seek(file_map->file, file_map->offs + size-1,DR_SEEK_SET));
    DR_ASSERT(dr_write_file(file_map->file,"",1));

    file_map->data = dr_map_file(file_map->file, &size, file_map->offs, NULL, DR_MEMPROT_READ | DR_MEMPROT_WRITE, 0);
    DR_ASSERT(file_map->data);
 
}

void close_memory_map_file(mmap_file_t * file_map, size_t size){
  
  dr_close_file(file_map->file);
  dr_unmap_file(file_map->data,size);
  
}


//raw writing to a file - for appending data to a file
#define NUM_PAGES 4

static int get_raw_filename(void * drcontext, const char * folder, const char *type, char * filename, size_t max_size){

    thread_id_t id = dr_get_thread_id(drcontext);
    dr_time_t time;
    dr_get_time(&time);

    return dr_snprintf(filename, max_size, "%s/%s_%s_%d_%d_%d.sql", folder, type, dr_get_application_name(), id, time.hour, time.minute);

}


void create_raw_file(void * drcontext, const char * folder, const char * type,  mmap_file_t * file){
 
  size_t page = dr_page_size();
  
  //obtain the filename
  get_raw_filename(drcontext,folder,type,file->filename,NUM_PAGES * page);
  file->file = dr_open_file(file->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
  file->offs = 0;
  file->filled = 0; 
  create_memory_map_file(file,NUM_PAGES * page);

}

void close_raw_file(mmap_file_t * file){
 
  size_t page = dr_page_size();

  int i = 0;
  char * data = file->data;

  for(i = file->filled; i < NUM_PAGES * page; i++){
    data[i] = ' ';
  }

  dr_close_file(file->file);
  dr_unmap_file(file->data,NUM_PAGES * page);
  

}


void flush_and_load(mmap_file_t * file){

  size_t page = dr_page_size();
  size_t total_size = NUM_PAGES * page;

  int i = 0;
  char * data = file->data;
  for(i = file->filled; i < total_size; i++){
    data[i] = ' '; //fill with spaces
  }

  //close_memory_map_file(file, total_size);
  dr_unmap_file(file->data,total_size);

  file->offs += total_size;
  create_memory_map_file(file, total_size);
  file->filled = 0;

}

void write_to_file(mmap_file_t * file, void * values, uint32_t size){

  size_t page = dr_page_size();
  size_t total_size = NUM_PAGES * page;
  
  if(file->filled + size > total_size){
    flush_and_load(file);
  }

  //dr_printf("%s\n",(char *)values);
  memcpy(file->data + file->filled, values, size);
  //dr_printf("%s\n",(char *)(file->data + file->filled));
  file->filled += size;
  
}

uint32_t filter_based_on_module(const char * module_name){

  return strcmp(dr_get_application_name(),module_name) == 0;

}






