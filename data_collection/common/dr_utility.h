#ifndef COMMON_DRUTILITY
#define COMMON_DRUTILITY

#include "common.h"
#include "dr_api.h"
#include "mmap.h"
#include <stdint.h>

int get_perthread_filename(void * drcontext, char * filename, size_t max_size);
uint32_t filter_based_on_module(const char * module_name);

void create_memory_map_file(mmap_file_t * file_map, size_t size);
void close_memory_map_file(mmap_file_t * file_map, size_t size);

void create_raw_file(void * drcontext, const char * folder, const char * type, mmap_file_t * file);
void write_to_file(mmap_file_t * file, void * values, uint32_t size);
void close_raw_file(mmap_file_t * file);

#endif
