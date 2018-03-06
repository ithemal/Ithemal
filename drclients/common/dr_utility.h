#ifndef COMMON_DRUTILITY
#define COMMON_DRUTILITY

#include "common.h"
#include "dr_api.h"
#include <stdint.h>

void create_memory_map_file(mmap_file_t * file_map, size_t size);
void close_memory_map_file(mmap_file_t * file_map, size_t size);

void create_raw_file(void * drcontext,mmap_file_t * file);
void write_to_file(mmap_file_t * file, void * values, uint32_t size);

#endif
