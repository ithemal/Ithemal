#ifndef CMODEL_UTILITY
#define CMODEL_UTILITY

#include <stddef.h> /* for offsetof */
#include "dr_api.h"
#include "mode.h"

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */

typedef struct {
  char filename[MAX_MODULE_SIZE];
  file_t file;
  void * data;
} mmap_file_t;


int get_filename(void * drcontext, char * filename, size_t max_size);
void insert_code(void * drcontext, volatile code_info_t * cinfo, instrlist_t * bb);

void create_memory_map_file(mmap_file_t * file_map, size_t size);

#endif
