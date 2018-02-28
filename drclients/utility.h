#ifndef CMODEL_UTILITY
#define CMODEL_UTILITY

#include <stddef.h> /* for offsetof */
#include "dr_api.h"
#include "mmap.h"

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */

int get_filename(void * drcontext, char * filename, size_t max_size);
void insert_code(void * drcontext, volatile code_info_t * cinfo, instrlist_t * bb);

#endif
