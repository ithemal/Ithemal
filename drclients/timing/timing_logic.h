#ifndef TIMING_LOGIC
#define TIMING_LOGIC

#include "dr_api.h"
#include "timing_mmap.h"
#include <stdint.h>

typedef void (*code_embedding_t) (void *, code_info_t *, instrlist_t *);

int get_filename(void * drcontext, char * filename, size_t max_size);
void textual_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
void token_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
void insert_timing(bb_data_t * bb, uint32_t time);
uint32_t filter_based_on_exec(const char * module_name);

#endif
