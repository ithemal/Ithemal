#ifndef COMMON_CODE_EMBEDDING
#define COMMON_CODE_EMBEDDING

#include "dr_api.h"
#include "common.h"
#include <stdint.h>

typedef bool (*code_embedding_t) (void *, code_info_t *, instrlist_t *);

bool raw_token(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
bool text_token(void * drcontext, code_info_t * cinfo, instrlist_t * bb);

bool text_xml(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
bool text_intel(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
bool text_att(void * drcontext, code_info_t * cinfo, instrlist_t * bb);

bool raw_bits(void * drcontext, code_info_t * cinfo, instrlist_t * bb);



#endif
