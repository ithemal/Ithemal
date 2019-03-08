#ifndef COMMON_CODE_EMBEDDING
#define COMMON_CODE_EMBEDDING

#include "dr_api.h"
#include "common.h"
#include <stdint.h>

typedef bool (*code_embedding_t) (void *, code_info_t *, instrlist_t *);

bool textual_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
bool textual_embedding_with_size(void * drcontext, code_info_t * cinfo, instrlist_t * bb);

bool token_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
bool token_text_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);



#endif
