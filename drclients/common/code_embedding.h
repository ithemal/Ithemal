#ifndef COMMON_CODE_EMBEDDING
#define COMMON_CODE_EMBEDDING

#include "dr_api.h"
#include "common.h"
#include <stdint.h>

//code embedding structure
typedef struct{
  uint32_t control;
  char module[MAX_MODULE_SIZE];
  void * module_start; 
  char code[MAX_CODE_SIZE];
  int32_t code_size;
  uint32_t rel_addr; 
  uint32_t num_instr;
  uint32_t span;
}code_info_t;


typedef void (*code_embedding_t) (void *, code_info_t *, instrlist_t *);

void textual_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);

void token_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);
void token_text_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb);



#endif
