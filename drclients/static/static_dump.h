#ifndef STATIC_DUMP
#define STATIC_DUMP

#include <stdint.h>

int insert_config(char * query, const char * compiler, const char * flags, uint32_t mode);
int insert_code_token(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode, uint32_t size);
int insert_code_text(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode, uint32_t size);
int insert_times(char * query, const char * program, uint32_t rel_addr, uint32_t arch, uint32_t time, uint32_t mode);
int complete_query(char * query, uint32_t size);


#endif
