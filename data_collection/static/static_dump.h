#ifndef STATIC_DUMP
#define STATIC_DUMP

#include <stdint.h>

int insert_config(char * query, const char * compiler, const char * flags, uint32_t mode, uint32_t arch);
int insert_code(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode, uint32_t size, const char * type);
int update_code(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode, uint32_t size, const char * type);

int insert_times(char * query, const char * program, uint32_t rel_addr, uint32_t arch, uint32_t time, uint32_t mode);
int complete_query(char * query, uint32_t size);


#endif
