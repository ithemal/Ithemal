#ifndef TIMING_DUMP
#define TIMING_DUMP

#include <stdint.h>

int insert_config(char * query, const char * compiler, const char * flags, uint32_t mode);
int get_config(char * query, const char * compiler, const char * flags, uint32_t mode);
int insert_code(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode, uint32_t size);
int insert_times(char * query, const char * program, uint32_t rel_addr, uint32_t arch, uint32_t time, uint32_t count, uint32_t mode);
int complete_query(char * query, uint32_t size);


#endif
