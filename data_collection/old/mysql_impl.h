#ifndef SNOOP_MYSQL
#define SNOOP_MYSQL

#include <stdint.h>

void connection_init();
void connection_close();
void insert_config(const char * compiler, const char * flags);
void insert_code(const char * program, uint32_t rel_addr, const char * code);
void insert_times(const char * program, uint32_t rel_addr, uint32_t  arch, uint32_t time);

#endif
