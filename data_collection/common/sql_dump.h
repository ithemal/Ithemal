#ifndef COMMON_SQL_DUMP
#define COMMON_SQL_DUMP

#include "common.h"


int insert_config(query_t * query, config_t * config, uint32_t mode);

int insert_code(query_t * query, code_info_t * cinfo, uint32_t mode);
int update_code(query_t * query, code_info_t * cinfo, uint32_t mode);

int insert_times(query_t * query, code_info_t * cinfo, uint32_t time, uint32_t arch, uint32_t mode);
int complete_query(query_t * query, uint32_t size);


#endif
