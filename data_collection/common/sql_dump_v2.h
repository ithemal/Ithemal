#ifndef COMMON_SQL_DUMP_V2
#define COMMON_SQL_DUMP_V2

#include "common.h"

int insert_architecture(query_t * query, config_t * config);
int insert_config(query_t * query, config_t * config);

int insert_code(query_t * query, code_info_t * cinfo);
int insert_code_metadata(query_t * query, code_info_t * cinfo);
int insert_disassembly(query_t * query, code_info_t * cinfo, uint32_t type);

#endif
