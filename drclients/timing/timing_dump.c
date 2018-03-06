#include <string.h>
#include <stdio.h>
#include "timing_dump.h"
#include "common.h"

int insert_config(char * query, const char * compiler, const char * flags, uint32_t mode){
  int pos = 0;
  if(mode == SQLITE){
    pos += sprintf(query, "INSERT INTO config (compiler, flags) VALUES ('%s','%s');", compiler, flags);
    pos += sprintf(query + pos,"CREATE TEMP TABLE _config (config_id INTEGER);");
    pos += sprintf(query + pos,"INSERT INTO _config (config_id) VALUES ((SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s'))", compiler, flags);
  }
  else{
    pos += sprintf(query, "INSERT INTO config (compiler, flags) VALUES ('%s','%s');", compiler, flags);
    pos += sprintf(query + pos,"SET @config_id = (SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s')",compiler,flags);
  }
  return pos;
}

int insert_code(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode){
  if(mode == SQLITE){
    return sprintf(query, "INSERT INTO code (config_id, program,rel_addr, code) VALUES ((SELECT config_id from _config),'%s',%d,'%s')",program, rel_addr, code);
  }
  else{
    return sprintf(query, "INSERT INTO code (config_id, program,rel_addr, code) VALUES (@config_id,'%s',%d,'%s')",program, rel_addr, code);
  }
}

int insert_times(char * query, const char * program, uint32_t rel_addr, uint32_t arch, uint32_t time, uint32_t mode){
  if(mode == SQLITE){
    return sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = (SELECT config_id from _config) AND program = '%s' AND rel_addr = %d), '%d', %d)",program,rel_addr,arch, time);
  }
  else{
    return sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = @config_id AND program = '%s' AND rel_addr = %d), '%d', %d)",program,rel_addr,arch, time);
  }
}


int complete_query(char * query, uint32_t size){
  query[size] = ';';
  query[size + 1] = '\n';
  return size + 2;
}
