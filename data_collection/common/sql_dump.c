#include <string.h>
#include <stdio.h>
#include "sql_dump.h"


const char * code_types[3] = {"code_intel", "code_att", "code_token"};


int insert_config(query_t * query, config_t * config, uint32_t mode){
  int pos = 0;
  if(mode == SQLITE){
    pos += sprintf(query, "INSERT INTO config (compiler, flags, arch) VALUES ('%s','%s', %d);", config->compiler, config->flags, config->arch);
    pos += sprintf(query + pos,"CREATE TEMP TABLE _config (config_id INTEGER);");
    pos += sprintf(query + pos,"INSERT INTO _config (config_id) VALUES ((SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s' AND arch = %d))", config->compiler, config->flags, config->arch);
  }
  else{
    pos += sprintf(query, "INSERT INTO config (compiler, flags, arch) VALUES ('%s','%s',%d);", config->compiler, config->flags, config->arch);
    pos += sprintf(query + pos,"SET @config_id = (SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s' AND arch = %d)", config->compiler, config->flags, config->arch);
  }
  return pos;
}

int insert_code(query_t * query, code_info_t * cinfo, uint32_t mode){

  uint32_t pos;
  if(mode == SQLITE){
    pos = sprintf(query, "INSERT INTO code (config_id, program,rel_addr, %s) VALUES ((SELECT config_id from _config),'%s',%d,'", code_types[cinfo->code_type], cinfo->module, cinfo->rel_addr);
  }
  else{
    pos = sprintf(query, "INSERT INTO code (config_id, program,rel_addr, %s) VALUES (@config_id,'%s',%d,'", code_types[cinfo->code_type], cinfo->module, cinfo->rel_addr);
  }
  
  int i = 0;
  for(i = 0; i < cinfo->code_size; i++){
    query[pos + i] = cinfo->code[i];
  } 

  pos += cinfo->code_size;
  pos += sprintf(query + pos, "')");
  return pos;

}

int update_code(query_t * query, code_info_t * cinfo,  uint32_t mode){

  uint32_t pos;
  pos = sprintf(query, "UPDATE code SET %s='", code_types[cinfo->code_type]);
  
  int i = 0;
  for(i = 0; i < cinfo->code_size; i++){
    query[pos + i] = cinfo->code[i];
  } 
  pos += cinfo->code_size;

  if(mode == SQLITE){
    pos += sprintf(query + pos, "' WHERE config_id=(SELECT config_id from _config) AND program='%s' AND rel_addr=%d", cinfo->module, cinfo->rel_addr);
  }
  else{
    pos += sprintf(query + pos, "' WHERE config_id=@config_id AND program='%s' AND rel_addr=%d", cinfo->module, cinfo->rel_addr);
  }
  
  return pos;

}

int insert_times(query_t * query, code_info_t * cinfo, uint32_t time, uint32_t arch, uint32_t mode){

  if(mode == SQLITE){
    return sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = (SELECT config_id from _config) AND program = '%s' AND rel_addr = %d), '%d', %d)", cinfo->module, cinfo->rel_addr, arch, time);
  }
  else{
    return sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = @config_id AND program = '%s' AND rel_addr = %d), '%d', %d)", cinfo->module, cinfo->rel_addr, arch, time);
  }
}

int complete_query(char * query, uint32_t size){
  query[size] = ';';
  query[size + 1] = '\n';
  return size + 2;
}
