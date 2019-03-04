#include <string.h>
#include <stdio.h>
#include "sql_dump.h"
#include "common.h"

int insert_config(char * query, const char * compiler, const char * flags, uint32_t mode, uint32_t arch){
  int pos = 0;
  if(mode == SQLITE){
    pos += sprintf(query, "INSERT INTO config (compiler, flags, arch) VALUES ('%s','%s', %d);", compiler, flags, arch);
    pos += sprintf(query + pos,"CREATE TEMP TABLE _config (config_id INTEGER);");
    pos += sprintf(query + pos,"INSERT INTO _config (config_id) VALUES ((SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s' AND arch = %d))", compiler, flags, arch);
  }
  else{
    pos += sprintf(query, "INSERT INTO config (compiler, flags, arch) VALUES ('%s','%s',%d);", compiler, flags, arch);
    pos += sprintf(query + pos,"SET @config_id = (SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s' AND arch = %d)",compiler, flags, arch);
  }
  return pos;
}

int insert_code(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode,  uint32_t size, const char * type){

  uint32_t pos;
  if(mode == SQLITE){
    pos = sprintf(query, "INSERT INTO code (config_id, program,rel_addr, %s) VALUES ((SELECT config_id from _config),'%s',%d,'", type, program, rel_addr);
  }
  else{
    pos = sprintf(query, "INSERT INTO code (config_id, program,rel_addr, %s) VALUES (@config_id,'%s',%d,'", type, program, rel_addr);
  }
  
  int i = 0;
  for(i = 0; i < size; i++){
    query[pos + i] = code[i];
  } 

  pos += size;
  pos += sprintf(query + pos, "')");
  return pos;

}

int update_code(char * query, const char * program, uint32_t rel_addr, const char * code, uint32_t mode,  uint32_t size, const char * type){

  uint32_t pos;
  pos = sprintf(query, "UPDATE code SET %s='", type);
  
  int i = 0;
  for(i = 0; i < size; i++){
    query[pos + i] = code[i];
  } 
  pos += size;

  if(mode == SQLITE){
    pos += sprintf(query + pos, "' WHERE config_id=(SELECT config_id from _config) AND program='%s' AND rel_addr=%d",program, rel_addr);
  }
  else{
    pos += sprintf(query + pos, "' WHERE config_id=@config_id AND program='%s' AND rel_addr=%d",program, rel_addr);
  }
  
  return pos;

}

int insert_times(char * query, const char * program, uint32_t rel_addr, uint32_t arch, uint32_t time, uint32_t mode){
  if(mode == SQLITE){
    return sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = (SELECT config_id from _config) AND program = '%s' AND rel_addr = %d), '%d', %d)",program,rel_addr,arch, time);
  }
  else{
    return sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = @config_id AND program = '%s' AND rel_addr = %d), '%d', %d)",program,rel_addr,arch, time);
  }
}

int update_static_info(char * query, const char * program, uint32_t rel_addr, uint32_t mode, uint32_t num_instr,uint32_t span){

  int pos;
  if(mode == SQLITE){
    pos = sprintf(query, "UPDATE code SET num_instr = %d, span = %d WHERE config_id = (SELECT config_id from _config) AND program = '%s' AND rel_addr = %d",num_instr,span,program, rel_addr);
  }
  else{
    pos = sprintf(query, "UPDATE code SET num_instr = %d, span = %d WHERE config_id = @config_id AND program = '%s' AND rel_addr = %d",num_instr,span,program, rel_addr);
  }
  
  return pos;


}


int complete_query(char * query, uint32_t size){
  query[size] = ';';
  query[size + 1] = '\n';
  return size + 2;
}
