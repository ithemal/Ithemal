#include <stdio.h>
#include <string.h>
#include <mysql.h>
#include <my_global.h>

#include "mysql_impl.h"

MYSQL * con;

void error(int e,MYSQL *con)
{
  //fprintf(stderr, "%d,%s\n", e, mysql_error(con));
}

void connection_init(){
  con = mysql_init(NULL);
  if(con == NULL){
    fprintf(stderr, "%s\n", mysql_error(con));
    exit(1);
  }
  int e = 0;
  if (e = mysql_real_connect(con, "localhost", "root", "mysql7788#", 
			 "costmodel", 43562, "/data/scratch/charithm/libraries/install/mysql/mysqld.sock", 0) == NULL) 
    {
      error(e,con);
    }   
}

void connection_close(){
  mysql_close(con);
}

void query_db(const char * query){
  int e = 0;
  if(e = mysql_query(con,query)){
    error(e,con);
  }
}

void insert_config(const char * compiler, const char * flags){
  sprintf(query, "INSERT INTO config (compiler, flags) VALUES ('%s','%s')", compiler, flags);
  query_db();
  sprintf(query,"SET @config_id = (SELECT config_id FROM config WHERE compiler = '%s' AND flags = '%s')",compiler,flags);
  query_db();
}

void insert_code(const char * program, uint32_t rel_addr, const char * code){
  sprintf(query, "INSERT INTO code (config_id, program,rel_addr, code) VALUES (@config_id,'%s',%d,'%s')",program, rel_addr, code);
  query_db();
}

void insert_times(const char * program, uint32_t rel_addr, uint32_t arch, uint32_t time){
  sprintf(query, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = @config_id AND program = '%s' AND rel_addr = %d), '%d', %d)",program,rel_addr,arch, time);
  query_db();
}
