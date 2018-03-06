#include <stdio.h>
#include <string.h>
#include <sqlite3.h>
#include <stdlib.h>
#include "sqlite3_impl.h"

sqlite3 * db;

void connection_init(){

  int rc = sqlite3_open("costmodel.db", &db);
  if (rc != SQLITE_OK) {
        
    fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
        
    exit(1);
  }  
   
}

void connection_close(){
  sqlite3_close(db);
}

void query_db(const char * query){

  char * err_msg = 0;
  int rc = sqlite3_exec(db, query, 0, 0, &err_msg);

  if (rc != SQLITE_OK ) {       
    fprintf(stderr, "SQL error: %s\n", err_msg);
    sqlite3_free(err_msg);                
  } 
}
