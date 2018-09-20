#ifndef COMMON_SQLITE3
#define COMMON_SQLITE3

void connection_init();
void connection_close();
void query_db(const char * query);

#endif
