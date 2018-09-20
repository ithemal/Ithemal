#ifndef COMMON_MYSQL
#define COMMON_MYSQL

void connection_init();
void connection_close();
void query_db(const char * query);

#endif
