#ifndef COMMON_COMMON
#define COMMON_COMMON

#include "dr_api.h"
#include <stdint.h>

//dr client data collection modes
#define SNOOP   1
#define SQLITE  2
#define RAW_SQL 3

//code output
#define TEXT   1
#define TOKEN  2

//snooping control modes
#define IDLE       0
#define DR_CONTROL 1
#define DUMP_ONE   2
#define DUMP_ALL   3
#define EXIT       4

//global character count constants
#define MAX_STRING_SIZE 128   //this is for generic strings
#define MAX_MODULE_SIZE 128
#define MAX_CODE_SIZE 1024
#define MAX_QUERY_SIZE 2048

//mmap file structure
typedef struct{
char filename[MAX_MODULE_SIZE];
file_t file;
void * data;
uint64_t offs;
uint32_t filled;
}mmap_file_t;



#endif
