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
#define MAX_CODE_SIZE 20480
#define MAX_QUERY_SIZE 20480

//for snooping - store filenames

/*
  store the names of files that will have data in FILENAME_FILES. The snooping application 
  will check all files for ready data to be written to the database.
  All timing and code data will be written to the mmapped files, not to
  thread local structures maintained by DR, so that snooping application will
  also have access to it.
 */

#define FILENAMES_FILE "/tmp/names.txt"

#define MAX_THREADS 16

typedef struct{
  uint32_t control;
  uint32_t num_modules;
  char modules[MAX_THREADS][MAX_MODULE_SIZE];
} thread_files_t;


//tokenizing code
#define REG_START 0
#define INT_IMMED REG_START + DR_REG_YMM15 + 1
#define FLOAT_IMMED INT_IMMED + 1
#define OPCODE_START FLOAT_IMMED + 1
#define MEMORY_START OPCODE_START + OP_LAST + 1

//mmap file structure
typedef struct{
char filename[MAX_MODULE_SIZE];
file_t file;
void * data;
uint64_t offs;
uint32_t filled;
}mmap_file_t;



#endif
