#ifndef COMMON_MMAP_FILE
#define COMMON_MMAP_FILE

//for snooping - store filenames

/*
  store the names of files that will have data in FILENAME_FILES. The snooping application 
  will check all files for ready data to be written to the database.
  All timing and code data will be written to the mmapped files, not to
  thread local structures maintained by DR, so that snooping application will
  also have access to it.
 */

#include "common.h"

//snooping control modes
#define IDLE       0
#define DR_CONTROL 1
#define DUMP_ONE   2
#define DUMP_ALL   3
#define EXIT       4

#define FILENAMES_FILE "/tmp/names.txt"

#define MAX_THREADS 16

typedef struct{
  uint32_t control;
  uint32_t num_modules;
  char modules[MAX_THREADS][MAX_MODULE_SIZE];
} thread_files_t;

//mmap file structure
typedef struct{
char filename[MAX_MODULE_SIZE];
file_t file;
void * data;
uint64_t offs;
uint32_t filled;
} mmap_file_t;

/*
  convention of stored data in mapped file
  scratch space for book keeping - 4 bytes * BK_SLOTS
 */

typedef struct{
  uint32_t control; 
  uint32_t dump_bb; //if a basic block needs to be dumped
  uint32_t num_bbs;
  uint32_t arch;
  mmap_file_t * static_file;
  uint32_t exited;
} bookkeep_t;


#define BK_SLOTS 16

#define QUERY_SIZE sizeof(char) * MAX_QUERY_SIZE
#define TOTAL_SIZE  QUERY_SIZE + sizeof(code_info_t) + sizeof(uint32_t) * BK_SLOTS

#define START_QUERY 0
#define START_CODE_DATA QUERY_SIZE
#define START_BK_DATA QUERY_SIZE + sizeof(code_info_t)


#endif
