#ifndef TIMING_MMAP
#define TIMING_MMAP

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */
#include "common.h"
/*
  store the names of files that will have data in FILENAME_FILES 
  (2 files per thread, one for code and time). The snooping application 
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

/*
  convention of stored data in mapped file
  scratch space for book keeping - 4 bytes * BK_SLOTS
  timeslots per BB - 4bytes * TIME_SLOTS
 */


typedef struct{
  uint32_t control;
  char module[MAX_MODULE_SIZE];
  void * module_start; 
  char code[MAX_CODE_SIZE];
  uint32_t code_size;
  uint32_t rel_addr; 
}code_info_t;


typedef struct{
  uint32_t control; 
  uint32_t dump_bb; //if a basic block needs to be dumped
  uint32_t prevtime;
  uint32_t nowtime;
  uint32_t num_bbs;
  uint32_t overhead;
  uint32_t arch;
  mmap_file_t * mmap_raw_file;
} bookkeep_t;

typedef struct{
  uint32_t slots_filled;
  uint32_t rel_addr;
  void * module_start;
} bb_metadata_t;

typedef char query_t;

typedef struct{
  uint32_t count;
  uint32_t average;
}timing_t;

#define NUM_BBS 40000
#define TIME_SLOTS 96
#define BK_SLOTS 16

#define QUERY_SIZE sizeof(char) * MAX_QUERY_SIZE
#define TOTAL_SIZE  QUERY_SIZE + sizeof(code_info_t) + sizeof(uint32_t) * ( BK_SLOTS + NUM_BBS * TIME_SLOTS ) 
#define BB_DATA_SIZE sizeof(uint32_t) * TIME_SLOTS 

#define START_QUERY 0
#define START_CODE_DATA QUERY_SIZE
#define START_BK_DATA QUERY_SIZE + sizeof(code_info_t)
#define START_BB_DATA QUERY_SIZE + sizeof(code_info_t) + sizeof(uint32_t) * BK_SLOTS

#define METADATA_SLOTS sizeof(bb_metadata_t) / sizeof(uint32_t)
#define TIMING_SLOTS (TIME_SLOTS - METADATA_SLOTS)/(sizeof(timing_t)/sizeof(uint32_t))

typedef struct{
  bb_metadata_t meta;
  timing_t times[TIMING_SLOTS];
}bb_data_t;

#endif
