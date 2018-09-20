#ifndef STATIC_MMAP
#define STATIC_MMAP

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */
#include "common.h"
#include "code_embedding.h"

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
} bookkeep_t;

typedef char query_t;

#define BK_SLOTS 16

#define QUERY_SIZE sizeof(char) * MAX_QUERY_SIZE
#define TOTAL_SIZE  QUERY_SIZE + sizeof(code_info_t) + sizeof(uint32_t) * BK_SLOTS

#define START_QUERY 0
#define START_CODE_DATA QUERY_SIZE
#define START_BK_DATA QUERY_SIZE + sizeof(code_info_t)

#endif
