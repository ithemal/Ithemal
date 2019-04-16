#ifndef COMMON_COMMON
#define COMMON_COMMON

#include "dr_api.h"
#include <stdint.h>

//global character count constants
#define MAX_STRING_SIZE 128   //this is for generic strings
#define MAX_MODULE_SIZE 1024
#define MAX_CODE_SIZE 1000000
#define MAX_QUERY_SIZE 102000

typedef struct{
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
  char name[MAX_STRING_SIZE];
  char vendor[MAX_STRING_SIZE];
} config_t;

//code embedding structure
typedef struct{
  uint32_t control;
  char module[MAX_MODULE_SIZE];
  void * module_start;
  char function[MAX_MODULE_SIZE];
  uint32_t code_type;
  unsigned char code[MAX_CODE_SIZE];
  int32_t code_size;
  uint64_t rel_addr;
  uint32_t num_instr;
  uint32_t span;
} code_info_t;


typedef unsigned char query_t;

//code type
#define CODE_TOKEN 0
#define CODE_INTEL 1
#define CODE_ATT   2


#endif
