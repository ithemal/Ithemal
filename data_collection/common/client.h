#ifndef COMMON_CLIENT
#define COMMON_CLIENT

#include <stdint.h>

//client arguments
typedef struct {
  char data_folder[MAX_STRING_SIZE];
  uint32_t abs_addr;
} client_arg_t;

#endif
