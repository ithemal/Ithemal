#ifndef STATIC_LOGIC
#define STATIC_LOGIC

#include "dr_api.h"
#include <stdint.h>

int num_instructions(instrlist_t * bb);
int span_bb(instrlist_t * bb);

#endif
