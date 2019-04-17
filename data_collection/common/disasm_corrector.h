#ifndef COMMON_DISASM_CORRECTOR
#define COMMON_DISASM_CORRECTOR

#include "common.h"

//operand types
#define MEM_TYPE 1
#define IMM_TYPE 2
#define REG_TYPE 3

#define MNEMONIC_SIZE 64
#define NUM_OPS 8
#define BUFFER_SIZE 1024


typedef struct {
  uint32_t type;
  char name[MNEMONIC_SIZE];
} op_t;

typedef struct {
  op_t operands[NUM_OPS];
  char name[MNEMONIC_SIZE];
  int num_ops;
} ins_t;


bool parse_instr_att(char * buffer, int length, ins_t * instr);
void correct_disasm_att(void *drcontext, ins_t * ins, instr_t * instr);




#endif
