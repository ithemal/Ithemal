#include "dr_api.h"
#include <stdlib.h> /* for realloc */
#include <assert.h>
#include <stddef.h> /* for offsetof */
#include <string.h> /* for memcpy */
#include <inttypes.h>
#include <unistd.h>
#include <stdint.h>

#include "common.h"
#include "client.h"
#include "code_embedding.h"

#define DEFAULT 0
#define TOKEN 1
#define ATT 2
#define INTEL 3
#define RAW 4

void print_intel(void * drcontext, instrlist_t * bb){
  int i = 0;
  instr_t* instr;
  disassemble_set_syntax(DR_DISASM_INTEL);

  for (instr = instrlist_first(bb); instr_get_next(instr) != NULL; instr = instr_get_next(instr)){
    instr_disassemble(drcontext, instr, STDOUT);
    dr_printf("\n");
  }
}


void print_att(void * drcontext, instrlist_t * bb){
  int i = 0;
  instr_t* instr;
  disassemble_set_syntax(DR_DISASM_ATT);

  for (instr = instrlist_first(bb); instr_get_next(instr) != NULL; instr = instr_get_next(instr)){
    instr_disassemble(drcontext, instr, STDOUT);
    dr_printf("\n");
  }
}


bool tokenize(void * drcontext, instrlist_t * bb, int output_typ) {
  //create the dump related data structures
  code_info_t cinfo;
  bool success = false;

  switch (output_typ) {
  case TOKEN:
    success = text_xml(drcontext, &cinfo, bb);
    break;
  case ATT:
    success = text_att(drcontext, &cinfo, bb);
    break;
  case INTEL:
    success = text_intel(drcontext, &cinfo, bb);
    break;
  }

  if (!success) return false;

  printf("%s\n", cinfo.code);
  return true;
}

instrlist_t * decode_instrs(void * drcontext, byte * raw, int len){

  unsigned char* start_pc = raw;
  unsigned char* end_pc = start_pc + len;
  instrlist_t* current_list = instrlist_create(drcontext);

  while(start_pc < end_pc){
    instr_t * instr = instr_create(drcontext);
    start_pc = decode(drcontext, start_pc, instr);
    instrlist_append(current_list, instr);
    if (!start_pc) {
      return NULL;
    }
  }

  //dummy instr to match tokenize interface
  instrlist_append(current_list, INSTR_CREATE_cpuid(drcontext));

  return current_list;
}


bool print_instr_bytes(void * drcontext, byte * raw, int len){
  unsigned char* start_pc = raw;
  unsigned char* end_pc = start_pc + len;

  while(start_pc < end_pc){
    instr_t * instr = instr_create(drcontext);
    unsigned char *prev_start = start_pc;
    start_pc = decode(drcontext, start_pc, instr);
    if (!start_pc) {
      return false;
    }
    for (unsigned char* i = prev_start; i < start_pc && i < end_pc; i++) {
      printf("%02x", *i);
    }
    printf("\n");
  }

  return true;
}


bool hex_to_byte(char hex, byte* ret){
  if (hex >= '0' && hex <= '9'){
    *ret += hex - '0';
    return true;
  } else if (hex >= 'a' && hex <= 'f') {
    *ret +=  hex - 'a' + 10;
    return true;
  } else {
    fprintf(stderr, "Illegal hex character '%c'.\n", hex);
    return false;
  }
}


bool hex_to_byte_array(const char* hex, byte* raw, int len){
  bool success = true;
  int i;
  for (i = 0; i < len; i++){
    raw[i] = 0;
    success &= hex_to_byte(hex[2 * i], &raw[i]);
    raw[i] <<= 4;
    success &= hex_to_byte(hex[2 * i + 1], &raw[i]);
  }
  return success;
}

int main(int argc, char *argv[]) {
  file_t elf;
  void *drcontext = dr_standalone_init();
  char hex[65536];

  if (argc < 2) {
    dr_fprintf(STDERR, "Usage: %s <hex_string> [(--att|--intel|--token|--raw)]\n", argv[0]);
    return 1;
  } else {
    strcpy(hex, argv[1]);
  }

  int output_typ = DEFAULT;

  int arg_idx;
  for (arg_idx = 2; arg_idx < argc; arg_idx++) {
    if (strcmp(argv[arg_idx], "--att") == 0) {
      if (output_typ != DEFAULT) {
        dr_fprintf(STDERR, "Can only provide one output type");
        return 1;
      }
      output_typ = ATT;
    } else if (strcmp(argv[arg_idx], "--intel") == 0) {
      if (output_typ != DEFAULT) {
        dr_fprintf(STDERR, "Can only provide one output type");
        return 1;
      }
      output_typ = INTEL;
    } else if (strcmp(argv[arg_idx], "--token") == 0) {
      if (output_typ != DEFAULT) {
        dr_fprintf(STDERR, "Can only provide one output type");
        return 1;
      }
      output_typ = TOKEN;
    } else if (strcmp(argv[arg_idx], "--raw") == 0) {
      if (output_typ != DEFAULT) {
        dr_fprintf(STDERR, "Can only provide one output type");
        return 1;
      }
      output_typ = RAW;
    } else {
      dr_fprintf(STDERR, "Unknown argument %s", argv[arg_idx]);
    }
  }

  if (output_typ == DEFAULT) {
    output_typ = TOKEN;
  }

  int len = strlen(hex);
  if (len % 2 == 1) {
    fprintf(stderr, "Hex string was length %d, but must be even!\n", len);
    return 1;
  }

  byte* b = malloc(len/2);

  if (!hex_to_byte_array(hex, b, len/2)) {
    fprintf(stderr, "Decode hex failed!\n");
    return 2;
  }

  if (output_typ == RAW) {
    return !print_instr_bytes(drcontext, b, len/2);
  }

  instrlist_t * bb = decode_instrs(drcontext, b, len/2);
  if (bb == NULL) {
    fprintf(stderr, "Decode BB failed!\n");
    return 3;
  }

  if (!tokenize(drcontext, bb, output_typ)) {
    fprintf(stderr, "Tokenize failed!\n");
    return 4;
  }

  return 0;
}
