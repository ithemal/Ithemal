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

void debug_print(void * drcontext, instrlist_t * bb){

  int i = 0;
  instr_t* instr;
  disassemble_set_syntax(DR_DISASM_INTEL);

  dr_printf("------------\n");
  for (instr = instrlist_first(bb); instr != NULL; instr = instr_get_next(instr)){
    instr_disassemble(drcontext, instr, STDOUT);
    dr_printf("\n");
  }
  dr_printf("-------------\n");

}


bool tokenize(void * drcontext, instrlist_t * bb, bool debug){
  //create the dump related data structures
  code_info_t cinfo;
  if (debug) {
    debug_print(drcontext, bb);
  }

  if (!text_xml(drcontext, &cinfo, bb)) {
    return false;
  }

  if (debug) {
    dr_printf("\ncode_size-%d(%d)\n",cinfo.code_size, MAX_CODE_SIZE);
  }
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
    dr_fprintf(STDERR, "Usage: %s <hex_string> [<debug>]\n", argv[0]);
    return 1;
  } else {
    strcpy(hex, argv[1]);
  }

  bool debug = false;
  if (argc >= 3) {
    if (strcmp(argv[2], "1") == 0) {
      debug = true;
    } else {
      fprintf(stderr, "Unknown argument for debug: \"%s\"; expected \"1\" (or nothing)\n", argv[2]);
      return 1;
    }
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

  instrlist_t * bb = decode_instrs(drcontext, b, len/2);
  if (bb == NULL) {
    fprintf(stderr, "Decode BB failed!\n");
    return 3;
  }

  if (!tokenize(drcontext, bb, debug)) {
    fprintf(stderr, "Tokenize failed!\n");
    return 4;
  }

  return 0;
}
