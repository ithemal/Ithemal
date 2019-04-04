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
  instr_t * instr;
  disassemble_set_syntax(DR_DISASM_INTEL);

  dr_printf("------------\n");
  for(instr = instrlist_first(bb); instr != NULL; instr = instr_get_next(instr)){
    instr_disassemble(drcontext, instr, STDOUT);
    dr_printf("\n");
  }
  dr_printf("-------------\n");

}


bool tokenize(void * drcontext, instrlist_t * bb, bool debug){


  //create the dump related data structures
  code_info_t cinfo;
  if(debug)
    debug_print(drcontext, bb);
  if(!text_token(drcontext, &cinfo, bb)){
    return false;
  }
  if(debug){
    dr_printf("\ncode_size-%d(%d)\n",cinfo.code_size, MAX_CODE_SIZE);
  }

  int i = 0;
  for(i = 0;i < cinfo.code_size; i++){
    dr_printf("%c",cinfo.code[i]);
  }
  return true;

}

instrlist_t * decode_instrs(void * drcontext, byte * raw, int len){

  unsigned char * start_pc = raw;
  unsigned char * end_pc = start_pc + len;
  instrlist_t * current_list = instrlist_create(drcontext);

  bool success = true;

  while(start_pc < end_pc){
    instr_t * instr = instr_create(drcontext);
    start_pc = decode(drcontext, start_pc, instr);
    instrlist_append(current_list, instr);
    if(!start_pc)
      return NULL;
  }
  //dummy instr to match tokenize interface
  instrlist_append(current_list, INSTR_CREATE_cpuid(drcontext));

  return current_list;

}

byte hex_to_byte(char hex){

  byte ret;
  if(hex >= '0' && hex <= '9'){
    ret = hex - '0';
  }
  else{
    ret =  hex - 'a' + 10;
  }
  return ret;
  
}


void hex_to_byte_array(const char * hex, byte * raw, int len){

  int i = 0;
  for(; i < len; i++){
    raw[i] = hex_to_byte(hex[2 * i]);
    raw[i] = (byte)(raw[i] << 4) + hex_to_byte(hex[2 * i + 1]); 
  }


}

int
main(int argc, char *argv[])
{
  file_t elf;
  void *drcontext = dr_standalone_init();
  char hex[65536];

  if (argc < 2) {
    dr_fprintf(STDERR, "Usage: %s <hex_string> <debug>\n", argv[0]);
    return 1;
  }
  else{
      strcpy(hex, argv[1]);
  }

  bool debug = false;
  if(argc == 3 && strcmp(argv[2],"1") == 0){
    debug = true;
  }

  int len = strlen(hex);
  byte * b = malloc(len/2);

  hex_to_byte_array(hex, b, len/2);
  instrlist_t * bb = decode_instrs(drcontext, b, len/2);
  if(bb != NULL){
    tokenize(drcontext, bb, debug);
  }

  return 0;
}
