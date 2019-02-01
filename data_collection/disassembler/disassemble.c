#include "dr_api.h"
#include <stdlib.h> /* for realloc */
#include <assert.h>
#include <stddef.h> /* for offsetof */
#include <string.h> /* for memcpy */


#define BUF_SIZE 10000

unsigned char buf[BUF_SIZE];


int
main(int argc, char *argv[])
{
  file_t f;
  void *drcontext = dr_standalone_init();
  if (argc != 2) {
    dr_fprintf(STDERR, "Usage: %s <tracefile>\n", argv[0]);
    return 1;
  }
  f = dr_open_file(argv[1], DR_FILE_READ | DR_FILE_ALLOW_LARGE);
  if (f == INVALID_FILE) {
    dr_fprintf(STDERR, "Error opening %s\n", argv[1]);
    return 1;
  }

  ssize_t read_bytes;
  instrlist_t * bb;
  instr_t * instr;
  byte * next_pc = buf;

  read_bytes = dr_read_file(f, buf, BUF_SIZE);

  dr_printf("read %d bytes\n", read_bytes);
  dr_printf("first byte %x\n", buf[0]);
  disassemble(drcontext, buf, STDOUT);

  instr = instr_create(drcontext);
  instr_init(drcontext, instr);

  disassemble_set_syntax(DR_DISASM_INTEL);

  while(next_pc - buf < BUF_SIZE){
    next_pc = decode(drcontext, next_pc, instr);
    if(!next_pc) break;
    instr_disassemble(drcontext, instr, STDOUT);
    dr_printf("\n");
    instr_reset(drcontext, instr);
  }

  /*bb = decode_as_bb(GLOBAL_DCONTEXT, buf);

  dr_printf("printing instructions\n");

  if(bb != NULL){
    for(instr = instrlist_first(bb); instr != instrlist_last(bb) ; instr = instr_get_next(instr)){
      instr_disassemble(drcontext, instr, STDOUT);
    }
    }*/
  

  //read_data(f, drcontext);
  dr_close_file(f);
  return 0;
}
