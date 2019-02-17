#include "dr_api.h"
#include <stdlib.h> /* for realloc */
#include <assert.h>
#include <stddef.h> /* for offsetof */
#include <string.h> /* for memcpy */
#include <inttypes.h>

unsigned char * buf;

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
  byte * next_pc;
  bool success;

  uint64 filesize;
  success = dr_file_size(f, &filesize);
  if(success)
    dr_printf("file size %" PRIu64 "\n",filesize);
  else{
    dr_printf("ERROR: cannot read file size\n");
    exit(-1);
  }
    

  buf = malloc(filesize);
  next_pc = buf;

  read_bytes = dr_read_file(f, buf, filesize);
  dr_printf("read %d bytes\n", read_bytes);

  instr = instr_create(drcontext);
  instr_init(drcontext, instr);

  disassemble_set_syntax(DR_DISASM_INTEL);

  while(next_pc - buf < filesize){
    next_pc = decode(drcontext, next_pc, instr);
    if(!next_pc) dr_printf("invalid instruction\n"); 
    if(!next_pc) break;
    instr_disassemble(drcontext, instr, STDOUT);
    dr_printf("-%d,\n", instr_is_cti(instr));
    instr_reset(drcontext, instr);
  }


  free(buf);
  dr_close_file(f);

  return 0;
}
