#include "dr_api.h"
#include <stdlib.h> /* for realloc */
#include <assert.h>
#include <stddef.h> /* for offsetof */
#include <string.h> /* for memcpy */
#include <inttypes.h>
#include <unistd.h>

#include "common.h"
#include "client.h"


typedef struct _bbs{
  int num_bbs;
  instrlist_t ** instrlists;
} bbs_t; 

client_arg_t client_args;


bbs_t *  parse_elf_binary(void * drcontext, unsigned char * buf, uint64 filesize){

  unsigned char * next_pc = buf;

  bbs_t * bbs = malloc(sizeof(bbs_t));
  bbs->num_bbs = 0;

  //first get the number of bbs
  while(next_pc - buf < filesize){
    instr_t * instr = instr_create(drcontext);
    next_pc = decode(drcontext, next_pc, instr);
    if(!next_pc) dr_printf("invalid instruction\n"); 
    if(!next_pc) break;
    if(instr_is_cti(instr))
      bbs->num_bbs++;
  }

  dr_printf("num_bbs-%d\n",bbs->num_bbs++);

  unsigned current_bb = 0;
  instrlist_t * current_list = instrlist_create(drcontext);
  bbs->instrlists[current_bb] = current_list;
  bbs->instrlists = malloc(bbs->num_bbs * sizeof(instrlist_t *));

  next_pc = buf;
  while(next_pc - buf < filesize){
    instr_t * instr = instr_create(drcontext);
    next_pc = decode(drcontext, next_pc, instr);
    instrlist_append(current_list, instr);
    if(!next_pc) dr_printf("invalid instruction\n"); 
    if(!next_pc) break;
    if(instr_is_cti(instr)){
      current_list = instrlist_create(drcontext);
      bbs->instrlists[current_bb++] = current_list;
    }
  }

  DR_ASSERT(current_bb == bbs->num_bbs);

  return bbs;

}


//void dump_sql(void * drcontext, bb_t * bbs, code_info_t * cinfo, bookkeep_t * bk, query_t * query){
//}


int
main(int argc, char *argv[])
{
  file_t f;
  void *drcontext = dr_standalone_init();
  if (argc != 3) {
    dr_fprintf(STDERR, "Usage: %s <elf_binary> <sql_folder>\n", argv[0]);
    return 1;
  }

  client_args.mode = RAW_SQL;
  client_args.dump_mode = TEXT;
  client_args.insert_or_update = INSERT_CODE;

  strncpy(client_args.compiler,"unknown", MAX_STRING_SIZE);
  strncpy(client_args.flags,"unknown", MAX_STRING_SIZE);
  strncpy(client_args.data_folder,argv[2], MAX_STRING_SIZE);

  f = dr_open_file(argv[1], DR_FILE_READ | DR_FILE_ALLOW_LARGE);
  if (f == INVALID_FILE) {
    dr_fprintf(STDERR, "Error opening %s\n", argv[1]);
    return 1;
  }

  unsigned char * buf;
  ssize_t read_bytes;
  bool success;

  uint64 filesize;
  success = dr_file_size(f, &filesize);
  if(success)
    dr_printf("file size %" PRIu64 "\n",filesize);
  else{
    dr_printf("ERROR: cannot read file size\n");
    exit(-1);
  }
    

  int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
  dr_printf("num-threads-%d\n",numCPU);

  buf = malloc(filesize);

  read_bytes = dr_read_file(f, buf, filesize);
  dr_printf("read %d bytes\n", read_bytes);

  bbs_t * bbs = parse_elf_binary(drcontext, buf, filesize);
  //dump_sql(drcontext);


  free(buf);
  dr_close_file(f);

  return 0;
}
