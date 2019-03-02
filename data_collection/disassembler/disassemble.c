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


typedef struct _bbs{
  int num_bbs;
  instrlist_t ** instrlists;
} bbs_t; 

client_arg_t client_args;

void debug_print(void * drcontext, bbs_t * bbs){

  int i = 0;
  instr_t * instr;
  disassemble_set_syntax(DR_DISASM_INTEL);

  for(i = 0; i < bbs->num_bbs; i++){
    instrlist_t * bb = bbs->instrlists[i];
    dr_printf("bb - %d\n",i);
    for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
      instr_disassemble(drcontext, instr, STDOUT);
      dr_printf("\n");
    }
  }

}

bbs_t * parse_elf_binary(void * drcontext, unsigned char * buf, char * metafilename){


#define MAX_FUNCS 1000
#define NAME_SIZE 40
  
  typedef struct _func{
    char name[NAME_SIZE];
    uint64_t offset;
    uint64_t size;
  } func_t;

  func_t func_bd[MAX_FUNCS];
  FILE * meta;
  uint32_t fnum = 0;

  //read the meta data file
  meta = fopen(metafilename, "r");
  while(fscanf(meta, "%s\t%lu\t%lu\n", func_bd[fnum].name, &func_bd[fnum].offset, &func_bd[fnum].size) != EOF){
    fnum++;
  }
  fclose(meta);

  unsigned char * start_pc = buf;
  unsigned char * end_pc = buf;
  bbs_t * bbs = malloc(sizeof(bbs_t));
  int i = 0;

  //find the number of bbs out there first
  bbs->num_bbs = 0;
  for(i = 0; i < fnum; i++){
    start_pc = &buf[func_bd[i].offset];
    end_pc = start_pc + func_bd[i].size;
    instr_t * instr = instr_create(drcontext);
    while(start_pc < end_pc){
      start_pc = decode(drcontext, start_pc, instr);
      if(!start_pc) dr_printf("invalid instruction\n"); 
      if(!start_pc) break;
      if(instr_is_cti(instr))
	bbs->num_bbs++;
    }
    instr_free(drcontext, instr);  
  }

  dr_printf("num_bbs-%d\n",bbs->num_bbs);

  //now create the actual bbs
  unsigned current_bb = 0;
  instrlist_t * current_list = instrlist_create(drcontext);

  bbs->instrlists = malloc(bbs->num_bbs * sizeof(instrlist_t *));
  bbs->instrlists[current_bb] = current_list;

  for(i = 0; i < fnum; i++){
    start_pc = &buf[func_bd[i].offset];
    end_pc = start_pc + func_bd[i].size;
    while(start_pc < end_pc){
      instr_t * instr = instr_create(drcontext);
      start_pc = decode(drcontext, start_pc, instr);
      instrlist_append(current_list, instr);
      if(!start_pc) dr_printf("invalid instruction\n"); 
      if(!start_pc) break;
      if(instr_is_cti(instr)){
	current_list = instrlist_create(drcontext);
	bbs->instrlists[current_bb++] = current_list;
      }
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
  file_t elf;
  void *drcontext = dr_standalone_init();
  if (argc != 4) {
    dr_fprintf(STDERR, "Usage: %s <elf_binary> <metadata_file> <sql_folder>\n", argv[0]);
    return 1;
  }

  client_args.mode = RAW_SQL;
  client_args.dump_mode = TEXT;
  client_args.insert_or_update = INSERT_CODE;

  strncpy(client_args.compiler,"unknown", MAX_STRING_SIZE);
  strncpy(client_args.flags,"unknown", MAX_STRING_SIZE);
  strncpy(client_args.data_folder,argv[3], MAX_STRING_SIZE);

  elf = dr_open_file(argv[1], DR_FILE_READ | DR_FILE_ALLOW_LARGE);
  if (elf == INVALID_FILE) {
    dr_fprintf(STDERR, "Error opening %s\n", argv[1]);
    return 1;
  }


  unsigned char * buf;
  ssize_t read_bytes;
  bool success;

  uint64 filesize;
  success = dr_file_size(elf, &filesize);
  if(!success){
    dr_printf("ERROR: cannot read file size\n");
    exit(-1);
  }
    
  //int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
  //dr_printf("num-threads-%d\n",numCPU);

  buf = malloc(filesize);
  read_bytes = dr_read_file(elf, buf, filesize);
  dr_printf("read %d bytes\n", read_bytes);

  bbs_t * bbs = parse_elf_binary(drcontext, buf, argv[2]);
  debug_print(drcontext, bbs);

  free(buf);
  dr_close_file(elf);

  return 0;
}
