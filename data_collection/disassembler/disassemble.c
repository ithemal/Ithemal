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
#include "sql_dump_v2.h"

#define FILENAME_SIZE 1000


client_arg_t client_args;
config_t config;



bool write_code(void * drcontext, FILE * sqlfile, query_t * query, code_info_t * cinfo, instrlist_t * bb){

  int sz = -1;

  raw_bits(drcontext,  cinfo, bb);
  if(cinfo->code_size == -1)
    return false;

  sz = insert_code(query, cinfo);
  if(sz == -1)
    return false;
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  fprintf(sqlfile,"%s",query);

  sz = insert_code_metadata(query, cinfo);
  if(sz == -1)
    return false;  
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  fprintf(sqlfile,"%s",query);

  disassemble_set_syntax(DR_DISASM_INTEL);

  text_intel(drcontext, cinfo, bb);
  if(cinfo->code_size == -1)
    return false;

  sz = insert_disassembly(query, cinfo, CODE_INTEL);
  if(sz == -1)
    return false;  
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  fprintf(sqlfile,"%s",query);

  return true;
  
}

void dump_sql(void * drcontext, const char * elfname,  instrlist_t * bb, uint32_t rel_addr, FILE * sqlfile){


  //create the dump related data structures
  query_t query[MAX_QUERY_SIZE];
  code_info_t cinfo;

  cinfo.rel_addr = rel_addr;
  strncpy(cinfo.module, elfname, MAX_MODULE_SIZE);
  write_code(drcontext, sqlfile, query, &cinfo, bb);


}

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

void parse_elf_binary(void * drcontext, unsigned char * buf, char * metafilename, char * sqlfilename, char * elfname){

  typedef struct _func{
    char fname[FILENAME_SIZE];
    uint32_t offset;
    uint32_t size;
  } func_t;

  func_t func_bd;
  FILE * meta;
  FILE * sql;
  uint32_t fnum = 0;
  uint32_t bbnum = 0;

  meta = fopen(metafilename, "r");
  sql = fopen(sqlfilename, "w");

  //configuration dumping
  query_t query[MAX_QUERY_SIZE];

  int sz = insert_architecture(query, &config);
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  fprintf(sql, "%s", query);

  sz = insert_config(query, &config);
  DR_ASSERT(sz <= MAX_QUERY_SIZE);
  fprintf(sql, "%s", query);


  while(fscanf(meta, "%s\t%u\t%u\n", func_bd.fname, &func_bd.offset, &func_bd.size) != EOF){

    //printf("%s\t%u\t%u\n", func_bd.fname, func_bd.offset, func_bd.size);

    unsigned char * start_pc = &buf[func_bd.offset];
    unsigned char * end_pc = start_pc + func_bd.size;
    instrlist_t * current_list = instrlist_create(drcontext);
    unsigned char * start_bb = start_pc;

    while(start_pc < end_pc){
      instr_t * instr = instr_create(drcontext);
      start_pc = decode(drcontext, start_pc, instr);
      instrlist_append(current_list, instr);
      if(!start_pc) dr_printf("invalid instruction\n"); 
      if(!start_pc) break;
      if(instr_is_cti(instr)){
	dump_sql(drcontext, elfname, current_list, start_bb - buf, sql);
	instrlist_clear(drcontext, current_list);
	start_bb = start_pc;
	bbnum++;
	if(bbnum % 100000 == 0) printf("bbnum-%d\n",bbnum);
      }
    }
    fnum++;     
  }


  fclose(sql);
  fclose(meta);

}


int
main(int argc, char *argv[])
{
  file_t elf;
  void *drcontext = dr_standalone_init();
  if (argc != 5) {
    dr_fprintf(STDERR, "Usage: %s <binary_name> <elf_binary> <metadata_file> <sql_file>\n", argv[0]);
    return 1;
  }

  char binaryname[FILENAME_SIZE];
  char elfpath[FILENAME_SIZE];
  char metadatapath[FILENAME_SIZE];
  char sqlpath[FILENAME_SIZE];

  strncpy(binaryname, argv[1], FILENAME_SIZE);
  strncpy(elfpath, argv[2], FILENAME_SIZE);
  strncpy(metadatapath, argv[3], FILENAME_SIZE);
  strncpy(sqlpath, argv[4], FILENAME_SIZE);

  strncpy(config.compiler,"unknown", MAX_STRING_SIZE);
  strncpy(config.flags,"unknown", MAX_STRING_SIZE);
  strncpy(config.name,"unknown", MAX_STRING_SIZE);
  strncpy(config.vendor,"unknown", MAX_STRING_SIZE);


  elf = dr_open_file(elfpath, DR_FILE_READ | DR_FILE_ALLOW_LARGE);
  if (elf == INVALID_FILE) {
    dr_fprintf(STDERR, "Error opening %s\n", elfpath);
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
    

  buf = malloc(filesize);
  read_bytes = dr_read_file(elf, buf, filesize);
  dr_printf("read %d bytes\n", read_bytes);

  parse_elf_binary(drcontext, buf, metadatapath, sqlpath, binaryname);
  free(buf);
  dr_close_file(elf);

  return 0;
}
