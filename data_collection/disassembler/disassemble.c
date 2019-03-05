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
#include "sql_dump.h"


typedef struct _bbs{
  int num_bbs;
  instrlist_t ** instrlists;
  int * pos;
} bbs_t; 

client_arg_t client_args;
config_t config;


bool populate_code(void * drcontext, instrlist_t * bb, code_info_t * cinfo, uint32_t dump_mode){
  

  if(dump_mode == DUMP_INTEL){
    printf("intel-%d\n",dump_mode);
    disassemble_set_syntax(DR_DISASM_INTEL);
    textual_embedding(drcontext, cinfo, bb);
    cinfo->code_type = CODE_INTEL;
    return true;
  }
  else if(dump_mode == DUMP_ATT){
    printf("att-%d\n",dump_mode);
    disassemble_set_syntax(DR_DISASM_ATT);
    textual_embedding_with_size(drcontext, cinfo, bb);
    cinfo->code_type = CODE_ATT;
    return true;
  }
  else if(dump_mode == DUMP_TOKEN){
    printf("token-%d\n",dump_mode);
    token_text_embedding(drcontext, cinfo, bb);
    cinfo->code_type = CODE_TOKEN;
    return true;
  }
  else{
    return false;
  }

} 

int fill_sql(FILE * file, query_t * query, code_info_t * cinfo, uint32_t insert){

  int sz = 0;
  if(insert)
    sz = insert_code(query, cinfo, client_args.op_mode);
  else
    sz = update_code(query, cinfo, client_args.op_mode);
  
  if(sz == -1){
    return sz;
  }
  
  DR_ASSERT(sz <= MAX_QUERY_SIZE - 3);
  sz = complete_query(query,sz);
  query[sz++] = '\0'; 
  return sz;
  
}

void dump_sql(void * drcontext, const char * elfname,  bbs_t * bbs, char * sql_filename){

  FILE * sqlfile = fopen(sql_filename, "w");

  if(sqlfile == NULL){
    printf("sql file not opened properly - %s\n", sql_filename);
    exit(-1);
  }

  //create the dump related data structures
  query_t query[MAX_QUERY_SIZE];
  code_info_t cinfo;

#define NUM_DUMP_MODES 3
  uint32_t dump_modes[NUM_DUMP_MODES] = {DUMP_INTEL, DUMP_ATT, DUMP_TOKEN};
  int i,j;

  int sz = insert_config(query, &config, client_args.dump_mode);
  DR_ASSERT(sz <= MAX_QUERY_SIZE - 3);
  sz = complete_query(query,sz);
  query[sz++] = '\0'; 
  fprintf(sqlfile, "%s\n", query);

  for(i = 0; i < bbs->num_bbs; i++){

    cinfo.rel_addr = bbs->pos[i];
    strncpy(cinfo.module, elfname, MAX_MODULE_SIZE);

    bool insert = client_args.insert_or_update;

    for(j = 0; j < NUM_DUMP_MODES; j++){

      if(populate_code(drcontext, bbs->instrlists[i],  &cinfo, client_args.dump_mode & dump_modes[j]))
	if(fill_sql(sqlfile, query, &cinfo, insert) != -1){
	  fprintf(sqlfile, "%s\n", query);
	  insert = false;
	}
    }
  }

  fclose(sqlfile);

}

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


#define MAX_FUNCS 1000000
#define NAME_SIZE 1000
  

  typedef struct _func{
    uint32_t offset;
    uint32_t size;
  } func_t;

  func_t func_bd[MAX_FUNCS];
  FILE * meta;
  uint32_t fnum = 0;
  char fname[NAME_SIZE];

  //read the meta data file
  printf("reading file\n");

  meta = fopen(metafilename, "r");
  while(fscanf(meta, "%s\t%lu\t%lu\n", fname, &func_bd[fnum].offset, &func_bd[fnum].size) != EOF){
    printf("%d-%s\t%lu\t%lu\n", fnum, fname, func_bd[fnum].offset, func_bd[fnum].size);
    fnum++;
  }

  fclose(meta);

  printf("done\n");

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
  bbs->pos = malloc(bbs->num_bbs * sizeof(int));
  bbs->instrlists[current_bb] = current_list;

  for(i = 0; i < fnum; i++){
    start_pc = &buf[func_bd[i].offset];
    end_pc = start_pc + func_bd[i].size;
    bbs->pos[current_bb] = func_bd[i].offset;

    while(start_pc < end_pc){
      if(current_bb > 1000) {bbs->num_bbs = current_bb; break;}
      instr_t * instr = instr_create(drcontext);
      start_pc = decode(drcontext, start_pc, instr);
      instrlist_append(current_list, instr);
      if(!start_pc) dr_printf("invalid instruction\n"); 
      if(!start_pc) break;
      if(instr_is_cti(instr)){
	current_list = instrlist_create(drcontext);
	bbs->instrlists[current_bb++] = current_list;
	bbs->pos[current_bb] = start_pc - buf;
      }
    }
  }

  //DR_ASSERT(current_bb == bbs->num_bbs);

  return bbs;

}


int
main(int argc, char *argv[])
{
  file_t elf;
  void *drcontext = dr_standalone_init();
  if (argc != 4) {
    dr_fprintf(STDERR, "Usage: %s <elf_binary> <metadata_file> <sql_folder>\n", argv[0]);
    return 1;
  }

  client_args.op_mode = RAW_SQL;
  client_args.dump_mode = DUMP_INTEL | DUMP_ATT | DUMP_TOKEN;
  client_args.insert_or_update = INSERT_CODE;

  strncpy(config.compiler,"unknown", MAX_STRING_SIZE);
  strncpy(config.flags,"unknown", MAX_STRING_SIZE);
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

#define FILENAME_SIZE 1000
  char sqlfilename[FILENAME_SIZE];
  sprintf(sqlfilename, "%s/%s.sql",client_args.data_folder, argv[1]);
  dump_sql(drcontext, argv[1], bbs, sqlfilename);


  free(buf);
  dr_close_file(elf);

  return 0;
}
