#include <stddef.h> /* for offsetof */
#include "dr_api.h"

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */

#include "utility.h"

void post_cleancall();

#define MAX_STRING_SIZE 100

//client arguments
typedef struct {
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
} client_arg_t;

client_arg_t client_args;

typedef struct {
  void * code;
  void * timing;
  file_t code_file;
  file_t timing_file;
} per_thread_t;


    /* thread init event */
static void 
thread_init(void * drcontext){

  per_thread_t * data = (per_thread_t)dr_thread_alloc(drcontext, sizeof(per_thread_t));    
  dr_set_tls_field(drcontext,data);
  
  char base[MAX_STRING_SIZE];
  get_base_filename(drcontext,filename,MAX_STRING_SIZE);

  char filename[MAX_STRING_SIZE];
  dr_snprintf(filename,MAX_STRING_SIZE,"%s_code.txt",base);
  
  data->code_file = dr_open_file(filename, DR_FILE_WRITE | DR_FILE_READ);
  data->code = dr_map_file(code_file,


  bookkeep_t * bk = (bookkeep_t *)data;
  bk->outfile = dr_open_file(filename, DR_FILE_WRITE | DR_FILE_READ);
  
  dr_printf("outfile-%s\n",filename);
  DR_ASSERT(bk->outfile);
  
  get_config(drcontext, bk->outfile,client_args.compiler, client_args.flags);
  
}

static inline void save_registers(void * drcontext, instrlist_t * bb, instr_t * where){
  dr_save_reg(drcontext,bb,where,DR_REG_RAX,SPILL_SLOT_1);
  dr_save_reg(drcontext,bb,where,DR_REG_RCX,SPILL_SLOT_2);
  dr_save_reg(drcontext,bb,where,DR_REG_RDX,SPILL_SLOT_3);
  dr_save_reg(drcontext,bb,where,DR_REG_RBX,SPILL_SLOT_4);
}

static inline void restore_registers(void * drcontext, instrlist_t * bb, instr_t * where){
  dr_restore_reg(drcontext,bb,where,DR_REG_RAX,SPILL_SLOT_1);
  dr_restore_reg(drcontext,bb,where,DR_REG_RCX,SPILL_SLOT_2);
  dr_restore_reg(drcontext,bb,where,DR_REG_RDX,SPILL_SLOT_3);
  dr_restore_reg(drcontext,bb,where,DR_REG_RBX,SPILL_SLOT_4);
}

#define REPEAT_TIMES 10

static void measure_overhead(void * drcontext, instrlist_t * bb, uint32_t num_bbs, void * data){

  instr_t * first = instrlist_first(bb);
  save_registers(drcontext,bb,first);
 
  uint32_t i = 0;

  for(i = 0; i < REPEAT_TIMES; i++){
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtsc(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(data,OPSZ_4), opnd_create_reg(DR_REG_EAX)));
    restore_registers(drcontext,bb,first);
    save_registers(drcontext,bb,first);
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtsc(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(data + sizeof(uint32_t),OPSZ_4), opnd_create_reg(DR_REG_EAX)));
    dr_insert_clean_call(drcontext,bb,first,post_cleancall,false,1,opnd_create_immed_uint(num_bbs,OPSZ_4));
  }

  restore_registers(drcontext,bb,first);

}

static dr_emit_flags_t
bb_creation_event(void * drcontext, void * tag, instrlist_t * bb, bool for_trace, bool translating){

  instr_t * first;
  instr_t * last;

  first = instrlist_first(bb);
  last = instrlist_last(bb);

  void * data = dr_get_tls_field(drcontext);
  bookkeep_t * bk = (bookkeep_t *)data;
  //we cannot store more than the buffer size
  if(bk->num_bbs >= NUM_BBS){
    return DR_EMIT_DEFAULT;
  }

  //first basic block is used to measure the overhead
  if(bk->num_bbs == 0){
    measure_overhead(drcontext, bb, bk->num_bbs, data);
    bk->num_bbs++;
    return DR_EMIT_DEFAULT;
  }

  //sql dump of the current BB's code
  insert_code(drcontext, bk->outfile, bb);

  /*
    before execution of the basic block
    save registers
    cpuid
    rdtsc
    spill eax
    restore registers
   */
 
  save_registers(drcontext,bb,first);
  instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
  instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtsc(drcontext));
  instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(data,OPSZ_4), opnd_create_reg(DR_REG_EAX)));
  restore_registers(drcontext,bb,first);
 
  /*
    after execution of the basic block - before the final conditional branch
    save registers
    cpuid
    rdtsc
    save eax
    clean call
    restore registers
   */
  
  save_registers(drcontext,bb,last);
  instrlist_meta_preinsert(bb,last,INSTR_CREATE_cpuid(drcontext));
  instrlist_meta_preinsert(bb,last,INSTR_CREATE_rdtsc(drcontext));
  instrlist_meta_preinsert(bb,last,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(data + sizeof(uint32_t),OPSZ_4), opnd_create_reg(DR_REG_EAX)));

  //fill up the static metadata for the bb
  bb_data_t * timing = (bb_data_t *)(data + START_BB_DATA);

  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = (int)first_pc - (int)md->start;
  
  timing[bk->num_bbs].meta.slots_filled = 0;
  timing[bk->num_bbs].meta.rel_addr = rel_addr;
  timing[bk->num_bbs].meta.module_start = md->start;
  
  //clean call for recording time and dumping if necessary
  dr_insert_clean_call(drcontext,bb,last,post_cleancall,false,1,opnd_create_immed_uint(bk->num_bbs,OPSZ_4));
  restore_registers(drcontext,bb,last);

  bk->num_bbs++;
  dr_printf("%d\n",bk->num_bbs);

  return DR_EMIT_DEFAULT;
  
}

void post_cleancall(uint32_t num_bbs){

  void * drcontext = dr_get_current_drcontext();
  void * data = dr_get_tls_field(drcontext);   

  bookkeep_t * bk = (bookkeep_t *)data;
  uint32_t before = bk->prevtime;
  uint32_t now = bk->nowtime;
 
  bb_data_t * timing = (bb_data_t *)(data + START_BB_DATA);
  uint32_t slots_filled = timing[num_bbs].meta.slots_filled; 

  uint32_t i = 0;
  //remember the first time slot is used for the counter
  if (slots_filled >= TIME_SLOTS - METADATA_SLOTS){ //dump to data base

    app_pc md_start = timing[num_bbs].meta.module_start;
    module_data_t * md = dr_lookup_module(md_start);
  
    for(i = 0; i < slots_filled; i++){
      insert_time(bk->outfile,
		  timing[num_bbs].times[i],
		  timing[num_bbs].meta.rel_addr,
		  dr_module_preferred_name(md),
		  proc_get_model());
    }

    timing[num_bbs].meta.slots_filled = 0;
    slots_filled = 0;
  }

  timing[num_bbs].times[slots_filled] = now - before;
  timing[num_bbs].meta.slots_filled++;

}

  /* thread exit event */
static void
thread_exit(void * drcontext){

  void * data = dr_get_tls_field(drcontext);
  bookkeep_t * bk = (bookkeep_t *)data;
  bb_data_t * timing = (bb_data_t *)(data + START_BB_DATA);
  uint32_t i = 0;
  uint32_t j = 0;

  for(i = 0; i < bk->num_bbs; i++){
    
    uint32_t slots_filled = timing[i].meta.slots_filled;
    app_pc md_start = timing[i].meta.module_start;
    module_data_t * md = dr_lookup_module(md_start);

    for(j = 0; j < slots_filled; j++){
      insert_time(bk->outfile,
		  timing[i].times[j],
		  timing[i].meta.rel_addr,
		  dr_module_preferred_name(md),
		  proc_get_model());
    }

  }

  dr_close_file(bk->outfile);
  
}

static void
event_exit(void)
{


}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    dr_set_client_name("BB timing client","");
    
    /* register events */
    dr_register_exit_event(event_exit);

    dr_register_thread_init_event(thread_init);
    dr_register_thread_exit_event(thread_exit);

    dr_register_bb_event(bb_creation_event);
    disassemble_set_syntax(DR_DISASM_INTEL);

    DR_ASSERT(argc == 3);
    strncpy(client_args.compiler,argv[1], MAX_STRING_SIZE);
    strncpy(client_args.flags,argv[2], MAX_STRING_SIZE);

    dr_printf("arguments - 1.%s, 2.%s\n",client_args.compiler,client_args.flags);

}
