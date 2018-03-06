#include <stddef.h> /* for offsetof */
#include "dr_api.h"

#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */

#include "mmap.h"
#include "utility.h"

void post_cleancall();

#define MAX_STRING_SIZE 100

//client arguments
typedef struct {
  char program[MAX_STRING_SIZE];
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
  uint32_t mode;
} client_arg_t;

client_arg_t client_args;
mmap_file_t filenames_file;
void * mutex;

#define BEGIN_CONTROL(C,ST,EN)						   \
  if(client_args.mode == SNOOP){					   \
    while(!__sync_bool_compare_and_swap(&C,ST,EN));			   \
  }									   \
 
#define END_CONTROL(C,ST,EN)						\
  if(client_args.mode == SNOOP){					\
    DR_ASSERT(__sync_bool_compare_and_swap(&C,ST,EN));			\
    while(C != IDLE);							\
  }									\


//thread init event
static void 
thread_init(void * drcontext){

  mmap_file_t * data = (mmap_file_t *)dr_thread_alloc(drcontext, sizeof(mmap_file_t));    
  dr_set_tls_field(drcontext,data);
  
  get_filename(drcontext,data->filename,MAX_MODULE_SIZE);
  
  if(client_args.mode == SNOOP){
    dr_mutex_lock(mutex);
    volatile thread_files_t * files = filenames_file.data;
    DR_ASSERT(files); //should be memory mapped 

    BEGIN_CONTROL(files->control,IDLE,DR_CONTROL)
    strcpy(files->modules[files->num_modules],data->filename);
    files->num_modules++;
    END_CONTROL(files->control,DR_CONTROL,DUMP_ONE);

    dr_mutex_unlock(mutex);  
  }

  //create the file and memory map it
  create_memory_map_file(data,TOTAL_SIZE);
  memset(data->data,0,TOTAL_SIZE);
 
  bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
  bk->arch = proc_get_model();
 
  
}

//bb analysis routines

//bb instrumentation routines
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
  volatile bookkeep_t * bk = (bookkeep_t *)(data + START_BK_DATA); 

  uint32_t i = 0;

  for(i = 0; i < REPEAT_TIMES; i++){
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtsc(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&bk->prevtime,OPSZ_4), opnd_create_reg(DR_REG_EAX)));
    restore_registers(drcontext,bb,first);
    save_registers(drcontext,bb,first);
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtsc(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&bk->nowtime,OPSZ_4), opnd_create_reg(DR_REG_EAX)));
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

  mmap_file_t * file = dr_get_tls_field(drcontext);
  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);

  //we cannot store more than the buffer size
  if(bk->num_bbs >= NUM_BBS){
    return DR_EMIT_DEFAULT;
  }

  //first basic block is used to measure the overhead
  if(bk->num_bbs == 0){
    measure_overhead(drcontext, bb, bk->num_bbs, file->data);
    bk->num_bbs++;
    return DR_EMIT_DEFAULT;
  }

  //get the running code / do analysis 
  BEGIN_CONTROL(cinfo->control,IDLE,DR_CONTROL);
  
  analyze_bb(drcontext, file->data, bb);
 
  END_CONTROL(cinfo->control,DUMP_ONE,DR_CONTROL);

  instrument_bb(drcontext, file->data, bb);
  // begin instrumentation
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
  instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&bk->prevtime,OPSZ_4), opnd_create_reg(DR_REG_EAX)));
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
  instrlist_meta_preinsert(bb,last,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&bk->nowtime,OPSZ_4), opnd_create_reg(DR_REG_EAX))); 
  //clean call for recording time and dumping if necessary
  dr_insert_clean_call(drcontext,bb,last,post_cleancall,false,1,opnd_create_immed_uint(bk->num_bbs,OPSZ_4));
  restore_registers(drcontext,bb,last);

  bk->num_bbs++;

  return DR_EMIT_DEFAULT;
 
}


void post_cleancall(uint32_t num_bbs){

  //starting with full dumping
  void * drcontext = dr_get_current_drcontext();
  mmap_file_t * file = dr_get_tls_field(drcontext);   

  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);
  uint32_t before = bk->prevtime;
  uint32_t now = bk->nowtime;
 
  bb_data_t * timing = (bb_data_t *)(file->data + START_BB_DATA);
  uint32_t slots_filled = timing[num_bbs].meta.slots_filled; 

  uint32_t i = 0;
  //remember the first time slot is used for the counter
  if (slots_filled >= TIME_SLOTS - METADATA_SLOTS){ //dump to data base

    if(client_args.mode == SNOOP){ 
      while(!__sync_bool_compare_and_swap(&bk->control,IDLE,DR_CONTROL)){} //dr should be in control
    }
    
    bk->dump_bb = num_bbs;

    if(client_args.mode == SNOOP){
      DR_ASSERT(__sync_bool_compare_and_swap(&bk->control,DR_CONTROL,DUMP_ONE));    
      while(bk->control != IDLE);
    }

    timing[num_bbs].meta.slots_filled = 0;
    slots_filled = 0;
  }

  insert_timing(&timing[num_bbs],now - before);
 
}

  /* thread exit event */
static void
thread_exit(void * drcontext){


  mmap_file_t * file = dr_get_tls_field(drcontext);
  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);

  BEGIN_CONTROL(bk->control,IDLE,DR_CONTROL);
 
  END_CONTROL(bk->control,DR_CONTROL,DUMP_ALL);
  
  mmap_file_t * data = dr_get_tls_field(drcontext);
  dr_unmap_file(data->data,TOTAL_SIZE);
  dr_close_file(data->file);
}

static void
event_exit(void)
{
  if(client_args.mode == SNOOP){
    dr_mutex_destroy(mutex);
    volatile thread_files_t * files = filenames_file.data;
    files->control = EXIT;
    while(files->control != IDLE);
    dr_unmap_file(filenames_file.data,sizeof(thread_files_t));
    dr_close_file(filenames_file.file);
  }
}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    
    /* register events */
    
    dr_register_thread_init_event(thread_init);
    
    dr_register_bb_event(bb_creation_event);
    
    dr_register_thread_exit_event(thread_exit);
    dr_register_exit_event(event_exit);
    
    disassemble_set_syntax(DR_DISASM_INTEL);

    /* client arguments */
    DR_ASSERT(argc == 5);
    client_args.mode = argv[1];
    strncpy(client_args.program.arg[2], MAX_STRING_SIZE);
    strncpy(client_args.compiler,argv[3], MAX_STRING_SIZE);
    strncpy(client_args.flags,argv[4], MAX_STRING_SIZE);
    
    if(client_args.mode == SNOOP){
      /* open filenames_file and mmap it */
      strcpy(filenames_file.filename,FILENAMES_FILE);
      create_memory_map_file(&filenames_file,sizeof(thread_files_t));
      memset(filenames_file.data,0,sizeof(thread_files_t));
      mutex = dr_mutex_create();
    }
  
}
