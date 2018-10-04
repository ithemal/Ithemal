#include <stddef.h> /* for offsetof */
#include <stdint.h> /* for data type definitions */
#include <string.h> /* may be for memset */
#include <stdlib.h>

#include "dr_api.h"
#include "timing_mmap.h"
#include "timing_logic.h"
#include "timing_dump.h"

#include "dr_utility.h"
#include "code_embedding.h"
#include "sqlite3_impl.h"
#include "common.h"


//client arguments
typedef struct {
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
  char data_folder[MAX_STRING_SIZE];
  uint32_t mode;
  uint32_t code_format;
  code_embedding_t embedding_func;
} client_arg_t;

client_arg_t client_args;
mmap_file_t filenames_file;
uint32_t num_threads = 0; //sqlite can handle only one thread
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


void post_cleancall(uint32_t num_bbs);

//thread init event
static void 
thread_init(void * drcontext){

  mmap_file_t * data = (mmap_file_t *)dr_thread_alloc(drcontext, sizeof(mmap_file_t));    
  dr_set_tls_field(drcontext,data);
  
  get_perthread_filename(drcontext,data->filename,MAX_MODULE_SIZE);
  data->offs = data->filled = 0;
  
  if(client_args.mode == SNOOP){ //create filenames file for snooping process to know where we are dumping data
    dr_mutex_lock(mutex);
    volatile thread_files_t * files = filenames_file.data;
    DR_ASSERT(files); //should be memory mapped 

    BEGIN_CONTROL(files->control,IDLE,DR_CONTROL)
    strcpy(files->modules[files->num_modules],data->filename);
    files->num_modules++;
    END_CONTROL(files->control,DR_CONTROL,DUMP_ONE);

    dr_mutex_unlock(mutex);  
  }
  else if (client_args.mode == SQLITE){ //keep track of data in this mode
    dr_mutex_lock(mutex);
    num_threads++;
    dr_mutex_unlock(mutex);
  }

  //create the file and memory map it and if sqlite mode then only dump data from thread 1
  if(client_args.mode != SQLITE || num_threads <= 1){
    data->file = dr_open_file(data->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
    create_memory_map_file(data,TOTAL_SIZE);
    memset(data->data,0,TOTAL_SIZE);
  
    //processor model information
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    bk->arch = proc_get_model();
  }
  else{
    data->data = NULL;
  }

  //for all modes
  bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
  bk->overhead = 2e8;

  //if mode is raw sql dumping 
  if(client_args.mode == RAW_SQL){
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    bk->dynamic_file = dr_thread_alloc(drcontext, sizeof(mmap_file_t));
    bk->static_file = dr_thread_alloc(drcontext, sizeof(mmap_file_t));
    create_raw_file(drcontext,client_args.data_folder,"dyn",bk->dynamic_file);
    create_raw_file(drcontext,client_args.data_folder,"static",bk->static_file);
  }

  //insert the config string (query)
  if(client_args.mode != SNOOP){
    query_t * query = (query_t *)(data->data + START_QUERY);
    bookkeep_t * bk = (bookkeep_t *)(data->data + START_BK_DATA);
    int sz = insert_config(query, client_args.compiler, client_args.flags, client_args.mode);
    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    //dr_printf("%s\n",query);
    if(client_args.mode == SQLITE){
      query_db(query);
    }
    else if(client_args.mode == RAW_SQL){
      sz = complete_query(query,sz);
      write_to_file(bk->static_file,query,sz);
    }


    sz = get_config(query, client_args.compiler, client_args.flags, client_args.mode);
    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    //dr_printf("%s\n",query);
    if(client_args.mode == SQLITE){
      query_db(query);
    }
    else if(client_args.mode == RAW_SQL){
      sz = complete_query(query,sz);
      write_to_file(bk->dynamic_file,query,sz);
    }

  }

  //dr_printf("thread %d initialized..\n",dr_get_thread_id(drcontext));

}

#define INS_THRESHOLD 3

//bb analysis routines
uint32_t populate_bb_info(void * drcontext, volatile code_info_t * cinfo, volatile bb_data_t * timing, instrlist_t * bb, code_embedding_t code_embedding){

  instr_t * first = instrlist_first(bb);
  instr_t * last  = instrlist_last(bb);
  instr_t * instr;
  int count = 0;
  for(instr = first; instr != last; instr = instr_get_next(instr)){
    count++;
  }
  if (count < INS_THRESHOLD) return -1;
  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = first_pc - md->start;
 
  strcpy(cinfo->module,dr_module_preferred_name(md));
  cinfo->module_start = md->start;
  cinfo->rel_addr = rel_addr;

  code_embedding(drcontext, cinfo, bb);

  timing->meta.slots_filled = 0;
  timing->meta.rel_addr = rel_addr;
  timing->meta.module_start = md->start;
  return 0;
}

//bb instrumentation routines
static inline void save_registers(void * drcontext, instrlist_t * bb, instr_t * where, uint32_t * mask){

  reg_id_t regs[4] = {DR_REG_RAX, DR_REG_RCX, DR_REG_RDX, DR_REG_RBX};
  dr_spill_slot_t spills[4] = {SPILL_SLOT_1,SPILL_SLOT_2,SPILL_SLOT_3,SPILL_SLOT_4};
  int i;
  
  for(i = 0; i < 4; i++){
    if(mask[i]){
      dr_save_reg(drcontext,bb,where, regs[i], spills[i]);
    }
  }

}

static inline void restore_registers(void * drcontext, instrlist_t * bb, instr_t * where, uint32_t * mask){

  reg_id_t regs[4] = {DR_REG_RAX, DR_REG_RCX, DR_REG_RDX, DR_REG_RBX};
  dr_spill_slot_t spills[4] = {SPILL_SLOT_1,SPILL_SLOT_2,SPILL_SLOT_3,SPILL_SLOT_4};
  int i;
  
  for(i = 0; i < 4; i++){
    if(mask[i]){
      dr_restore_reg(drcontext,bb,where, regs[i], spills[i]);
    }
  }


}

void check_used_and_live(instrlist_t * bb, uint32_t * used, uint32_t * live){

  instr_t * instr;
  uint32_t num_srcs, num_dsts;
  uint32_t i,j;
  reg_id_t regs[4] = {DR_REG_RAX, DR_REG_RCX, DR_REG_RDX, DR_REG_RBX};

  for(j = 0; j < 4; j++){
    used[j] = 0;
    live[j] = 0;
  }

  for(instr = instrlist_first(bb); instr != NULL; instr = instr_get_next(instr)){

    num_srcs = instr_num_srcs(instr);
    for(i = 0; i < num_srcs; i++){
      opnd_t op = instr_get_src(instr,i);
      for(j = 0; j < 4; j++){
	if(opnd_uses_reg(op, regs[j])){
	  used[j] = 1;
	}
      }
    }

    num_dsts = instr_num_dsts(instr);
    for(i = 0; i < num_dsts; i++){
      opnd_t op = instr_get_dst(instr,i);
      for(j = 0; j < 4; j++){
	if(opnd_uses_reg(op, regs[j])){
	  live[j] = 1;
	}
      }
    }

  }

}

void timing_instrumentation(void * drcontext, instrlist_t * bb, instr_t * first, instr_t * last, uint32_t bb_num, volatile bookkeep_t * bk){

  
  uint32_t used[4];
  uint32_t live[4];
  uint32_t all[4] = {1,1,1,1};
  uint32_t i;

  instr_t * instr;
  

  check_used_and_live(bb, used, live);

#define REPEAT_BB 3

  for(i = 0; i < REPEAT_BB; i++){
 
    save_registers(drcontext,bb,first, all);
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtscp(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&(bk->prevtime_lo),OPSZ_4), opnd_create_reg(DR_REG_EAX)));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&(bk->prevtime_hi),OPSZ_4), opnd_create_reg(DR_REG_EDX)));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&(bk->proc_before),OPSZ_4), opnd_create_reg(DR_REG_ECX)));
    if(bb_num){
      restore_registers(drcontext,bb,first, all);
      for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
	instrlist_meta_preinsert(bb, first, instr_clone(drcontext,instr));
      }
      save_registers(drcontext,bb,first, all);
    }
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_rdtscp(drcontext));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&(bk->nowtime_lo),OPSZ_4), opnd_create_reg(DR_REG_EAX)));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&(bk->nowtime_hi),OPSZ_4), opnd_create_reg(DR_REG_EDX)));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_mov_st(drcontext, OPND_CREATE_ABSMEM(&(bk->proc_after),OPSZ_4), opnd_create_reg(DR_REG_ECX)));
    instrlist_meta_preinsert(bb,first,INSTR_CREATE_cpuid(drcontext));
    dr_insert_clean_call(drcontext,bb,first,post_cleancall,false,1,opnd_create_immed_uint(bb_num,OPSZ_4));
    restore_registers(drcontext,bb,first, all);
 
  }
 
}

#define REPEAT_TIMES 10

static void measure_overhead(void * drcontext, instrlist_t * bb, volatile bookkeep_t * bk){
  
  instr_t * first = instrlist_first(bb);
  uint32_t i = 0;
  for(i = 0; i < REPEAT_TIMES; i++){
    timing_instrumentation(drcontext,bb,first,first,0,bk);
  }

}

void post_cleancall(uint32_t num_bbs){

  //starting with full dumping
  void * drcontext = dr_get_current_drcontext();
  mmap_file_t * file = dr_get_tls_field(drcontext);   

  volatile query_t * query = (query_t *)(file->data + START_QUERY);
  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);

  if(bk->proc_before != bk->proc_after){
    return;
  }

  
  uint64_t b_hi = bk->prevtime_hi;
  uint64_t b_lo = bk->prevtime_lo;
  uint64_t n_hi = bk->nowtime_hi;
  uint64_t n_lo = bk->nowtime_lo;

  uint64_t before = (b_hi << 32) + b_lo;
  uint64_t now = (n_hi << 32) + n_lo;
  
  //DR_ASSERT(now > before);
  //DR_ASSERT((now - before) < UINT32_MAX);

  
  bb_data_t * timing = (bb_data_t *)(file->data + START_BB_DATA);
  uint32_t slots_filled = timing[num_bbs].meta.slots_filled; 

  uint32_t i = 0;
  
  uint32_t time = now - before;
  //if(timing[num_bbs].meta.rel_addr == 2034)
  //dr_printf("%d,%llu,%llu,%llu\n",timing[num_bbs].meta.rel_addr, before, now, time);



  if (slots_filled >= TIMING_SLOTS){ //dump to data base

    if(num_bbs == 0) return; //first bb is not dumped; it's for calculating the overhead

    //dr_printf("dumping call...\n");
    BEGIN_CONTROL(bk->control,IDLE,DR_CONTROL);
    
    bk->dump_bb = num_bbs;

    bb_data_t * bb_timing = &timing[num_bbs];
    if(client_args.mode != SNOOP){
      module_data_t * md = dr_lookup_module(bb_timing->meta.module_start);
      const char * module_name = dr_module_preferred_name(md);
      //dr_printf("module-%s\n",module_name);
      int sz = 0;
      for(i = 0; i < slots_filled; i++){
	//if(bb_timing->times[i].average < bk->overhead){
	//  dr_fprintf(STDERR,"%s,%llu,%llu,%llu\n",module_name,bb_timing->meta.rel_addr,bb_timing->times[i].average,bk->overhead);
	//}
	//DR_ASSERT(bb_timing->times[i].average > bk->overhead);
	sz = insert_times(query,
			  module_name,
			  bb_timing->meta.rel_addr,
			  bk->arch,
			  bb_timing->times[i].average, 
			  bb_timing->times[i].count,
			  client_args.mode);
	DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
	//dr_printf("%s\n",query);
	if(client_args.mode == SQLITE){
	  query_db(query);
	}
	else if(client_args.mode == RAW_SQL){
	  sz = complete_query(query,sz);
	  write_to_file(bk->dynamic_file,query,sz);
	}
      }
    }

    END_CONTROL(bk->control,DR_CONTROL,DUMP_ONE);

    timing[num_bbs].meta.slots_filled = 0;
    slots_filled = 0;
  }

  if(num_bbs != 0){
    insert_timing(&timing[num_bbs],time);
  }
  else{
    if(bk->overhead > time){
      bk->overhead = time;
    }
    //dr_printf("overhead - %d, %d\n", time, bk->overhead);

  }
 
}

void debug_print(void * drcontext, void * tag, instrlist_t * bb, uint32_t num_bb){
 
  instr_t * instr;
  instr_t * first = instrlist_first_app(bb);
  instr_t * last = instrlist_last(bb);

  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = first_pc - md->start;
  
  /*if((rel_addr == 4112 || rel_addr == 2608) && strcmp("2mm", dr_module_preferred_name(md)) == 0){
    dr_fprintf(STDERR,"%d, %d\n",num_bb, rel_addr);
    for(instr = instrlist_first(bb); instr != NULL; instr = instr_get_next(instr)){
      instr_disassemble(drcontext, instr, STDERR);
      dr_fprintf(STDERR,"\n");
    }
    }*/
 
}

static dr_emit_flags_t
bb_creation_event(void * drcontext, void * tag, instrlist_t * bb, bool for_trace, bool translating){

  instr_t * first;
  instr_t * last;

  first = instrlist_first(bb);
  last = instrlist_last(bb);

  mmap_file_t * file = dr_get_tls_field(drcontext);
  if(file->data == NULL){
    return DR_EMIT_DEFAULT;    
  }


  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);
  volatile code_info_t * cinfo = (code_info_t *)(file->data + START_CODE_DATA);
  volatile bb_data_t * tinfo = (bb_data_t *)(file->data + START_BB_DATA);
  volatile query_t * query = (query_t *)(file->data + START_QUERY);

  //we cannot store more than the buffer size
  if(bk->num_bbs >= NUM_BBS){
    //dr_printf("overflow of num bbs\n");
    return DR_EMIT_DEFAULT;
  }

  //first basic block is used to measure the overhead
  if(bk->num_bbs == 0){
    populate_bb_info(drcontext,cinfo,&tinfo[bk->num_bbs],bb,client_args.embedding_func);
    measure_overhead(drcontext, bb, bk);
    bk->num_bbs++;
    return DR_EMIT_DEFAULT;
  }

  
  //we are filtering based on exec
  module_data_t * md = dr_lookup_module(instr_get_app_pc(first));

  if(!filter_based_on_module(dr_module_preferred_name(md))){
    return DR_EMIT_DEFAULT;
  }

  //bb analysis 
  BEGIN_CONTROL(cinfo->control,IDLE,DR_CONTROL);

  if(populate_bb_info(drcontext,cinfo,&tinfo[bk->num_bbs],bb,client_args.embedding_func)){
    return DR_EMIT_DEFAULT;
  }

  if(cinfo->code_size == -1){ //we couldn't record the entire basic block
    return DR_EMIT_DEFAULT;
  }

  if(client_args.mode != SNOOP){
    int sz = insert_code(query,cinfo->module,cinfo->rel_addr,cinfo->code, client_args.mode, cinfo->code_size);
    if(sz == -1){ //we couldn't record the query in full
      return DR_EMIT_DEFAULT;
    }
    //dr_printf("%s\n",query);
    DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
    if(client_args.mode == SQLITE){
      query_db(query);
    }
    else if(client_args.mode == RAW_SQL){
      sz = complete_query(query,sz);
      write_to_file(bk->static_file,query,sz);
    }
  }
 
  END_CONTROL(cinfo->control,DUMP_ONE,DR_CONTROL);

  //app_pc first_pc = instr_get_app_pc(first);
  //md = dr_lookup_module(first_pc);
  //uint32_t rel_addr = first_pc - md->start;
  //uint32_t save = 1;

  //if(rel_addr == 2034 && strcmp("2mm", dr_module_preferred_name(md)) == 0){
  //  save = 0;
  //}

  //debug_print(drcontext, tag, bb, bk->num_bbs);
  //bb instrumentation
  //timing_instrumentation(drcontext, bb, first, first, 0, bk);
  //timing_instrumentation(drcontext, bb, first, first, 0, bk);
  timing_instrumentation(drcontext, bb, first, last, bk->num_bbs, bk);
  //debug_print(drcontext, tag, bb, bk->num_bbs);

  bk->num_bbs++;

  return DR_EMIT_DEFAULT;
 
}



  /* thread exit event */
static void
thread_exit(void * drcontext){


  //dr_printf("thread exiting %d...\n",dr_get_thread_id(drcontext));
  mmap_file_t * file = dr_get_tls_field(drcontext);
  volatile bookkeep_t * bk = (bookkeep_t *)(file->data + START_BK_DATA);
  volatile bb_data_t * timing = (bb_data_t *)(file->data + START_BB_DATA);
  volatile query_t * query = (query_t *)(file->data + START_QUERY);
  
  BEGIN_CONTROL(bk->control,IDLE,DR_CONTROL);
 
  if(client_args.mode != SNOOP){
    int i = 0;
    int j = 0;
    
    for(i = 1; i < bk->num_bbs; i++){
      bb_data_t * bb_time = &timing[i];
      uint32_t slots_filled = bb_time->meta.slots_filled;
      module_data_t * md = dr_lookup_module(bb_time->meta.module_start);
      const char * module_name = dr_module_preferred_name(md);
      int sz = 0;
      for(j = 0; j < slots_filled; j++){
	//if(bb_time->times[j].average < bk->overhead){
	//  dr_fprintf(STDERR,"%s,%llu,%llu,%llu\n",module_name,bb_time->meta.rel_addr,bb_time->times[j].average,bk->overhead);
	//}
	//DR_ASSERT(bb_time->times[j].average > bk->overhead);
	sz = insert_times(query,
		     module_name,
		     bb_time->meta.rel_addr,
		     bk->arch,
		     bb_time->times[j].average,
		     bb_time->times[j].count,  
		     client_args.mode);
	DR_ASSERT(sz <= MAX_QUERY_SIZE - 2);
	//dr_printf("%s\n",query);
	if(client_args.mode == SQLITE){
	  query_db(query);
	}
	else if(client_args.mode == RAW_SQL){
	  sz = complete_query(query,sz);
	  write_to_file(bk->dynamic_file,query,sz);
	}
      }
    }
  }

  END_CONTROL(bk->control,DR_CONTROL,DUMP_ALL);
  
  if(client_args.mode == RAW_SQL){
    close_raw_file(bk->dynamic_file);
    close_raw_file(bk->static_file);
  }
  close_memory_map_file(file,TOTAL_SIZE);

  //dr_printf("thread exiting %d...\n",dr_get_thread_id(drcontext));
 
}

static void
event_exit(void)
{
  dr_mutex_destroy(mutex);
    
  if(client_args.mode == SNOOP){
    volatile thread_files_t * files = filenames_file.data;
    files->control = EXIT;
    while(files->control != IDLE);
    close_memory_map_file(&filenames_file,sizeof(thread_files_t));
  }
  if(client_args.mode == SQLITE){
    connection_close();
  }

}


DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{

  //dr_printf("timing client starting...\n");
    
    /* register events */
    dr_register_thread_init_event(thread_init);    
    dr_register_bb_event(bb_creation_event);
    dr_register_thread_exit_event(thread_exit);
    dr_register_exit_event(event_exit);
    
    disassemble_set_syntax(DR_DISASM_INTEL);

    /* client arguments */
    DR_ASSERT(argc == 6);
    client_args.mode = atoi(argv[1]);
    client_args.code_format = atoi(argv[2]);
    strncpy(client_args.compiler,argv[3], MAX_STRING_SIZE);
    strncpy(client_args.flags,argv[4], MAX_STRING_SIZE);
    strncpy(client_args.data_folder, argv[5], MAX_STRING_SIZE);


    if(client_args.code_format == TEXT){
      client_args.embedding_func = textual_embedding;
    }
    else if(client_args.code_format == TOKEN){
      client_args.embedding_func = token_text_embedding;
    }

    mutex = dr_mutex_create();
    num_threads = 0;

    //dr_printf("mode - %d\n",client_args.mode);
    
    if(client_args.mode == SNOOP){
      /* open filenames_file and mmap it */
      strcpy(filenames_file.filename,FILENAMES_FILE);
      filenames_file.filled = 0;
      filenames_file.offs = 0;
      filenames_file.file = dr_open_file(filenames_file.filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
      create_memory_map_file(&filenames_file,sizeof(thread_files_t));
      memset(filenames_file.data,0,sizeof(thread_files_t));
    }
    else if(client_args.mode == SQLITE){
      connection_init();
    }

    //dr_printf("op start %d, reg_start %d, int %d, float %d, mem start %d\n",OPCODE_START,REG_START,INT_IMMED,FLOAT_IMMED,MEMORY_START);
  
}
