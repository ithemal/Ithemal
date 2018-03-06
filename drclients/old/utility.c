#include "utility.h"

//getting filename for database
int get_filename(void * drcontext, char * filename, size_t max_size){

    thread_id_t id = dr_get_thread_id(drcontext);
    dr_time_t time;
    dr_get_time(&time);
    return dr_snprintf(filename, max_size, "/tmp/%s_%d_%d_%d.txt", dr_get_application_name(), id, time.hour, time.minute);

}

//creating memory mapped files
void create_memory_map_file(mmap_file_t * file_map, size_t size){
  
    file_map->file = dr_open_file(file_map->filename, DR_FILE_WRITE_OVERWRITE | DR_FILE_READ);
    DR_ASSERT(file_map->file);

    DR_ASSERT(dr_file_seek(file_map->file,size-1,DR_SEEK_SET));
    DR_ASSERT(dr_write_file(file_map->file,"",1));

    file_map->data = dr_map_file(file_map->file, &size, 0, NULL, DR_MEMPROT_READ | DR_MEMPROT_WRITE, 0);
    DR_ASSERT(file_map->data);
 
}


void get_code_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  int pos = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - pos);
    cinfo->code[pos] = '\n';
  }
  cinfo->code[pos] = '\0';

}

void populate_timing(bb_data_t * bb, uint32_t newtime){

}

void populate_code(void * drcontext, code_info_t * cinfo, bb_data_t * time, instrlist_t * bb){

  instr_t * first = instrlist_first(bb);
  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = (int)first_pc - (int)md->start;
  
  strcpy(cinfo->module,dr_module_preferred_name(md));
  cinfo->module_start = md->start;
  cinfo->rel_addr = rel_addr;

  get_code_embedding(drcontext, cinfo, bb);

  timing->meta.slots_filled = 0;
  timing->meta.rel_addr = rel_addr;
  timing->meta.module_start = md->start;

}

void insert_code(void * drcontext, volatile code_info_t * cinfo, instrlist_t * bb){
  
  while(!__sync_bool_compare_and_swap(&cinfo->control,IDLE,DR_CONTROL));

  instr_t * first = instrlist_first(bb);
  app_pc first_pc = instr_get_app_pc(first);
  module_data_t * md = dr_lookup_module(first_pc);
  uint32_t rel_addr = (int)first_pc - (int)md->start;
  
  strcpy(cinfo->module,dr_module_preferred_name(md));
  cinfo->module_start = md->start;
  cinfo->rel_addr = rel_addr;

  instr_t * instr;
  int pos = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - pos);
    cinfo->code[pos] = '\n';
  }
  cinfo->code[pos] = '\0';

  //dr_printf("%s,%d\n",cinfo->code,pos);
  DR_ASSERT(__sync_bool_compare_and_swap(&cinfo->control,DR_CONTROL,DUMP_ONE));

  while(cinfo->control != IDLE);
  
}

