#include "utility.h"

//getting filename for database
int get_filename(void * drcontext, char * filename, size_t max_size){

    thread_id_t id = dr_get_thread_id(drcontext);
    dr_time_t time;
    dr_get_time(&time);
    return dr_snprintf(filename, max_size, "%s_%d_%d_%d.txt", dr_get_application_name(), id, time.hour, time.minute);

}

//functions to create .sql file
void get_config(void * drcontext, file_t file, const char * compiler, const char * flags){

  dr_fprintf(file,"USE costmodel;\n");
  dr_fprintf(file,"INSERT INTO config (compiler, flags) VALUES ('%s','%s');\n",compiler,flags);
  dr_fprintf(file, "SET @ci = (SELECT config_id FROM config WHERE compiler = '%s' AND flags ='%s');\n",compiler,flags);

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

  dr_printf("%s,%d\n",cinfo->code,pos);
  DR_ASSERT(__sync_bool_compare_and_swap(&cinfo->control,DR_CONTROL,DUMP_ONE));

  while(cinfo->control != IDLE);
  
}

void insert_time(file_t file, uint32_t time, uint32_t rel_addr, const char * program, uint32_t arch){
 
  dr_fprintf(file, "INSERT INTO times (code_id, arch, time) VALUES ((SELECT code_id FROM code WHERE config_id = @ci AND rel_addr = %d AND program = '%s'),%d,%d);\n",rel_addr,program,arch,time);

}
