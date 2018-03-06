#include "timing_logic.h"

//getting filename for database
int get_filename(void * drcontext, char * filename, size_t max_size){

    thread_id_t id = dr_get_thread_id(drcontext);
    dr_time_t time;
    dr_get_time(&time);
    return dr_snprintf(filename, max_size, "/tmp/%s_%d_%d_%d.txt", dr_get_application_name(), id, time.hour, time.minute);

}


void get_code_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  int pos = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - 1 -  pos);
    DR_ASSERT(pos <= MAX_CODE_SIZE - 1);
    cinfo->code[pos] = '\n';
  }
  cinfo->code[pos] = '\0';

}

#define PERCENTAGE_THRESHOLD 10

void insert_timing(bb_data_t * bb, uint32_t time){

  DR_ASSERT(sizeof(bb_data_t) == sizeof(uint32_t) * TIME_SLOTS);
  uint32_t slots_filled = bb->meta.slots_filled;

  int i = 0;
  
  uint32_t found = 0;
  for(i = 0; i < slots_filled; i++){
    uint32_t av = bb->times[i].average;
    uint32_t count = bb->times[i].count;
    uint32_t abs_diff = (time > av) ? time - av : av - time;
    av = (av == 0) ? av + 1 : av;
    uint32_t per_diff = (abs_diff * 100) / av;

    if(per_diff < PERCENTAGE_THRESHOLD){
      //dr_printf("%d\n",per_diff);
      bb->times[i].count++;
      bb->times[i].average = (av * count + time)/(count + 1);
      found = 1;
      break;
    }
  }

  if(!found){
    bb->times[slots_filled].count = 1;
    bb->times[slots_filled].average = time;
    bb->meta.slots_filled++;
  }
  
}


uint32_t filter_based_on_exec(const char * module_name){

  return strcmp(dr_get_application_name(),module_name) == 0;

}


