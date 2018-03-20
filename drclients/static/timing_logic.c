#include "timing_logic.h"

//getting filename for database
int get_filename(void * drcontext, char * filename, size_t max_size){

    thread_id_t id = dr_get_thread_id(drcontext);
    dr_time_t time;
    dr_get_time(&time);
    return dr_snprintf(filename, max_size, "/tmp/%s_%d_%d_%d.txt", dr_get_application_name(), id, time.hour, time.minute);

}

//we may decide to skip basic blocks with certain instrs and operands

#define DELIMITER -1

void tokenize_operand(void * drcontext, uint16_t * cpos, opnd_t op, uint32_t * mem){
  
  uint16_t value = 0;

  //dr_printf("%d,%d,%d,%d\n",opnd_is_reg(op),opnd_is_immed_int(op),opnd_is_immed_float(op),opnd_is_memory_reference(op));

  //registers
  if(opnd_is_reg(op)){
    value = REG_START + opnd_get_reg(op);
  }
  //immediates
  else if(opnd_is_immed_int(op)){
    value = INT_IMMED;
  }
  else if(opnd_is_immed_float(op)){
    value = FLOAT_IMMED;
  }
  //memory :(
  else if(opnd_is_memory_reference(op)){
    value = MEMORY_START + *mem;
    (*mem)++;
  }
  else{
    opnd_disassemble(drcontext,op,STDOUT);
    dr_printf("\n");
  }

  DR_ASSERT(value); //should have a non-zero value
  DR_ASSERT(!opnd_is_pc(op)); //we do not consider branch instructions
  
  *cpos = value;

}


void token_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){
  instr_t * instr;
  int pos = 0;
  int i = 0;
  
  uint16_t * cpos = cinfo->code;

  uint32_t mem = 0;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    uint16_t opcode = instr_get_opcode(instr);   
    cpos[pos++] = OPCODE_START + opcode;

    opnd_t op;
    for(i = 0; i < instr_num_srcs(instr); i++){
      op = instr_get_src(instr,i);
      tokenize_operand(drcontext, &cpos[pos], op, &mem);
      pos++;
    }
    for(i = 0; i < instr_num_dsts(instr); i++){
      op = instr_get_dst(instr,i);
      tokenize_operand(drcontext, &cpos[pos], op, &mem);
      pos++;
    }

    //delimiter
    cpos[pos++] = DELIMITER;
    
  }

  cinfo->code_size = sizeof(uint16_t) * pos;
  
}


int tokenize_text_operand(void * drcontext, char * cpos, uint32_t pos, opnd_t op, uint32_t * mem){
  
  uint16_t value = 0;

  //registers
  if(opnd_is_reg(op)){
    value = REG_START + opnd_get_reg(op);
  }
  //immediates
  else if(opnd_is_immed_int(op)){
    value = INT_IMMED;
  }
  else if(opnd_is_immed_float(op)){
    value = FLOAT_IMMED;
  }
  //memory :(
  else if(opnd_is_memory_reference(op)){
    value = MEMORY_START + *mem++;
  }
  else{
    opnd_disassemble(drcontext,op,STDOUT);
    dr_printf("\n");
  }

  DR_ASSERT(value); //should have a non-zero value
  DR_ASSERT(!opnd_is_pc(op)); //we do not consider branch instructions
  
  return dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos ,"%d,", value);   

}


void token_text_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){
  instr_t * instr;
  int pos = 0;
  int i = 0;
  
  char * cpos = cinfo->code;

  uint32_t mem = 0;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    pos += dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos ,"%d,", OPCODE_START + instr_get_opcode(instr));   

    opnd_t op;
    for(i = 0; i < instr_num_srcs(instr); i++){
      op = instr_get_src(instr,i);
      pos += tokenize_text_operand(drcontext, cpos, pos, op, &mem);
    }
    for(i = 0; i < instr_num_dsts(instr); i++){
      op = instr_get_dst(instr,i);
      pos += tokenize_text_operand(drcontext, cpos, pos, op, &mem);
    }
    
  }

  cinfo->code_size = pos;
  
}



void textual_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  int pos = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - 1 -  pos);
    DR_ASSERT(pos <= MAX_CODE_SIZE - 1);
    cinfo->code[pos] = '\n';
  }

  cinfo->code_size = pos;

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


