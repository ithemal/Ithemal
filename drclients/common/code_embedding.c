#include "code_embedding.h"

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
    value = MEMORY_START + *mem;
    (*mem)++;
  }
  else{
    opnd_disassemble(drcontext,op,STDOUT);
    dr_printf("\n");
  }

  DR_ASSERT(value); //should have a non-zero value
  DR_ASSERT(!opnd_is_pc(op)); //we do not consider branch instructions
  
  return dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos ,"%d,", value);   

}

bool filter_instr(instr_t * instr){

  //first it cannot be a rip relative instruction
  if(instr_has_rel_addr_reference(instr)){
    return true;
  }


  uint32_t tainted[12] = {DR_REG_R13, DR_REG_R13D, DR_REG_R13W, DR_REG_R13L,
			  DR_REG_R14, DR_REG_R14D, DR_REG_R14W, DR_REG_R14L,
			  DR_REG_R15, DR_REG_R15D, DR_REG_R15W, DR_REG_R15L};

  uint32_t i = 0;

  for(i = 0; i < 12; i++){
    if(instr_reg_in_dst(instr, tainted[i])){
      return true;
    } 
  }
  return false;

}


void token_text_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){
  instr_t * instr;
  int pos = 0;
  int i = 0;
  int ret = 0;
  
  char * cpos = cinfo->code;

  uint32_t mem = 0;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    if(filter_instr(instr)) continue;

    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos ,"%d,%d,", OPCODE_START + instr_get_opcode(instr), DELIMITER);
    if(ret != -1) pos += ret;
    else { cinfo->code_size = -1; return; }
    

    opnd_t op;
    for(i = 0; i < instr_num_srcs(instr); i++){
      op = instr_get_src(instr,i);
      ret = tokenize_text_operand(drcontext, cpos, pos, op, &mem);
      if(ret != -1) pos += ret;
      else { cinfo->code_size = -1; return; }
    }

    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos, "%d,", DELIMITER);
    if(ret != -1) pos += ret;
    else { cinfo->code_size = -1; return; }
   
    for(i = 0; i < instr_num_dsts(instr); i++){
      op = instr_get_dst(instr,i);
      ret = tokenize_text_operand(drcontext, cpos, pos, op, &mem);
      if(ret != -1) pos += ret;
      else { cinfo->code_size = -1; return; }
    }

    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos, "%d,", DELIMITER);
    if(ret != -1) pos += ret;
    else { cinfo->code_size = -1; return; }
    
  }

  cinfo->code_size = pos;
  
}



void textual_embedding(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  int pos = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    if(filter_instr(instr)) continue;
    
    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - 1 -  pos);
    cinfo->code[pos++] = '\n';
    DR_ASSERT(pos <= MAX_CODE_SIZE);
  }

  cinfo->code_size = pos;

}

