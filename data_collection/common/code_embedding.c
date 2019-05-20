#include "code_embedding.h"
#include "disasm_corrector.h"
#include <string.h>

//delimiter token types
#define SRC_DEM   -1
#define DST_DEM   -2
#define END_DEM   -3
#define MEM_TAG   -4


//tokenizing code
#define REG_START 0
#define INT_IMMED REG_START + DR_REG_YMM15 + 1
#define FLOAT_IMMED INT_IMMED + 1
#define OPCODE_START FLOAT_IMMED + 1
#define MEMORY_START OPCODE_START + OP_LAST + 1


static void print_opnds(instr_t * instr){

  char temp[MNEMONIC_SIZE];

  void * drcontext = dr_get_current_drcontext();

  int num_srcs = instr_num_srcs(instr);
  int i = 0;

  dr_printf("srcs:");
  for(i = 0; i < num_srcs; i++){
    opnd_t op = instr_get_src(instr,i);
    opnd_disassemble_to_buffer(drcontext, op, temp, MNEMONIC_SIZE);
    dr_printf("%s,",temp);
  }
  dr_printf("\n");

  int num_dsts = instr_num_dsts(instr);
  i = 0;

  dr_printf("dsts:");
  for(i = 0; i < num_dsts; i++){
    opnd_t op = instr_get_dst(instr,i);
    opnd_disassemble_to_buffer(drcontext, op, temp, MNEMONIC_SIZE);
    dr_printf("%s,",temp);
  }
  dr_printf("\n");

}



bool filter_instr(instr_t * instr){

  //first it cannot be a rip relative instruction
  if(instr_has_rel_addr_reference(instr)){
    return true;
  }

  int i;
  uint32_t omitted[10] = {OP_rep_ins, OP_rep_outs, OP_rep_movs, OP_rep_stos, OP_rep_lods, OP_rep_cmps, OP_rep_scas, OP_repne_cmps, OP_repne_scas, OP_xbegin};

  for(i = 0; i <10; i++){
    if(instr_get_opcode(instr) == omitted[i]){
      return true;
    }
  }

  return false;

}

bool raw_bits(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  byte * pc = cinfo->code;
  int i = 0;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    if(instr_is_app(instr) && !filter_instr(instr)){
      int length = instr_length(drcontext, instr);
      for(i = 0; i < length; i++){
	*pc = instr_get_raw_byte(instr, i);
	pc++;
      }
    }
  }
  cinfo->code_size = pc - (byte *)cinfo->code;
  return true;

}



void raw_token_operand(void * drcontext, uint16_t * cpos, opnd_t op, uint32_t * mem){

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


bool raw_token(void * drcontext, code_info_t * cinfo, instrlist_t * bb){
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
      raw_token_operand(drcontext, &cpos[pos], op, &mem);
      pos++;
    }
    for(i = 0; i < instr_num_dsts(instr); i++){
      op = instr_get_dst(instr,i);
      raw_token_operand(drcontext, &cpos[pos], op, &mem);
      pos++;
    }

    //delimiter
    cpos[pos++] = END_DEM;

  }

  cinfo->code_size = sizeof(uint16_t) * pos;
  return true;
}

void op_error(void* drcontext, opnd_t op, instr_t* instr) {
  fprintf(stderr, "Illegal (or unknown) operand:\n");
  opnd_disassemble(drcontext, op, STDERR);
  fprintf(stderr, "\nIn instruction:\n");
  instr_disassemble(drcontext, instr, STDERR);
  fprintf(stderr, "\n");
  exit(-1);
}

int text_token_operand(void * drcontext, char * cpos, uint32_t pos, opnd_t op, instr_t * instr){

#define MAX_TOKENS 10

  int tokens[MAX_TOKENS];
  uint32_t num_tokens = 0;

  //registers
  if(opnd_is_reg(op)){
    tokens[num_tokens++] = REG_START + opnd_get_reg(op);
  }
  //immediates
  else if(opnd_is_immed_int(op) || opnd_is_immed_int64(op)){
    tokens[num_tokens++] = INT_IMMED;
  }
  else if(opnd_is_immed_float(op)){
    tokens[num_tokens++] = FLOAT_IMMED;
  }
  //memory :(
  else if(opnd_is_memory_reference(op)){

    tokens[num_tokens++] = MEM_TAG;
    if(opnd_is_base_disp(op)){

      reg_id_t base = opnd_get_base(op);
      reg_id_t index = opnd_get_index(op);
      int disp = opnd_get_disp(op);

      if(base != REG_NULL)
	tokens[num_tokens++] = REG_START + opnd_get_base(op);
      if(index != REG_NULL)
	tokens[num_tokens++] = REG_START + opnd_get_index(op);
      if(disp != 0)
	tokens[num_tokens++] = INT_IMMED;
    }
    else if(opnd_is_abs_addr(op)){
      tokens[num_tokens++] = INT_IMMED;
    }
    else if(opnd_is_rel_addr(op)){
      tokens[num_tokens++] = INT_IMMED;
    }
    else{
      instr_disassemble(drcontext, instr, STDOUT);
      opnd_disassemble(drcontext, op, STDOUT);
      exit(-1);
    }
    tokens[num_tokens++] = MEM_TAG;

  }
  else{
    instr_disassemble(drcontext, instr, STDOUT);
    opnd_disassemble(drcontext, op, STDOUT);
    exit(-1);
  }

  DR_ASSERT(num_tokens > 0); //should have at least one token
  DR_ASSERT(!opnd_is_pc(op)); //we do not consider branch instructions

  int i;
  int written = 0;
  int ret = 0;

  for(i = 0; i < num_tokens; i++){
    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos, "%d,", tokens[i]);
    if(ret != -1){
      written += ret; pos += ret;
    }
    else return ret;
  }

  return written;

}



bool text_token(void * drcontext, code_info_t * cinfo, instrlist_t * bb){
  instr_t * instr;
  int pos = 0;
  int i = 0;
  int ret = 0;

  char * cpos = cinfo->code;

  uint32_t mem = 0;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    if(filter_instr(instr)) continue;

    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos ,"%d,%d,", OPCODE_START + instr_get_opcode(instr), SRC_DEM);
    if(ret != -1) pos += ret;
    else { cinfo->code_size = -1; return false; }

    opnd_t op;
    for(i = 0; i < instr_num_srcs(instr); i++){
      op = instr_get_src(instr,i);
      ret = text_token_operand(drcontext, cpos, pos, op, instr);
      if(ret != -1) pos += ret;
      else { cinfo->code_size = -1; return false; }
    }

    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos, "%d,", DST_DEM);
    if(ret != -1) pos += ret;
    else { cinfo->code_size = -1; return false; }

    for(i = 0; i < instr_num_dsts(instr); i++){
      op = instr_get_dst(instr,i);
      ret = text_token_operand(drcontext, cpos, pos, op, instr);
      if(ret != -1) pos += ret;
      else { cinfo->code_size = -1; return false; }
    }

    ret = dr_snprintf(cpos + pos, MAX_CODE_SIZE - pos, "%d,", END_DEM);
    if(ret != -1) pos += ret;
    else { cinfo->code_size = -1; return false; }

  }

  cinfo->code_size = pos;
  return true;

}


bool text_intel(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  disassemble_set_syntax(DR_DISASM_INTEL);

  instr_t * instr;
  int pos = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    if(filter_instr(instr)) continue;

    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - 1 -  pos);
    cinfo->code[pos++] = '\n';

    if(pos > MAX_CODE_SIZE){
      cinfo->code[MAX_CODE_SIZE - 1] = '\0';
      dr_printf("max code size reached %s:\n", cinfo->code);
    }

  }

  cinfo->code_size = pos;
  return true;

}



bool text_att(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  disassemble_set_syntax(DR_DISASM_ATT);

  instr_t * instr;
  int pos = 0;

  char disasm[BUFFER_SIZE];

  ins_t ins;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    if(filter_instr(instr)) continue;

    int length = instr_disassemble_to_buffer(drcontext, instr, disasm, BUFFER_SIZE);

    //dr_printf("raw:%s:\n",disasm);
    if(!parse_instr_att(disasm, length, &ins)){
      //dr_printf("parse error %s:\n",disasm);
      return false;
    }
    //dr_printf("before:");
    //print_instr(&ins);

    correct_disasm_att(drcontext, &ins, instr);
    //dr_printf("after:");
    //print_instr(&ins);

    int j = 0;
    int w = 0;

    w = dr_snprintf(cinfo->code + pos, MAX_CODE_SIZE - pos, "%s ", ins.name);
    DR_ASSERT(w > 0);
    pos += w;

    for(j = 0; j < ins.num_ops; j++){
      w = dr_snprintf(cinfo->code + pos, MAX_CODE_SIZE - pos, "%s", ins.operands[j].name);

      DR_ASSERT(w > 0);
      pos += w;
      if(j != ins.num_ops - 1){
	w = dr_snprintf(cinfo->code + pos, MAX_CODE_SIZE - pos, ", ");
	DR_ASSERT(w > 0);
	pos += w;
      }
    }
    w = dr_snprintf(cinfo->code + pos, MAX_CODE_SIZE - pos,  "\n");
    DR_ASSERT(w > 0);
    pos += w;

    if(pos > MAX_CODE_SIZE){
      cinfo->code[MAX_CODE_SIZE - 1] = '\0';
      dr_printf("max size reached %s:\n",cinfo->code);
    }

    DR_ASSERT(pos <= MAX_CODE_SIZE);


  }


  cinfo->code_size = pos;
  return true;
}

uint32_t text_xml_operand(void * drcontext, char * buf, uint32_t bufidx, opnd_t op, instr_t * instr) {
  bufidx += sprintf(&buf[bufidx], "<operand>");

  int tokens[256];
  uint32_t num_tokens = 0;

  if(opnd_is_reg(op)) {
    bufidx += sprintf(&buf[bufidx], "%d", REG_START + opnd_get_reg(op));
  } else if(opnd_is_immed_int(op) || opnd_is_immed_int64(op)){
    bufidx += sprintf(&buf[bufidx], "%d", INT_IMMED);
  } else if(opnd_is_immed_float(op)){
    bufidx += sprintf(&buf[bufidx], "%d", FLOAT_IMMED);
  } else if(opnd_is_memory_reference(op)) {
    bufidx += sprintf(&buf[bufidx], "<mem>");
    if(opnd_is_base_disp(op)) {
      reg_id_t base = opnd_get_base(op);
      reg_id_t index = opnd_get_index(op);
      int disp = opnd_get_disp(op);

      if (base != REG_NULL) {
        bufidx += sprintf(&buf[bufidx], "<base>%d</base>", REG_START + base);
      }
      if (index != REG_NULL) {
        bufidx += sprintf(&buf[bufidx], "<index>%d</index>", REG_START + index);
      }
      if (disp != 0) {
        bufidx += sprintf(&buf[bufidx], "<disp>%d</disp>", INT_IMMED);
      }
    } else if (opnd_is_abs_addr(op)) {
      bufidx += sprintf(&buf[bufidx], "%d", INT_IMMED);
    } else if (opnd_is_rel_addr(op)) {
      bufidx += sprintf(&buf[bufidx], "%d", INT_IMMED);
    } else {
      op_error(drcontext, op, instr);
    }
    bufidx += sprintf(&buf[bufidx], "</mem>");
  } else {
    op_error(drcontext, op, instr);
  }

  bufidx += sprintf(&buf[bufidx], "</operand>");

  return bufidx;
}


bool text_xml(void * drcontext, code_info_t * cinfo, instrlist_t * bb){
  char* buf = cinfo->code;
  uint32_t bufidx = 0;
  bufidx += sprintf(&buf[bufidx], "<block>");

  instr_t* instr;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)) {
    if(filter_instr(instr)) {
      continue;
    }

    bufidx += sprintf(&buf[bufidx], "<instr>");
    bufidx += sprintf(&buf[bufidx], "<opcode>%d</opcode>", OPCODE_START + instr_get_opcode(instr));
    bufidx += sprintf(&buf[bufidx], "<srcs>");

    int i;
    for(i = 0; i < instr_num_srcs(instr); i++) {
      opnd_t op = instr_get_src(instr, i);
      bufidx = text_xml_operand(drcontext, buf, bufidx, op, instr);
    }

    bufidx += sprintf(&buf[bufidx], "</srcs><dsts>");

    for(i = 0; i < instr_num_dsts(instr); i++){
      opnd_t op = instr_get_dst(instr,i);
      bufidx = text_xml_operand(drcontext, buf, bufidx, op, instr);
    }

    bufidx += sprintf(&buf[bufidx], "</dsts>");
    bufidx += sprintf(&buf[bufidx], "</instr>");
  }


  bufidx += sprintf(&buf[bufidx], "</block>");
  cinfo->code_size = bufidx;
  return true;
}
