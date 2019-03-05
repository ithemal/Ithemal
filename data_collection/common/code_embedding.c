#include "code_embedding.h"
#include "change_opcode.h"
#include <string.h>

#define DELIMITER -1

//tokenizing code
#define REG_START 0
#define INT_IMMED REG_START + DR_REG_YMM15 + 1
#define FLOAT_IMMED INT_IMMED + 1
#define OPCODE_START FLOAT_IMMED + 1
#define MEMORY_START OPCODE_START + OP_LAST + 1


//operand types
#define MEM_TYPE 1
#define IMM_TYPE 2
#define REG_TYPE 3

#define MNEMONIC_SIZE 64
#define NUM_OPS 8
#define BUFFER_SIZE 1024


typedef struct {
  uint32_t type;
  char name[MNEMONIC_SIZE];
} op_t;

typedef struct {
  op_t operands[NUM_OPS];
  char name[MNEMONIC_SIZE];
  int num_ops;
} ins_t;



void print_instr(ins_t * ins){

  dr_printf("ins: ");
  dr_printf("%s ",ins->name);
  int i = 0;
  for(; i < ins->num_ops; i++){
    dr_printf("%s, ",ins->operands[i].name);
  }
  dr_printf("\n");

}

//assumes a cleaned up instruction with only valid mnemonics
bool parse_instr_att(const char * buffer, int length, ins_t * instr){
  
  int i = 0;

  instr->num_ops = 0;

  //skip white space
  while(i < length && buffer[i] == ' ') i++;

  uint32_t start_opcode = i;
  while(buffer[i] != ' '){
    instr->name[i - start_opcode] = buffer[i];
    i++;
  }
  instr->name[i - start_opcode] = '\0';

  if(strcmp(instr->name,"rep") == 0 || strcmp(instr->name,"repne") == 0){  //handle repetition opcodes correctly
    instr->num_ops = 0;
    while(i < length){
      instr->name[i - start_opcode] = buffer[i];
      i++;
    }
    instr->name[i - start_opcode] = '\0';
    return true;
  }


  //skip white space
  while(i < length && buffer[i] == ' ') i++;

  uint32_t start_operand = i;
  uint32_t op_num = 0;
  instr->num_ops = 0;

  while(i < length){
    
    if(buffer[i] == '$'){
      instr->operands[op_num].type = IMM_TYPE;
      while(i < length && buffer[i] != ',' && buffer[i] != ' '){
	instr->operands[op_num].name[i - start_operand] = buffer[i];
	i++;
      }
      instr->operands[op_num].name[i - start_operand] = '\0';
      while(i < length && buffer[i] != ',') i++;
    }
    else{ //can be memory or reg
      if(buffer[i] != '%'){ //then it is memory for sure
	instr->operands[op_num].type = MEM_TYPE;
	bool found_open = false;
	while(buffer[i] != ')'){
	  if(i >= length) return false;
	  if(buffer[i] == '(') found_open = true;
	  instr->operands[op_num].name[i - start_operand] = buffer[i];
	  i++;
	}
	if(!found_open) return false;
	instr->operands[op_num].name[i - start_operand] = buffer[i];
	instr->operands[op_num].name[i - start_operand + 1] = '\0';
	while(i < length && buffer[i] != ',') i++;
      }
      else{
	if(i + 3 < length && buffer[i] == ':'){  //segment register
	  instr->operands[op_num].type = MEM_TYPE;
	  //if , comes before (
	  int j = i + 4;
	  while(j < length){
	    if(buffer[j] == '(' || buffer[j] == ',') break;
	    j++;
	  }
	  if(j < length && buffer[j] == '('){ //has base index
	    bool found_open = false;
	    while(buffer[i] != ')'){
	      if(i >= length) return false;      
	      if(buffer[i] == '(') found_open = true;
	      instr->operands[op_num].name[i - start_operand] = buffer[i];
	      i++;
	    }
	    if(!found_open) return false;
	    instr->operands[op_num].name[i - start_operand] = buffer[i];
	    instr->operands[op_num].name[i - start_operand + 1] = '\0';
	    while(i < length && buffer[i] != ',') i++;
	  }
	  else if(j < length && buffer[j] == ','){ //no base index
	    while(i < length && buffer[i] != ',' && buffer[i] != ' '){
	      instr->operands[op_num].name[i - start_operand] = buffer[i];
	      i++;
	    }
	    instr->operands[op_num].name[i - start_operand] = '\0';
	    while(i < length && buffer[i] != ',') i++;
	  }
	  else if(j == length){ //final operand
	    while(i < length && buffer[i] != ' '){
	      instr->operands[op_num].name[i - start_operand] = buffer[i];
	      i++;
	    }
	    instr->operands[op_num].name[i - start_operand] = '\0';
	    while(i < length && buffer[i] != ',') i++;
	  }
	}
	else{
	  instr->operands[op_num].type = REG_TYPE;
	  while(i < length && buffer[i] != ' ' && buffer[i] != ','){
	    instr->operands[op_num].name[i - start_operand] = buffer[i];
	    i++;
	  }
	  instr->operands[op_num].name[i - start_operand] = '\0';
	  while(i < length && buffer[i] != ',') i++;
	}
      }

    }

   
    i++;
    while(i < length && buffer[i] == ' ') i++;
    start_operand = i;
    op_num++;

  }

  instr->num_ops = op_num;

  return true;
  

}


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
  else if(opnd_is_immed_int(op) || opnd_is_immed_int64(op)){
    value = INT_IMMED;
  }
  else if(opnd_is_rel_addr(op)){ 
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

  uint32_t omitted[10] = {OP_rep_ins, OP_rep_outs, OP_rep_movs, OP_rep_stos, OP_rep_lods, OP_rep_cmps, OP_rep_scas, OP_repne_cmps, OP_repne_scas, OP_xbegin};

  for(i = 0; i <10; i++){
    if(instr_get_opcode(instr) == omitted[i]){
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

    if(pos > MAX_CODE_SIZE){
      cinfo->code[MAX_CODE_SIZE - 1] = '\0';
      dr_printf("%s:\n", cinfo->code);
    }

    DR_ASSERT(pos <= MAX_CODE_SIZE);
  }

  cinfo->code_size = pos;

}


char get_size_prefix(uint32_t size){
  
  switch(size){
  case 1: return 'b';
  case 2: return 'w';
  case 4: return 'l';
  case 8: return 'q';
  default: return 'e';
  }


}



void remove_data(char * buffer, unsigned length){
  
  char first[MNEMONIC_SIZE];  
  int i = 0;

  while(i < length && buffer[i] == ' ') i++;

  int start = i;
  while(i < length && buffer[i] != ' '){
    first[i - start] = buffer[i];
    i++;
  }

  first[4] = '\0';


  if(strcmp(first,"data") == 0 || strcmp(first,"lock") == 0){

    while(i < length && buffer[i] == ' ') i++;

    start = i;
    while(i < length){
      buffer[i - start] = buffer[i];
      i++;
    }
    
    int j = i - start;
    for(; j < length; j++){
      buffer[j] = ' ';
    }

  }

}

void switch_operands(ins_t * ins, int in1, int in2){


  char temp[MNEMONIC_SIZE];
  
  int len1 = strlen(ins->operands[in1].name);
  int len2 = strlen(ins->operands[in2].name);

  strncpy(temp, ins->operands[in1].name, len1);
  temp[len1] = '\0';
  strncpy(ins->operands[in1].name, ins->operands[in2].name, len2);
  ins->operands[in1].name[len2] = '\0';
  strncpy(ins->operands[in2].name, temp, len1);
  ins->operands[in2].name[len1] = '\0';
}


int check_for_opcode(int * opcodes, int num_opcodes, instr_t * instr){

  int i = 0;
  bool found = false;
  int opcode_num = instr_get_opcode(instr);

  for(; i < num_opcodes; i++){
    if(opcode_num == opcodes[i]){
      found = true;
      break;
    }
  }

  if(found) return i;
  else return -1;

}



void print_opnds(instr_t * instr){

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


void change_operands(ins_t * ins, instr_t * instr){

  //ok, change all operands with r<>l to r<>b
  int i;
  int j;

 
  for(i = 0; i < ins->num_ops; i++){
    if(ins->operands[i].type == REG_TYPE){
      
      int len = strlen(ins->operands[i].name);
      char * name = ins->operands[i].name;
      if(name[1] == 'r' && name[len - 1] == 'l'){
	bool found = true;
	for(j = 2; j < len - 1; j++){
	  if(name[j] < '0' || name[j] > '9'){
	    found = false;
	    break;
	  }
	}

	if(found){
	  name[len - 1] = 'b';
	}
	
      }
    }
  }


  int opcode = instr_get_opcode(instr);

  switch(opcode){
  
  case OP_pshufd:
  case OP_vcvtsi2sd:
  case OP_vmulsd:
  case OP_vmulpd:
  case OP_vsubsd:
  case OP_vaddsd:
  case OP_vdivsd:
  case OP_vfmadd231sd:
  case OP_vfnmadd231sd:
  case OP_vfmadd132sd:
  case OP_vfnmadd132sd:
  case OP_vfmadd213sd:
  case OP_vfnmadd213sd:
  case OP_vextracti128:
  case OP_vfmsub132sd:
  case OP_imul:{
    if(ins->num_ops == 3)
      switch_operands(ins,0,1);
    break;
  }

  case OP_cmpxchg:
  case OP_vpabsd:
  case OP_vpabsw:
  case OP_vpabsb:{
    ins->num_ops = 2;
    break;
  }

  case OP_vpinsrd:
  case OP_vinserti128:
  case OP_vinsertf128:{
    switch_operands(ins,0,2);

    if(opcode == OP_vinserti128 || opcode == OP_vinsertf128){
      if(ins->operands[1].name[1] == 'y'){ //cannot be a ymm
	ins->operands[1].name[1] = 'x';
      }
    }

    break;
  }

  case OP_vcvtdq2pd:{ //dr bug
    if(ins->operands[0].type == REG_TYPE){
      ins->operands[0].name[1] = 'x';
    }
    break;
  }
  
  case OP_cmp:
  case OP_test:
  case OP_ptest:
  case OP_vucomiss:
  case OP_vucomisd:
  case OP_vcomiss:
  case OP_vcomisd:
  case OP_vptest:
  case OP_vtestps: 
  case OP_vtestpd: 
  case OP_bound:
  case OP_bt:
  case OP_ucomiss: 
  case OP_ucomisd:
  case OP_comiss: 
  case OP_comisd: 
  case OP_invept: 
  case OP_invvpid: 
  case OP_invpcid:{
     DR_ASSERT(ins->num_ops >= 2);
     switch_operands(ins, 0, 1);
     break;
  }

  case OP_mul:
  case OP_div:
  case OP_idiv:{
    ins->num_ops = 1;
    break;
  }

  case OP_vpslld:
  case OP_vpslldq:
  case OP_vpsllq:
  case OP_vpsllw:
  case OP_vpsrad:
  case OP_vpsraw:
  case OP_vpsrld:
  case OP_vpsrldq:
  case OP_vpsrlq:
  case OP_vpsrlw:

  case OP_vpcmpistri:
  case OP_pcmpistri:{
    break;
  }

  case OP_xadd:{
    //if(ins->operands[0].type == MEM_TYPE){ //ex - xaddl (%rbx), %eax
    switch_operands(ins, 0, 1);
    //}
    break;
  }

  case OP_vpmovzxbw:
  case OP_vpmovzxwd:
  case OP_vpmovzxdq:
  case OP_vpmovsxbw:
  case OP_vpmovsxwd:
  case OP_vpmovsxdq:{
    if(ins->num_ops == 2){ //TODO - check actual DR operands and then emit ymm or xmm
      if(ins->operands[0].type == REG_TYPE){
	if(ins->operands[0].name[1] == 'y')
	  ins->operands[0].name[1] = 'x';
      }
      if(ins->operands[1].type == REG_TYPE){
	if(ins->operands[1].name[1] == 'y')
	  ins->operands[1].name[1] = 'x';	
      }
    }
    break;
  }
    

  default:{
    //can change later - we need to change the operand order - is this correct??
    if(ins->num_ops >= 3){
      int srcs = ins->num_ops - 1;
      int front = 0;
      int back = srcs - 1;

      while(front < back){
	switch_operands(ins, front, back);
	front++;
	back--;
      }
    }


  }
    

  }



}




void change_opcodes(ins_t * ins,  instr_t * instr){
  
#undef num_opcodes 
#define num_opcodes 2

  int opcodes[num_opcodes] = {OP_cwde, OP_cdq};
  char alt_names[num_opcodes][MNEMONIC_SIZE] = {"cwtl", "cltd"};

  int opcode = instr_get_opcode(instr);
  
  switch(opcode){
  case OP_cwde:
  case OP_cdq:{
      
    int index = check_for_opcode(opcodes, num_opcodes,  instr);
    strncpy(ins->name, alt_names[index], strlen(alt_names[index]) + 1);
    break;

  }

  case OP_vmovd:{
    DR_ASSERT(instr_num_srcs(instr) == 1 && instr_num_dsts(instr) == 1);
    opnd_t src = instr_get_src(instr,0);
    opnd_t dst = instr_get_dst(instr,0);
    int src_size = opnd_size_in_bytes(opnd_get_size(src));
    int dst_size = opnd_size_in_bytes(opnd_get_size(dst));
    int min_size = src_size > dst_size ? dst_size : src_size;
    
    if(min_size != 4){ //something wrong
      char suffix = get_size_prefix(min_size);
      if(suffix == 'e') return;
      else ins->name[4] = suffix;      
    }  
    break;
  }

  case OP_vcvtsi2sd:
  case OP_vcvtsi2ss:
  case OP_cvtsi2sd:
  case OP_cvtsi2ss:{
   
    int size = opnd_size_in_bytes(opnd_get_size(instr_get_src(instr,1)));
    char suffix = get_size_prefix(size);
    if(suffix == 'e') return;
    else ins->name[9] = suffix;
    break;

  }
  }

}


void correct_movs(ins_t * ins, instr_t * instr){

#undef num_opcodes
#define num_opcodes 3

  int opcodes[num_opcodes] = {OP_movsx, OP_movzx, OP_movsxd};

  int index = check_for_opcode(opcodes, num_opcodes,  instr);
  if(index == -1) return;

  opnd_t opnd = instr_get_dst(instr, 0);
  opnd_size_t size = opnd_get_size(opnd);
  uint32_t size_in_bytes = opnd_size_in_bytes(size);

  char dst_prefix = get_size_prefix(size_in_bytes);
  
  opnd = instr_get_src(instr, 0);
  size = opnd_get_size(opnd);
  size_in_bytes = opnd_size_in_bytes(size);

  char src_prefix = get_size_prefix(size_in_bytes);
  
  if(src_prefix == 'e' || dst_prefix == 'e') return;

  ins->name[4] = src_prefix;
  ins->name[5] = dst_prefix;
  ins->name[6] = '\0';


}


bool add_operand_size(ins_t * ins, instr_t * instr){
 
  //do we need to add the prefix?
  int opcode = instr_get_opcode(instr);

  if(opcode >= 1105){ //opcode count for the generated change_opcode array
    return false;
  }

  if(!change_opcode[opcode]){
    return false;
  }
  
  //get the maximum write size
  int num_dsts = instr_num_dsts(instr);
  int i = 0;
  int j = 0;
  int maxsize = 0;

  for(i = 0; i < num_dsts; i++){
    opnd_t opnd = instr_get_dst(instr, i);
    opnd_size_t size = opnd_get_size(opnd);
    uint32_t size_in_bytes = opnd_size_in_bytes(size);
    maxsize = maxsize < size_in_bytes ? size_in_bytes : maxsize;
  }

  bool immed_op = false;
  int num_srcs = instr_num_srcs(instr);
  for(i = 0; i < num_srcs; i++){
    opnd_t opnd = instr_get_src(instr,i);
    if(opnd_is_immed(opnd)){
      immed_op = true;
      continue;
    } 
    opnd_size_t size = opnd_get_size(opnd);
    uint32_t size_in_bytes = opnd_size_in_bytes(size);
    if(num_dsts == 0)   //if destinations are zero get it from the srcs
      maxsize = maxsize < size_in_bytes ? size_in_bytes : maxsize;
  }
  
  //some special cases
  //get the intel opcode
  char intel[BUFFER_SIZE];
  char name[MNEMONIC_SIZE];
  disassemble_set_syntax(DR_DISASM_INTEL);
  void * drcontext = dr_get_current_drcontext();
  int length = instr_disassemble_to_buffer(drcontext, instr, intel, BUFFER_SIZE);
  remove_data(intel, length);
  i = 0;
  while(intel[i] == ' ') i++;
  int start = i;
  while(intel[i] != ' '){
    name[i - start] = intel[i];
    i++;
  }
  name[i - start] = '\0';
  disassemble_set_syntax(DR_DISASM_ATT);
  

  //are they the same? if not return
  if(strcmp(ins->name, name) != 0){
    return false;
  }


  //rep instructions are correct
  if(strstr(ins->name,"rep")){
    return false;
  }
  
  //ok, add the prefix
  char prefix = get_size_prefix(maxsize);
  
  if(prefix == 'e'){
    return false;
  }


  //now we need to insert this letter to the end of the opcode
  int opcode_sz = strlen(ins->name);
  ins->name[opcode_sz] = prefix;
  ins->name[opcode_sz + 1] = '\0';


}



void textual_embedding_with_size(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  int pos = 0;
 
  char disasm[BUFFER_SIZE];

  ins_t ins;
  
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    if(filter_instr(instr)) continue;

    int length = instr_disassemble_to_buffer(drcontext, instr, disasm, BUFFER_SIZE);


    //dr_printf("raw:%s:\n",disasm);
    remove_data(disasm, length);
    if(!parse_instr_att(disasm, length, &ins)){
      dr_printf("%s:\n",disasm);
      continue;
    }
    //dr_printf("before:");
    //print_instr(&ins);

    add_operand_size(&ins, instr);
    correct_movs(&ins, instr);
    change_opcodes(&ins, instr);
    change_operands(&ins, instr);
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
      dr_printf("%s:\n",cinfo->code);
    }

    DR_ASSERT(pos <= MAX_CODE_SIZE); 
    

  }


  cinfo->code_size = pos;
 

}

