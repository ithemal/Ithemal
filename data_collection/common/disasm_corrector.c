#include "disasm_corrector.h"
#include "change_opcode.h"
#include "string.h"

/************** Internal Functions *****************/

static void print_instr(ins_t * ins){

  dr_printf("ins: ");
  dr_printf("%s ",ins->name);
  int i = 0;
  for(; i < ins->num_ops; i++){
    dr_printf("%s, ",ins->operands[i].name);
  }
  dr_printf("\n");

}



static void remove_data(char * buffer, unsigned length){

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



static char get_size_prefix(uint32_t size){

  switch(size){
  case 1: return 'b';
  case 2: return 'w';
  case 4: return 'l';
  case 8: return 'q';
  default: return 'e';
  }


}


static void switch_operands(ins_t * ins, int in1, int in2){


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


static int check_for_opcode(int * opcodes, int num_opcodes, instr_t * instr){

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

static void change_operands(ins_t * ins, instr_t * instr){

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




static void change_opcodes(ins_t * ins,  instr_t * instr){

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
  case OP_vcvtsi2ss:{

    DR_ASSERT(instr_num_srcs(instr) == 2);

    int size = opnd_size_in_bytes(opnd_get_size(instr_get_src(instr,1)));
    char suffix = get_size_prefix(size);
    if(suffix == 'e') return;
    else ins->name[9] = suffix;
    break;

  }

  case OP_cvtsi2sd:
  case OP_cvtsi2ss:{

    DR_ASSERT(instr_num_srcs(instr) == 1);

    int size = opnd_size_in_bytes(opnd_get_size(instr_get_src(instr,0)));
    char suffix = get_size_prefix(size);
    if(suffix == 'e') return;
    else ins->name[8] = suffix;
    break;

  }

  }

}


static void correct_movs(ins_t * ins, instr_t * instr){

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


static bool add_operand_size(void * drcontext, ins_t * ins, instr_t * instr){
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




/*********** public functions *********************/

void correct_disasm_att(void *drcontext, ins_t * ins, instr_t * instr) {
    add_operand_size(drcontext, ins, instr);
    correct_movs(ins, instr);
    change_opcodes(ins, instr);
    change_operands(ins, instr);

}

bool parse_instr_att(char * buffer, int length, ins_t * instr){

  //cleans up instruction with only valid mnemonics
  remove_data(buffer, length);

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
