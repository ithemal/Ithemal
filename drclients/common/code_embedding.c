#include "code_embedding.h"
#include "change_opcode.h"

#define DELIMITER -1

//operand types
#define MEM_TYPE 1
#define IMM_TYPE 2
#define REG_TYPE 3

#define MNEMONIC_SIZE 32
#define NUM_OPS 5

typedef struct {
  uint32_t type;
  char name[MNEMONIC_SIZE];
} op_t;

typedef struct {
  op_t operands[NUM_OPS];
  char name[MNEMONIC_SIZE];
  int num_ops;
} ins_t;



//assumes a cleaned up instruction with only valid mnemonics
void parse_instr(const char * buffer, int length, ins_t * instr){
  
  int i = 0;

  //skip white space
  while(i < length && buffer[i] == ' ') i++;

  uint32_t start_opcode = i;
  while(buffer[i] != ' '){
    instr->name[i - start_opcode] = buffer[i];
    i++;
  }
  instr->name[i - start_opcode] = '\0';

  if(strcmp(instr->name,"rep") == 0){  //handle repetition opcodes correctly
    instr->num_ops = 0;
    while(i < length){
      instr->name[i - start_opcode] = buffer[i];
      i++;
    }
    instr->name[i - start_opcode] = '\0';
    return;
  }


  //skip white space
  while(i < length && buffer[i] == ' ') i++;

  uint32_t start_operand = i;
  uint32_t op_num = 0;
  instr->num_ops = 0;

  while(i < length){
    
    if(buffer[i] == '$'){
      instr->operands[op_num].type = IMM_TYPE;
      while(i < length && buffer[i] != ','){
	instr->operands[op_num].name[i - start_operand] = buffer[i];
	i++;
      }
    }
    else{ //can be memory or reg
      if(buffer[i] != '%'){ //then it is memory for sure
	instr->operands[op_num].type = MEM_TYPE;
	bool found_open = false;
	while(buffer[i] != ')'){
	  DR_ASSERT(i < length);
	  if(buffer[i] == '(') found_open = true;
	  instr->operands[op_num].name[i - start_operand] = buffer[i];
	  i++;
	}
	DR_ASSERT(found_open);
	instr->operands[op_num].name[i - start_operand] = buffer[i];
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
	      DR_ASSERT(i < length);
	      if(buffer[i] == '(') found_open = true;
	      instr->operands[op_num].name[i - start_operand] = buffer[i];
	      i++;
	    }
	    DR_ASSERT(found_open);
	    instr->operands[op_num].name[i - start_operand] = buffer[i];
	    while(i < length && buffer[i] != ',') i++;
	  }
	  else if(j < length && buffer[j] == ','){ //no base index
	    while(i < length && buffer[i] != ','){
	      instr->operands[op_num].name[i - start_operand] = buffer[i];
	      i++;
	    }
	  }
	}
	else{
	  instr->operands[op_num].type = REG_TYPE;
	  while(i < length && buffer[i] != ','){
	    instr->operands[op_num].name[i - start_operand] = buffer[i];
	    i++;
	  }
	}
      }

    }

   
    instr->operands[op_num].name[i - start_operand] = '\0';
    i++;
    while(i < length && buffer[i] == ' ') i++;
    start_operand = i;
    op_num++;

  }

  instr->num_ops = op_num;
  

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


char get_size_prefix(uint32_t size){
  
  switch(size){
  case 1: return 'b';
  case 2: return 'w';
  case 4: return 'l';
  case 8: return 'q';
  default: return 'e';
  }


}



void correct_movs(char * buffer, unsigned start, unsigned end, instr_t * instr){


  int opcodes[2] = {OP_movsx, OP_movzx};

  int i = 0;
  int opcode_num = instr_get_opcode(instr);

  bool found = false;
  for(i = 0; i < 19; i++){
    if(opcode_num == opcodes[i]){
      found = true; break;
    }
  }

  if(!found) return;


  opnd_t opnd = instr_get_dst(instr, 0);
  opnd_size_t size = opnd_get_size(opnd);
  uint32_t size_in_bytes = opnd_size_in_bytes(size);

  char dst_prefix = get_size_prefix(size_in_bytes);
  
  opnd = instr_get_src(instr, 0);
  size = opnd_get_size(opnd);
  size_in_bytes = opnd_size_in_bytes(size);

  char src_prefix = get_size_prefix(size_in_bytes);
  
  if(src_prefix == 'e' || dst_prefix == 'e') return;


}

void remove_data(char * buffer, unsigned length){
  
  char first[24];  
  int i = 0;

  while(i < length && buffer[i] == ' ') i++;

  int start = i;
  while(i < length && buffer[i] != ' '){
    first[i - start] = buffer[i];
    i++;
  }

  first[4] = '\0';


  if(strcmp(first,"data") == 0){

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


void correct_operand_ordering(char * buffer, unsigned start, unsigned end, instr_t * instr){
  
  int opcodes[19] = {OP_cmp,OP_test,OP_ptest,OP_vucomiss,OP_vucomisd,OP_vcomiss,OP_vcomisd,
                    OP_vptest, OP_vtestps, OP_vtestpd, OP_bound, OP_bt, OP_ucomiss, OP_ucomisd,
                    OP_comiss, OP_comisd, OP_invept, OP_invvpid, OP_invpcid};


  int i = 0;
  int opcode_num = instr_get_opcode(instr);

  bool found = false;
  for(i = 0; i < 19; i++){
    if(opcode_num == opcodes[i]){
      found = true; break;
    }
  }

  if(!found) return;


  char opcode[32]; int oplen = 0;
  char src1[32]; int s1len = 0;
  char src2[32]; int s2len = 0;

  bool start1 = false;
  bool start2 = false;

  for(i = start; i < end - 1; i++){
    
    if(!start1 && !start2){
      if(buffer[i] != ' '){
	opcode[oplen++] = buffer[i];
      }
      else{
	start1 = true;
      }
    }
    

    if(start1 && buffer[i] == ','){
      start1 = false;  start2 = true;
    }
    else if(start1 && buffer[i] != ' '){
      src1[s1len++] = buffer[i];
    }
    else if(start2 && buffer[i] != ' '){
      src2[s2len++] = buffer[i];
    }

  }

  int where = start;

  for(i = 0; i <oplen; i++){
    buffer[where++] = opcode[i];
  }
  buffer[where++] = ' ';
  for(i = 0; i <s2len; i++){
    buffer[where++] = src2[i];
  }
  buffer[where++] = ',';
  buffer[where++] = ' ';
  for(i = 0; i <s1len; i++){
    buffer[where++] = src1[i];
  }
  

  for(i = where; i < end - 1; i++){
    buffer[i] = ' ';
  }
  buffer[end - 1] = '\n';


}


bool add_operand_size(char * buffer, unsigned start, unsigned end, instr_t * instr){
 
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

  //if destinations are zero get it from the srcs
  if(num_dsts == 0){
    int num_srcs = instr_num_srcs(instr);
    for(i = 0; i < num_srcs; i++){
      opnd_t opnd = instr_get_src(instr,i);
      if(opnd_is_immed(opnd)) continue;
      opnd_size_t size = opnd_get_size(opnd);
      uint32_t size_in_bytes = opnd_size_in_bytes(size);
      maxsize = maxsize < size_in_bytes ? size_in_bytes : maxsize;
    }
  }


  char prefix = get_size_prefix(maxsize);
  
  if(prefix == 'e'){
    return false;
  }

#define TEMP_SIZE 1024
  char temp_buff[TEMP_SIZE];

  //now we need to insert the opcode
  //find the first space
  for(i = start; i < end; i++){
    if(buffer[i] == ' '){
      if(buffer[i + 1] == ' '){//if the next character is also a space no prob!
	buffer[i] = prefix;
	return false;
      }
      else{
	if((end - start) > TEMP_SIZE){
	  return false;
	}
	if(end + 1 >= MAX_CODE_SIZE){//we cannot insert
	  return false;
	}
	else{
	  for(j = i; j < end; j++){
	    temp_buff[j - i] = buffer[j]; 
	  }
	  buffer[i] = prefix;
	  for(j = i + 1; j <= end; j++){
	    buffer[j] = temp_buff[j - i - 1];
	  }
	}
      }
      return true;
    }
  }

  return false;


 

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


void print_instr(ins_t * ins){

  dr_printf("%s\n",ins->name);
  int i = 0;
  for(; i < ins->num_ops; i++){
    dr_printf("%s\n",ins->operands[i].name);
  }

}


void textual_embedding_with_size(void * drcontext, code_info_t * cinfo, instrlist_t * bb){

  instr_t * instr;
  int pos = 0;
  int instr_count = 0;

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    if(filter_instr(instr)) continue;
    instr_count++;
  }

  ins_t * instrs = dr_thread_alloc(drcontext, sizeof(ins_t) * instr_count);

  int i = 0;  
  char disasm[1024];
  
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    if(filter_instr(instr)) continue;

    ins_t * ins = &instrs[i];
    int length = instr_disassemble_to_buffer(drcontext, instr, disasm, 1024);

    remove_data(disasm, length);

    dr_printf("%s\n",disasm);

    parse_instr(disasm, length, ins);
    
    print_instr(ins);

    //dr_printf("%d\n",ins->num_ops);

    int j = 0;
    int w = 0;

    w = sprintf(cinfo->code + pos, "%s ", ins->name);
    DR_ASSERT(w > 0);
    pos += w;
    
    for(j = 0; j < ins->num_ops; j++){
      w = sprintf(cinfo->code + pos, "%s", ins->operands[j].name);
      DR_ASSERT(w > 0);
      pos += w;
      if(j != ins->num_ops - 1){
	w = sprintf(cinfo->code + pos,  ", ");
	DR_ASSERT(w > 0);
	pos += w;
      }
    }
    w = sprintf(cinfo->code + pos,  "\n"); 
    DR_ASSERT(w > 0);
    pos += w;

    i++;
    DR_ASSERT(pos <= MAX_CODE_SIZE); 
  }

  cinfo->code_size = pos;
  dr_thread_free(drcontext, instrs, sizeof(ins_t) * instr_count);


}


/*

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){

    if(filter_instr(instr)) continue;
    
    pos += instr_disassemble_to_buffer(drcontext,instr,cinfo->code + pos, MAX_CODE_SIZE - 1 -  pos);
    cinfo->code[pos++] = '\n';
    remove_data(cinfo->code, prev_pos, pos, instr);
    if(add_operand_size(cinfo->code, prev_pos, pos, instr)){
      pos++;
    }
    correct_operand_ordering(cinfo->code, prev_pos, pos, instr);
    prev_pos = pos;
    DR_ASSERT(pos <= MAX_CODE_SIZE);
  }

  cinfo->code_size = pos;

 */
