#include "dynamic_logic.h"
#include "hashtable.h"
#include <stdint.h>
#include <string.h>


//#define DEBUG

int span_for_instr(instr_t * instr, instrlist_t * bb, uint32_t * span,  hashtable_t * map);

int num_instructions(instrlist_t * bb){
  
  uint32_t i = 0;
  instr_t * instr;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    i++;
  } 
  return i;

}


#define MAX_INSTRUCTIONS 100
//partial register dependencies not counted - can deal with reg_overlap later
//each memory reference is assumed to be disjoint
int span_for_instr(instr_t * instr, instrlist_t * bb, uint32_t * span,  hashtable_t * map){
  
  void * drcontext = dr_get_current_drcontext();
  int index = hashtable_lookup(map,instr);
  if(span[index] != -1){
    return span[index];
  }


  int i = 0;
  int j = 0;
  int span_max = 0;

  for(i = 0; i < instr_num_dsts(instr); i++){
    
    opnd_t dst_i = instr_get_dst(instr,i);

    //either it becomes dead or it is used
    instr_t * src_instr;
    bool dead = false;

    for(src_instr = instr_get_next(instr); src_instr != instrlist_last(bb); src_instr = instr_get_next(src_instr)){
      for(j = 0; j < instr_num_srcs(src_instr); j++){
	opnd_t src_j = instr_get_src(src_instr,j);
	if(opnd_same(dst_i,src_j)){

#ifdef DEBUG
	  instr_disassemble(drcontext,instr,STDOUT);
	  dr_printf("->");
	  instr_disassemble(drcontext,src_instr,STDOUT);
	  dr_printf("->");
	  opnd_disassemble(drcontext,dst_i,STDOUT);
	  dr_printf("\n");
#endif

	  int span_i = span_for_instr(src_instr,bb,span,map);
	  if(span_i + 1 > span_max) span_max = span_i + 1;
	}
      }
      for(j = 0; j < instr_num_dsts(src_instr); j++){
	opnd_t dst_j = instr_get_dst(src_instr,j);
	if(opnd_same(dst_i,dst_j)){
	  dead = true;
	  break;
	}
      }
      if(dead) break;
    }

  }

  span[index] = span_max;

  return span_max;

} 


int span_bb(instrlist_t * bb){

  hashtable_t map;
  instr_t * instr = instrlist_first(bb);
  uint32_t i = 1;
  uint32_t span[MAX_INSTRUCTIONS];
  

  memset(span,-1,sizeof(uint32_t) * MAX_INSTRUCTIONS);
  hashtable_init(&map,6,HASH_INTPTR,false);
  

  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    hashtable_add(&map,(void *)instr,(void *)i);
    i++; 
  } 

  int span_max = 0;
  for(instr = instrlist_first(bb); instr != instrlist_last(bb); instr = instr_get_next(instr)){
    int span_i = span_for_instr(instr,bb,span,&map);
    span_max = span_i > span_max ? span_i : span_max;
  } 

  return span_max;

}


