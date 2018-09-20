#include "timing_logic.h"

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
      uint64_t cum_time = (uint64_t)av * (uint64_t)count + (uint64_t)time;
      uint64_t average_time = cum_time / ((uint64_t)count + 1);
      bb->times[i].average = average_time;
      if(bb->times[i].average == 0){
	dr_fprintf(STDERR,"time, av, count, cum_time - %d, %d, %d, %llu\n",time, av, bb->times[i].count, cum_time);
      }
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





