#ifndef CMODEL_MODE
#define CMODEL_MODE

/*
 data collection modes
 */
#define RAW_SQL 0
#define SNOOP   1
#define SQLITE  2


/*
  control values - this determines who gets control of the data and what to do with it
 */

#define IDLE       0
#define DR_CONTROL 1
#define DUMP_ONE   2
#define DUMP_ALL   3
#define EXIT       4



#endif
