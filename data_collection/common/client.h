
//dump modes are defined here - bit mask
#define DUMP_INTEL  0x1 //dumps intel
#define DUMP_ATT    0x2 //dumps att
#define DUMP_TOKEN  0x4 //dumps token output

//whether we generate SQL statements for updating or inserting code
#define INSERT_CODE 1
#define UPDATE_CODE 2

//client arguments
typedef struct {
  char compiler[MAX_STRING_SIZE];
  char flags[MAX_STRING_SIZE];
  char data_folder[MAX_STRING_SIZE];
  uint32_t mode;
  uint32_t code_format;
  uint32_t dump_mode;
  uint32_t insert_or_update;
} client_arg_t;
