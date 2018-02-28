#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>

#include "mmap.h"



void * create_mmap(const char * filename, uint32_t size){

  int i;
  int fd;
  int result;
  void *map;  /* mmapped array of int's */


  /* Open a file for writing.
   *  - Creating the file if it doesn't exist.
   *  - Truncating it to 0 size if it already exists. (not really needed)
   *
   * Note: "O_WRONLY" mode is not sufficient when mmaping.
   */
  fd = open(filename, O_RDWR | O_CREAT, (mode_t)0600);
  if (fd == -1) {
    perror("Error opening file for writing");
    exit(EXIT_FAILURE);
  }

  /* Stretch the file size to the size of the (mmapped) array of ints
   */
  result = lseek(fd, size-1, SEEK_SET);
  if (result == -1) {
    close(fd);
    perror("Error calling lseek() to 'stretch' the file");
    exit(EXIT_FAILURE);
  }
    
  /* Something needs to be written at the end of the file to
   * have the file actually have the new size.
   * Just writing an empty string at the current file position will do.
   *
   * Note:
   *  - The current position in the file is at the end of the stretched 
   *    file due to the call to lseek().
   *  - An empty string is actually a single '\0' character, so a zero-byte
   *    will be written at the last byte of the file.
   */
  result = write(fd, "", 1);
  if (result != 1) {
    close(fd);
    perror("Error writing last byte of the file");
    exit(EXIT_FAILURE);
  }

  /* Now the file is ready to be mmapped.
   */
  map = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    close(fd);
    perror("Error mmapping the file");
    exit(EXIT_FAILURE);
  }

  return map;

}

typedef struct{
  void * module_start;
  char module[MAX_MODULE_SIZE];
} modules_t;

volatile thread_files_t * files;
volatile void * per_thread_file[MAX_THREADS];
uint32_t num_files;

void get_new_file(){
  if(files->control == DUMP_ONE){
    assert(num_files == files->num_modules - 1); //we just got a new one
    printf("new file - %s\n",files->modules[num_files]);
    per_thread_file[num_files] = create_mmap(files->modules[num_files],TOTAL_SIZE);
    assert(per_thread_file[num_files]);
    num_files++;
    while(!__sync_bool_compare_and_swap(&files->control,DUMP_ONE,IDLE)){}
  }
}

void check_for_exit(){
  if(files->control == EXIT){
    while(!__sync_bool_compare_and_swap(&files->control,EXIT,IDLE)){}
    exit(0);
  }
}


void check_for_new_code(volatile void * file){

  volatile code_info_t * cinfo = (code_info_t *)(file + START_CODE_DATA);
  if(cinfo->control == DUMP_ONE){
    printf("%s,%llu,%d,%s\n",cinfo->module,cinfo->module_start,cinfo->rel_addr,cinfo->code);
    while(!__sync_bool_compare_and_swap(&cinfo->control,DUMP_ONE,IDLE)){}
  }
}

void check_for_new_times(){

}


int main(int argc, char *argv[])
{

  files = create_mmap(FILENAMES_FILE,sizeof(thread_files_t));
  assert(files);
  num_files = 0;

  int i = 0;

  while(1){
    get_new_file();
    for(i = 0; i < num_files; i++){
      check_for_new_code(per_thread_file[i]);
    }
    check_for_exit();

  }

  return 0;
}
