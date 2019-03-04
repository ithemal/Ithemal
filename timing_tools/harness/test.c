#define _GNU_SOURCE

#include <sys/types.h>
#include <sys/user.h>
#include <sys/ptrace.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <sys/syscall.h>

#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "rdpmc.h"
#include <linux/perf_event.h>
#include <linux/ioctl.h>

#define SHM_FD 42

#define ITERS 16

#ifndef NDEBUG
#define LOG(args...) printf(args)
#else
#define LOG(...) while(0)
#define perror(prefix) while(0)
#endif

extern void run_test();
extern void run_test_nop();
// we don't really care about signature of the function here
// just need the address
extern void map_and_restart();

char x;

void attach_to_child(pid_t pid, int wr_fd) {
  ptrace(PTRACE_SEIZE, pid, NULL, NULL);
  write(wr_fd, &x, 1);
  close(wr_fd);
}

int create_shm_fd(char *path, int size) {
  int fd = shm_open(path, O_RDWR|O_CREAT, 0777);
  shm_unlink(path);
  ftruncate(fd, size);
  return fd;
}

int perf_event_open(struct perf_event_attr *hw_event,
                    pid_t pid, int cpu, int group_fd,
                    unsigned long flags)
{
  return syscall(__NR_perf_event_open, hw_event, pid, cpu,
                 group_fd, flags);
}

void restart_child(pid_t pid, void *restart_addr, void *fault_addr, int shm_fd) {
  LOG("RESTARTING AT %p, fault addr = %p\n", restart_addr, fault_addr);
  struct user_regs_struct regs;
  ptrace(PTRACE_GETREGS, pid, NULL, &regs);
  perror("get regs");
  regs.rip = (unsigned long)restart_addr;
  regs.rax = (unsigned long)fault_addr;
  regs.r11 = shm_fd;
  regs.r12 = MAP_SHARED;
  regs.r13 = PROT_READ|PROT_WRITE;
  ptrace(PTRACE_SETREGS, pid, NULL, &regs);
  perror("set regs");
  ptrace(PTRACE_CONT, pid, NULL, NULL);
  perror("cont");
}

// handling up to these many faults from child process
#define MAX_FAULTS 10240

struct pmc_counters {
  unsigned long clock, core_cyc, l1_read_misses;
};

// return counters
struct pmc_counters *measure() {
  // allocate 2 pages, the first one for testing
  // the second one for writing down result
  int shm_fd = create_shm_fd("shm-path", 4096 * 2);

  int fds[2];
  pipe(fds);

  pid_t pid = fork();
  if (pid) { // parent process
    close(fds[0]);

    char *child_mem = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
    char *aux_mem = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 4096);
    perror("mmap aux mem");
    bzero(aux_mem, 4096);
    // the test harness uses the first 8 bytes to count test iterations
    struct pmc_counters *counters = (struct pmc_counters *)(aux_mem + 8);

    attach_to_child(pid, fds[1]);

    // TODO: kill the child
    int i;
    for (i = 0; i < MAX_FAULTS; i++) {
      int stat;
      pid = wait(&stat);
      if (WIFEXITED(stat) || pid == -1) {
        int retcode = WEXITSTATUS(stat);
        if (retcode == 0) {
          return counters;
        }
        return NULL;
      }

      // something wrong must have happened
      // find out what happened
      siginfo_t sinfo;
      ptrace(PTRACE_GETSIGINFO, pid, NULL, &sinfo);
      struct user_regs_struct regs;
      ptrace(PTRACE_GETREGS, pid, NULL, &regs);
      LOG("bad inst is at %p, signal = %d\n", (void *)regs.rip, sinfo.si_signo);
      if (sinfo.si_signo != 11 && sinfo.si_signo != 5)
        abort();

      // find out address causing the segfault
      void *fault_addr = sinfo.si_addr;
      void *restart_addr = &map_and_restart;
      if (sinfo.si_signo == 5)
        fault_addr = (void *)0x2324000;

      // before we restart, we initialize child_mem for consistency
      int i;
      for (i = 0; i < 512; i++) {
        ((unsigned long *)child_mem)[i] = 0x2324000;
      }
      restart_child(pid, restart_addr, fault_addr, shm_fd);
    }

    // if we've reached this point, the child is hopeless
    kill(pid, SIGKILL);
    return NULL;

  } else { // child process
    // setup counter for L1 cache miss
    struct rdpmc_ctx ctx;
    struct perf_event_attr l1_read_attr = {
      .type = PERF_TYPE_HW_CACHE,
      .config =
        ((PERF_COUNT_HW_CACHE_L1D) |
         (PERF_COUNT_HW_CACHE_OP_READ << ITERS) |
         (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
      .size = PERF_ATTR_SIZE_VER0,
      .sample_type = PERF_SAMPLE_READ,
      .exclude_kernel = 1
    };
    rdpmc_open_attr(&l1_read_attr, &ctx, 0);
    LOG("PMC IDX = %d\n", ctx.buf->index);

    // setup counter for core cycle
    rdpmc_open(PERF_COUNT_HW_CPU_CYCLES, &ctx);

    // wait for parent
    close(fds[1]);
    read(fds[0], &x, 1);
    dup2(shm_fd, SHM_FD);

    // pin this process before we run the test
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(0, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
    setpriority(PRIO_PROCESS, 0, 0);

    // this does not return
    run_test();
  }

  // this is unreachable
  return NULL;
}

// TODO: deal with cases where all tests have unacceptable number of l1 misses?
compute_overhead(struct pmc_counters *counters,
                      int n,
                      long *clock_overhead,
                      long *core_cyc_overhead) {
  *clock_overhead = 0;
  *core_cyc_overhead = 0;
  int i;

  // we count a test as acceptable if there is no cache misses
  int num_acceptable_tests = 0;

  for (i = 0; i < n; i++) {
    if (counters[i].l1_read_misses != 0)
      continue;
    num_acceptable_tests += 1;
    *clock_overhead += counters[i].clock;
    *core_cyc_overhead += counters[i].core_cyc;
  }

  *clock_overhead /= num_acceptable_tests;
  *core_cyc_overhead /= num_acceptable_tests;
}

int main() {
  // `measure` writes the result here
  struct pmc_counters *counters = measure();

  if (!counters) {
    fprintf(stderr, "failed to run test\n");
    return 1;
  }

  printf("Clock\tCore_cyc\tL1_read_misses\n");
  int i;
  for (i = 0; i < ITERS; i++) {
    printf("%ld\t%ld\t%ld\n",
        counters[i].clock, 
        counters[i].core_cyc,
        counters[i].l1_read_misses);
  }

}
