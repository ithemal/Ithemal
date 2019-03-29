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

#define INIT_VALUE 0x23240

#define CTX_SWTCH_FD 100

#define ITERS 16

#define OFFSET_TO_COUNTERS 32

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

// addresses of the `mov rcx, *` instructions
// before we run rdpmc. we modify these instructions
// once we've setup the programmable pmcs
extern char l1_read_misses_a[];
extern char l1_read_misses_b[];
extern char l1_write_misses_a[];
extern char l1_write_misses_b[];
extern char icache_misses_a[];
extern char icache_misses_b[];

// boundary of the actual test harness
extern char code_begin[];
extern char code_end[];

struct perf_event_attr l1_read_attr = {
  .type = PERF_TYPE_HW_CACHE,
  .config =
    ((PERF_COUNT_HW_CACHE_L1D) |
     (PERF_COUNT_HW_CACHE_OP_READ << 8) |
     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
  .size = PERF_ATTR_SIZE_VER0,
  .sample_type = PERF_SAMPLE_READ,
  .exclude_kernel = 1
};
struct perf_event_attr l1_write_attr = {
  .type = PERF_TYPE_HW_CACHE,
  .config =
    ((PERF_COUNT_HW_CACHE_L1D) |
     (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
  .size = PERF_ATTR_SIZE_VER0,
  .sample_type = PERF_SAMPLE_READ,
  .exclude_kernel = 1
};
struct perf_event_attr icache_attr = {
  .type = PERF_TYPE_HW_CACHE,
  .config =
    ((PERF_COUNT_HW_CACHE_L1I) |
     (PERF_COUNT_HW_CACHE_OP_READ << 8) |
     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
  .size = PERF_ATTR_SIZE_VER0,
  .sample_type = PERF_SAMPLE_READ,
  .exclude_kernel = 1
};
struct perf_event_attr ctx_swtch_attr = {
  .type = PERF_TYPE_SOFTWARE,
  .config = PERF_COUNT_SW_CONTEXT_SWITCHES,
  .exclude_idle = 0
};

char x;

// round addr to beginning of the page it's in
char *round_to_page_start(char *addr) {
  return (char *)(((uint64_t)addr >> 12) << 12);
}

char *round_to_next_page(char *addr) {
  return (char *)((((uint64_t)addr + 4096) >> 12) << 12);
}

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

// emit inst to do `mov rcx, val'
void emit_mov_rcx(char *inst, int val) {
  if (val < 0)
    return;

  inst[0] = 0xb9;
  inst[1] = val;
  inst[2] = 0;
  inst[3] = 0;
  inst[4] = 0;
}

int is_event_supported(struct perf_event_attr *attr) {
  struct rdpmc_ctx ctx;
  int ok = !rdpmc_open_attr(attr, &ctx, 0);
  rdpmc_close(&ctx);
  return ok;
}

// handling up to these many faults from child process
#define MAX_FAULTS 10240

struct pmc_counters {
  uint64_t clock;
  uint64_t core_cyc;
  uint64_t l1_read_misses;
  uint64_t l1_write_misses;
  uint64_t icache_misses;
  uint64_t context_switches;
};

// return counters
struct pmc_counters *measure(
    int *l1_read_supported,
    int *l1_write_supported,
    int *icache_supported) {
  // allocate 3 pages, the first one for testing
  // the rest for writing down result
  int shm_fd = create_shm_fd("shm-path", 4096 * 3);

  int fds[2];
  pipe(fds);

  char *aux_mem = mmap(NULL, 4096 * 2, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 4096);
  perror("mmap aux mem");
    bzero(aux_mem, 4096);

  pid_t pid = fork();
  if (pid) { // parent process
    close(fds[0]);

    char *child_mem = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
    struct pmc_counters *counters = (struct pmc_counters *)(aux_mem + OFFSET_TO_COUNTERS);

    attach_to_child(pid, fds[1]);

    // find out which PMCs are supported
    *l1_read_supported = is_event_supported(&l1_read_attr);
    *l1_write_supported = is_event_supported(&l1_write_attr);
    *icache_supported = is_event_supported(&icache_attr);

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
        fault_addr = (void *)INIT_VALUE;

      // before we restart, we initialize child_mem for consistency
      int i;
      for (i = 0; i < 512; i++) {
        ((unsigned long *)child_mem)[i] = INIT_VALUE;
      }
      restart_child(pid, restart_addr, fault_addr, shm_fd);
    }

    // if we've reached this point, the child is hopeless
    kill(pid, SIGKILL);
    return NULL;

  } else { // child process
    // setup PMCs
    struct rdpmc_ctx ctx;
    rdpmc_open_attr(&l1_read_attr, &ctx, 0);
    int l1_read_misses_idx = ctx.buf->index - 1;
    LOG("L1 READ IDX = %d\n", ctx.buf->index);

    rdpmc_open_attr(&l1_write_attr, &ctx, 0);
    int l1_write_misses_idx = ctx.buf->index - 1;
    LOG("L1 WRITE IDX = %d\n", ctx.buf->index);

    rdpmc_open_attr(&icache_attr, &ctx, 0);
    int icache_misses_idx = ctx.buf->index - 1;
    LOG("ICACHE IDX = %d\n", ctx.buf->index);

    int ret = rdpmc_open_attr(&ctx_swtch_attr, &ctx, 0);
    if (ret != 0) {
      LOG("unable to count context switches\n");
      abort();
    }
    dup2(ctx.fd, CTX_SWTCH_FD);

    errno = 0;

    // unprotect the test harness 
    // so that we can emit instructions to use
    // the proper pmc index
    char *begin = round_to_page_start(code_begin);
    char *end = round_to_next_page(code_end);
    mprotect(begin, end-begin, PROT_EXEC|PROT_READ|PROT_WRITE);
    perror("mprotect");

    emit_mov_rcx(l1_read_misses_a, l1_read_misses_idx);
    emit_mov_rcx(l1_read_misses_b, l1_read_misses_idx);
    emit_mov_rcx(l1_write_misses_a, l1_write_misses_idx);
    emit_mov_rcx(l1_write_misses_b, l1_write_misses_idx);
    emit_mov_rcx(icache_misses_a, icache_misses_idx);
    emit_mov_rcx(icache_misses_b, icache_misses_idx);

    // re-protect the harness
    mprotect(begin, end-begin, PROT_EXEC);
    perror("mprotect");

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
  int l1_read_supported, l1_write_supported, icache_supported;
  struct pmc_counters *counters = measure(&l1_read_supported, &l1_write_supported, &icache_supported);


  if (!counters) {
    fprintf(stderr, "failed to run test\n");
    return 1;
  }

  printf("Clock\tCore_cyc\tL1_read_misses\tL1_write_misses\tiCache_misses\tContext_switches\n");
  int i;
  for (i = 0; i < ITERS; i++) {
    printf("%ld\t%ld\t%ld\t%ld\t%ld\t%ld\n",
        counters[i].clock, 
        counters[i].core_cyc,
        l1_read_supported ? counters[i].l1_read_misses : -1,
        l1_write_supported ? counters[i].l1_write_misses : -1,
        icache_supported ? counters[i].icache_misses : -1,
        counters[i].context_switches);
  }

}
