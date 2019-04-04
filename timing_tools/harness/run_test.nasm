%define page_size 4096

%define sys_munmap  11
%define sys_mmap    9
%define sys_exit    60
%define sys_read    0

%define iterations 16

%define shm_fd 42

%define ctx_swtch_fd 100

%define PROT_READ   0x1
%define PROT_WRITE  0x2

%define MAP_SHARED  0x01

%define tsc_offset      0
%define core_cyc_offset 8
%define l1_read_misses_offset 16
%define l1_write_misses_offset 24
%define icache_misses_offset 32
%define ctx_swtch_offset 40

; 8 bytes * 6 counters
; (tsc, core_cyc, i1 read, i1 write, icache, ctx swtch)
%define counter_set_size 48

%define aux_mem   0x0000700000000000
%define counter_array     0x0000700000000020
%define highest_addr      0x0000700000001000
%define iterator          aux_mem
%define tmp               0x0000700000000008

%ifndef NDEBUG 
%define debug 1
%else
%define debug 0
%endif

%define init_value 0x2324000

%define counter_core_cyc 0x40000001

%define reps 100

%macro setup_memory 0
  ; round code_begin/r13 down to page boundary
  mov r13, code_begin
  round_to_page_boundary(r13)
  
  ; unmap everything up to code_begin/r13
  mov rax, sys_munmap
  mov rdi, 0x0  ; addr
  mov rsi, r13  ; len
  syscall

  ; round code_end/r14 up to page boundary
  mov r14, code_end
  add r14, page_size
  add r14, page_size
  round_to_page_boundary(r14)

  ; unmap everything starting from code_end/r14
  mov rdi, r14  ; addr
  mov rsi, highest_addr
  sub rsi, r14  ; len
  mov rax, sys_munmap
  syscall

  ;; FIXME: only map aux mem *after* we've mapped pages for the test code
  ; this way we can guarantee that the test code *does not* modify aux mem
  ; map a page to keep the counters
  mov rdi, aux_mem ; address
  mov rsi, 8092               ; length
  mov rdx, PROT_READ          ; prot
  or  rdx, PROT_WRITE         ; --
  mov r10, MAP_SHARED         ; flag
  mov r8,  42                 ; fd
  mov r9,  4096               ; offset
  mov rax, sys_mmap
  syscall
%endmacro

%macro round_to_page_boundary 1
  ; 2^12 = 4096
  shr %1, 12
  shl %1, 12
%endmacro

global run_test
global map_and_restart
%macro test_impl 0
  %rep reps
  %include "bb.nasm"
  %endrep
%endmacro

global l1_read_misses_a
global l1_read_misses_b
global l1_write_misses_a
global l1_write_misses_b
global icache_misses_a
global icache_misses_b

global code_begin
global code_end

SECTION .text align = 4096

%macro initialize 0
  mov rax, init_value
  round_to_page_boundary(rax)

  ; point rbp, rsp to middle of the page
  mov rbp, rax
  add rbp, 2048
  mov rsp, rbp
  
  mov rax, init_value
  mov rbx, init_value 
  mov rcx, init_value
  mov rdx, init_value
  mov rsi, init_value
  mov rdi, init_value
  mov r8,  init_value
  mov r9,  init_value
  mov r10, init_value
  mov r11, init_value
  mov r12, init_value
  mov r13, init_value
  mov r14, init_value
  mov r15, init_value

%endmacro
;
%macro combine_rax_rdx 1
  mov %1, rdx
  shl %1, 32
  or  %1, rax
%endmacro

%macro read_time_stamp 1
  ; read the time stamp into %1
  rdtsc
  combine_rax_rdx(%1)
%endmacro

%macro read_perf_counter 2
  mov rcx, %1
  rdpmc

  combine_rax_rdx(%2)
%endmacro

%macro read 3
; macro to do a sys_read
; `read fd, buf, bytes`
  mov  rax, sys_read
  mov  rdi, %1 ; fd
  mov  rsi, %2 ; buf
  mov  rdx, %3 ; num bytes
  syscall
%endmacro

; marker of the begin of test code,
; which we need to make sure we don't
; acccidentally unmap itself
code_begin: 

map_and_restart:
  ; mmap the page containing rax
  round_to_page_boundary(rax)
  mov rdi, rax         ; address
  mov rsi, 4096        ; length
  mov rdx, r13         ; prot
  mov r10, r12         ; flag
  mov r8,  shm_fd      ; fd
  xor r9,  r9          ; offset
  mov rax, sys_mmap
  syscall

  ; now go back to the test code
  jmp test_begin

run_test:
  setup_memory

  ; ready to go. 
  ; get the parent process to trigger
  ; a map containing init_value
  int 3

test_begin:
  initialize

  ;; do mappings
  test_impl

%if debug
  mov       rax, 1                  ; system call for write
  mov       rdi, 1                  ; file handle 1 is stdout
  mov       rsi, msg_mapping_done            ; address of string to output
  mov       rdx, 17                 ; number of bytes
  syscall                           ; invoke operating system to do the write
%endif


  ;;;;;;;;;;; TEST BEGIN ;;;;;;;;;;;

  mov rbx, iterator
  mov qword [rbx], 0

test_loop:

  mov rax, counter_array 

  ; rbx = i
  mov rbx, iterator
  mov rbx, [rbx]

  ; r15 = counter + i*(size of one set of counters)
  mov r15, rbx
  imul r15, counter_set_size
  add r15, rax

  ; record the number of ctx switches
  ; syscall preserves values of r15,
  ; so we are in good shape
  lea rdx, [r15 + ctx_swtch_offset]
  read ctx_swtch_fd, rdx, 8

  ; initialize now because sycall overwrites some regs
  mov rdx, tmp
  mov [rdx], r15
  initialize
  mov rdx, tmp
  mov r15, [rdx]
  
  ; bring in cache before we measure cache misses
  mov [r15], rax
  cpuid ;

  ; read cache misses before we serialize and read cycles
  ; put a label here so that we can override the pmc index
l1_read_misses_a:
  read_perf_counter 0, rcx
  mov [r15 + l1_read_misses_offset], rcx

l1_write_misses_a:
  read_perf_counter 0, rcx
  mov [r15 + l1_write_misses_offset], rcx

icache_misses_a:
  read_perf_counter 0, rcx
  mov [r15 + icache_misses_offset], rcx

  cpuid ; serialize

  read_perf_counter counter_core_cyc, rcx
  mov [r15 + core_cyc_offset], rcx

; re-initialize clobbered registers
  mov rax, r11
  mov rbx, rax
  mov rcx, rax
  mov rdx, rax
  mov r15, rax

  test_impl

  ;; CPUID's runtime depends on value of rax!!!
  xor rax, rax
  cpuid ; serialize

  ; read counters (
  ; tsc = r11,
  ; core_cyc = r12,
  ; l1_read = r13
  ; l1_write = r14,
  ; icache = r8)
  read_perf_counter counter_core_cyc, r12

  ; prevent the processor from issuing insts
  ; before we read core cyc
  xor rax, rax
  cpuid

l1_read_misses_b:
  read_perf_counter 0, r13
l1_write_misses_b:
  read_perf_counter 0, r14
icache_misses_b:
  read_perf_counter 0, r8
  mov rax, counter_array 

  ; rbx = i
  mov rbx, iterator
  mov rbx, [rbx]

  ; r15 = counter + 8 + i*(size of one set of counters)
  mov r15, rbx
  imul r15, counter_set_size
  add r15, rax

  mov rdx, [r15+tsc_offset]
  sub r11, rdx
  mov [r15 + tsc_offset], r11

  mov rdx, [r15 + core_cyc_offset]
  sub r12, rdx
  mov [r15 + core_cyc_offset], r12 

  mov rdx, [r15 + l1_read_misses_offset]
  sub r13, rdx
  mov [r15 + l1_read_misses_offset], r13

  mov rdx, [r15 + l1_write_misses_offset]
  sub r14, rdx
  mov [r15 + l1_write_misses_offset], r14

  mov rdx, [r15 + icache_misses_offset]
  sub r8, rdx
  mov [r15 + icache_misses_offset], r8

  ; finally calculate number of context switches
  mov rdx, tmp
  read ctx_swtch_fd, rdx, 8
  mov rdx, tmp
  mov rcx, [rdx]
  mov rdx, [r15 + ctx_swtch_offset]
  sub rcx, rdx
  mov qword [r15 + ctx_swtch_offset], rcx

  inc rbx
  cmp rbx, iterations
  mov rax, iterator
  mov [rax], rbx
  jb test_loop

  ;;;;;;;;;;; TEST END ;;;;;;;;;;;

%if debug
  mov       rax, 1                  ; system call for write
  mov       rdi, 1                  ; file handle 1 is stdout
  mov       rsi, msg_test_done      ; address of string to output
  mov       rdx, 17                 ; number of bytes
  syscall                           ; invoke operating system to do the write
%endif

  mov rdi, 0
  mov rax, sys_exit
  syscall

%if debug
msg_mapping_done: db "finished mapping", 10
msg_test_done: db "finished testing", 10
%endif

; marker of the begin of test code,
code_end:
  nop
