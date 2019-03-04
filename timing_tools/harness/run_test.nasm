%define page_size 4096

%define sys_munmap  11
%define sys_mmap    9
%define sys_exit    60

%define iterations 16

%define shm_fd 42

%define PROT_READ   0x1
%define PROT_WRITE  0x2

%define MAP_SHARED  0x01

%define tsc_offset      0
%define core_cyc_offset 8
%define l1_read_misses_offset  16

%define aux_mem   0x0000700000000000
%define counter_array     0x0000700000000008
%define highest_addr      0x0000700000001000
%define iterator          aux_mem

%ifndef NDEBUG 
%define debug 1
%else
%define debug 0
%endif

%define init_value 0x2324000

%define counter_core_cyc 0x40000001
; this is not the most portable way of deciding the pmc index
; but we setup the l1 read miss counter first,
; so it should be fine?
%define counter_l1_read_misses  0

%define reps 100

%macro setup_memory 0
  ; round begin/r13 down to page boundary
  mov r13, begin
  round_to_page_boundary(r13)
  
  ; unmap everything up to begin/r13
  mov rax, sys_munmap
  mov rdi, 0x0  ; addr
  mov rsi, r13  ; len
  syscall

  ; round end/r14 up to page boundary
  mov r14, end
  add r14, page_size
  add r14, page_size
  round_to_page_boundary(r14)

  ; unmap everything starting from end/r14
  mov rdi, r14  ; addr
  mov rsi, highest_addr
  sub rsi, r14  ; len
  mov rax, sys_munmap
  syscall

  ; map a page to keep the counters
  mov rdi, aux_mem ; address
  mov rsi, 4096               ; length
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

; marker of the begin of test code,
; which we need to make sure we don't
; acccidentally unmap itself
begin: 

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

.test_loop:

  initialize

  ; make sure the cell for tsc is in cache
  mov rax, counter_array 
  ; rbx = i
  mov rbx, iterator
  mov rbx, [rbx]
  ; r15 = counter + i*24
  mov r15, rbx
  imul r15, 24
  add r15, rax
  mov [r15], rax
  
  ; bring in cache
  mov [r15], rax
  cpuid ; serialize

  read_time_stamp rcx
  mov [r15 + tsc_offset], rcx

  read_perf_counter counter_core_cyc, rcx
  mov [r15 + core_cyc_offset], rcx

  read_perf_counter counter_l1_read_misses, rcx
  mov [r15 + l1_read_misses_offset], rcx

  ; re-initialize clobbered registers
  mov rax, init_value
  mov rbx, init_value
  mov rcx, init_value
  mov rdx, init_value
  mov r15, init_value

  test_impl

  ;; CPUID's runtime depends on value of rax!!!
  xor rax, rax
  cpuid ; serialize

  ; read counters (tsc = r11, core_cyc = r12, ref_cyc = r13)
  read_time_stamp r11
  read_perf_counter counter_core_cyc, r12
  read_perf_counter counter_l1_read_misses, r13
  mov rax, counter_array 

  ; rbx = i
  mov rbx, iterator
  mov rbx, [rbx]

  ; r15 = counter + 8 + i*24
  mov r15, rbx
  imul r15, 24
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

  inc rbx
  cmp rbx, iterations
  mov rax, iterator
  mov [rax], rbx
  jb .test_loop

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
end:
  nop
