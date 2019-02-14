;----------------------------------------------------------------------------
;                        PMCTestB64.nasm              © 2013-08-20 Agner Fog
;
;                PMC Test program for multiple threads
;                           NASM syntax
;
; This program is intended for testing the performance of a little piece of 
; code written in assembly language. 
; The code to test is inserted at the place marked "Test code start".
; All sections that can be modified by the user are marked with ###########. 
; 
; The code to test will be executed REPETITIONS times and the test results
; will be output for each repetition. This program measures how many clock
; cycles the code to test takes in each repetition. Furthermore, it is 
; possible to set a number of Performance Monitor Counters (PMC) to count 
; the number of micro-operations (uops), cache misses, branch mispredictions,
; etc.
; 
; The setup of the Performance Monitor Counters is microprocessor-specific.
; The specifications for PMC setup for each microprocessor family is defined
; in the tables CounterDefinitions and CounterTypesDesired.
; 
; See PMCTest.txt for instructions.
; 
; (c) Copyright 2000 - 2013 by Agner Fog. GNU General Public License www.gnu.org/licenses
;-----------------------------------------------------------------------------

default rel

; Operating system: 0 = Linux, 1 = Windows
%ifndef WINDOWS
%define  WINDOWS  0
%endif

; %define USEAVX 1

; Define whether AVX and YMM registers used
%ifndef  USEAVX
%define  USEAVX   0
%endif

; Define cache line size (to avoid threads sharing cache lines):
%define CACHELINESIZE  64

; Define warmup count to get into max frequency state
%define WARMUPCOUNT 10000000

global TestLoop
global CounterTypesDesired
global NumThreads
global MaxNumCounters
global UsePMC
global PThreadData
global ThreadDataSize
global ClockResultsOS
global PMCResultsOS
global ThreadData
global NumCounters
global Counters
global EventRegistersUsed
global UserData
global RatioOut
global TempOut
global RatioOutTitle
global TempOutTitle


SECTION .data   align = CACHELINESIZE


;##############################################################################
;#
;#            List of desired counter types and other user definitions
;#
;##############################################################################
; Here you can select which performance monitor counters you want for your test.
; Select id numbers from the table CounterDefinitions[] in PMCTestA.cpp.

%define USE_PERFORMANCE_COUNTERS   1        ; Tell if you are using performance counters

; Maximum number of PMC counters
%define MAXCOUNTERS   6              ; must match value in PMCTest.h

; Number of PMC counters
%define NUM_COUNTERS  4              ; must match value in PMCTest.h

CounterTypesDesired:
    DD      1        ; core cycles (Intel only)
    DD      9        ; instructions
    DD    100        ; uops

times (MAXCOUNTERS - ($-CounterTypesDesired)/4)  DD 0

; Number of repetitions of test.
%define REPETITIONS  8

; Number of threads
%define NUM_THREADS   1

; Subtract overhead from clock counts (0 if not)
%define SUBTRACT_OVERHEAD  1

; Number of repetitions in loop to find overhead
%define OVERHEAD_REPETITIONS  4

; Define array sizes
%assign MAXREPEAT  REPETITIONS

;------------------------------------------------------------------------------
;
;                  global data
;
;------------------------------------------------------------------------------


; Per-thread data:
align   CACHELINESIZE, DB 0
; Data for first thread
ThreadData:                                                ; beginning of thread data block
CountTemp:     times  (MAXCOUNTERS + 1)          DD   0    ; temporary storage of counts
CountOverhead: times  (MAXCOUNTERS + 1)          DD  -1    ; temporary storage of count overhead
ClockResults:  times   REPETITIONS               DD   0    ; clock counts
PMCResults:    times  (REPETITIONS*MAXCOUNTERS)  DD   0    ; PMC counts
align 8, DB 0
RSPSave                                          DQ   0    ; save stack pointer
ALIGN   CACHELINESIZE, DB 0                                ; Make sure threads don't use same cache lines
THREADDSIZE  equ     ($ - ThreadData)                      ; size of data block for each thread

; Define data blocks of same size for remaining threads
%if  NUM_THREADS > 1
  times ((NUM_THREADS-1)*THREADDSIZE)            DB 0
%endif

; Global data
PThreadData     DQ    ThreadData                 ; Pointer to measured data for all threads
NumCounters     DD    0                          ; Will be number of valid counters
MaxNumCounters  DD    NUM_COUNTERS               ; Tell PMCTestA.CPP length of CounterTypesDesired
UsePMC          DD    USE_PERFORMANCE_COUNTERS   ; Tell PMCTestA.CPP if RDPMC used. Driver needed
NumThreads      DD    NUM_THREADS                ; Number of threads
ThreadDataSize  DD    THREADDSIZE                ; Size of each thread data block
ClockResultsOS  DD    ClockResults-ThreadData    ; Offset to ClockResults
PMCResultsOS    DD    PMCResults-ThreadData      ; Offset to PMCResults
Counters:             times MAXCOUNTERS   DD 0   ; Counter register numbers used will be inserted here
EventRegistersUsed    times MAXCOUNTERS   DD 0   ; Set by MTMonA.cpp
RatioOut        DD    0, 0, 0, 0                 ; optional ratio output. Se PMCTest.h
TempOut         DD    0                          ; optional arbitrary output. Se PMCTest.h
RatioOutTitle   DQ    0                          ; optional column heading
TempOutTitle    DQ    0                          ; optional column heading



;##############################################################################
;#
;#                 User data
;#
;##############################################################################
ALIGN   CACHELINESIZE, DB 0

; Put any data definitions your test code needs here

UserData           times 100000H  DB 0
OrigRsp          DB    0
OrigRbp          DB    0


;------------------------------------------------------------------------------
;
;                  Macro definitions
;
;------------------------------------------------------------------------------

%macro SERIALIZE 0             ; serialize CPU
       ; my additions - have values for all registers
       ;mov rsi, 0x50
       ;mov rdi, 0x100
       ;mov r8, 0x60
       ;mov r9, 0x70
       ;mov r10, 0x80
       ;mov r11, 0x90
       ;mov r12, 0x100
       ; end my additions
       xor     eax, eax
       cpuid
%endmacro

%macro CLEARXMMREG 1           ; clear one xmm register
   pxor xmm%1, xmm%1
%endmacro 

%macro CLEARALLXMMREG 0        ; set all xmm or ymm registers to 0
   %if  USEAVX
      VZEROALL                 ; set all ymm registers to 0
   %else
      %assign i 0
      %rep 16
         CLEARXMMREG i         ; set all 16 xmm registers to 0
         %assign i i+1
      %endrep
   %endif
%endmacro


;------------------------------------------------------------------------------
;
;                  Test Loop
;
;------------------------------------------------------------------------------
SECTION .text   align = 16

;extern "C" int TestLoop (int thread) {
; This function runs the code to test REPETITIONS times
; and reads the counters before and after each run:

TestLoop:
        push    rbx
        push    rbp
        push    r12
        push    r13
        push    r14
        push    r15
%if     WINDOWS                    ; These registers must be saved in Windows, not in Linux
        push    rsi
        push    rdi
        sub     rsp, 0A8H           ; Space for saving xmm6 - 15 and align
        movaps  [rsp], xmm6
        movaps  [rsp+10H], xmm7
        movaps  [rsp+20H], xmm8
        movaps  [rsp+30H], xmm9
        movaps  [rsp+40H], xmm10
        movaps  [rsp+50H], xmm11
        movaps  [rsp+60H], xmm12
        movaps  [rsp+70H], xmm13
        movaps  [rsp+80H], xmm14
        movaps  [rsp+90H], xmm15        
        mov     r15d, ecx          ; Thread number
%else   ; Linux
        mov     r15d, edi          ; Thread number
%endif
        
; Register use:
;   r13: pointer to thread data block
;   r14: loop counter
;   r15: thread number
;   rax, rbx, rcx, rdx: scratch
;   all other registers: available to user program

;##############################################################################
;#
;#                 Warm up
;#
;##############################################################################
; Get into max frequency state

%if WARMUPCOUNT

        mov ecx, WARMUPCOUNT / 10
        mov eax, 1
        align 16
Warmuploop:
        %rep 10
        imul eax, ecx
        %endrep
        dec ecx
        jnz Warmuploop

%endif


;##############################################################################
;#
;#                 User Initializations 
;#
;##############################################################################
; You may add any initializations your test code needs here.
; Registers esi, edi, ebp and r8 - r12 will be unchanged from here to the 
; Test code start.
; 

        finit                ; clear all FP registers
        
        CLEARALLXMMREG       ; clear all xmm or ymm registers

        imul eax, r15d, 2020h ; separate data for each thread
        lea rsi, [UserData]
        add rsi, rax
        lea rdi, [rsi+120h]
        xor ebp, ebp

        ;;;;;; zero everything ;;;;;;;;;;;
        xor rax, rax
        xor rbx, rbx
        xor rcx, rcx
        xor rdx, rdx
        xor rsi, rsi
        xor rdi, rdi
        ;xor rbp, rbp
        ;xor rsp, rsp
        xor r8, r8
        xor r9, r9
        xor r10, r10
        xor r11, r11
        xor r12, r12
        xor r14, r14
        xor r15, r15
        

;##############################################################################
;#
;#                 End of user Initializations 
;#
;##############################################################################

        lea     r13, [ThreadData]             ; address of first thread data block
        imul    eax, r15d, THREADDSIZE        ; offset to thread data block
        add     r13, rax                      ; address of current thread data block
        mov     [r13+(RSPSave-ThreadData)],rsp ; save stack pointer

%if  SUBTRACT_OVERHEAD
; First test loop. Measure empty code
        xor     r14d, r14d                    ; Loop counter

TEST_LOOP_1:

        SERIALIZE
      
        ; Read counters
%assign i  0
%rep    NUM_COUNTERS
        mov     ecx, [Counters + i*4]
        rdpmc
        mov     [r13 + i*4 + 4 + (CountTemp-ThreadData)], eax
%assign i  i+1
%endrep
      

        SERIALIZE

        ; read time stamp counter
        rdtsc
        mov     [r13 + (CountTemp-ThreadData)], eax

        SERIALIZE

        ; Empty. Test code goes here in next loop

        SERIALIZE

        ; read time stamp counter
        rdtsc
        sub     [r13 + (CountTemp-ThreadData)], eax        ; CountTemp[0]

        SERIALIZE

        ; Read counters
%assign i  0
%rep    NUM_COUNTERS
        mov     ecx, [Counters + i*4]
        rdpmc
        sub     [r13 + i*4 + 4 + (CountTemp-ThreadData)], eax
%assign i  i+1
%endrep

        SERIALIZE

        ; find minimum counts
%assign i  0
%rep    NUM_COUNTERS + 1
        mov     eax, [r13+i*4+(CountTemp-ThreadData)]       ; -count
        neg     eax
        mov     ebx, [r13+i*4+(CountOverhead-ThreadData)]   ; previous count
        cmp     eax, ebx
        cmovb   ebx, eax
        mov     [r13+i*4+(CountOverhead-ThreadData)], ebx   ; minimum count        
%assign i  i+1
%endrep
        
        ; end second test loop
        inc     r14d
        cmp     r14d, OVERHEAD_REPETITIONS
        jb      TEST_LOOP_1

%endif  ; SUBTRACT_OVERHEAD

        
; Second test loop. Measure user code
        xor     r14d, r14d                    ; Loop counter

TEST_LOOP_2:

        SERIALIZE
      
        ; Read counters
%assign i  0
%rep    NUM_COUNTERS
        mov     ecx, [Counters + i*4]
        rdpmc
        mov     [r13 + i*4 + 4 + (CountTemp-ThreadData)], eax
%assign i  i+1
%endrep

        ;;;;;;;;;;;;; INITIALIZE ;;;;;;;;;;;;;
        ; SAVE RSP
        mov [OrigRsp], rsp
        mov [OrigRbp], rbp

        ; point stack/frame pointers to middle of UserData
        lea rsp, [UserData + 131072]
        lea rbp, [UserData + 131072]
        xor rsi, rsi
        xor rdi, rdi
        ; these gets overwritten by CPUID, so we initialize between
        ; SERIALIZE and the actual test code
        ; xor rax, rax
        ; xor rbx, rbx
        ; xor rcx, rcx
        ; xor rdx, rdx
        ;;;;;;;;;;;;; END INIT ;;;;;;;;;;;;;

        SERIALIZE

        ; read time stamp counter
        rdtsc
        mov     [r13 + (CountTemp-ThreadData)], eax

        SERIALIZE

;##############################################################################
;#
;#                 Test code start
;#
;##############################################################################

; Put the assembly code to test here
; Don't modify r13, r14, r15!

; ½½

;mov dword [UserData], 100
align 16
LL:

;;;; zero regs overwritten by CPUID
xor rax, rax
xor rbx, rbx
xor rcx, rcx
xor rdx, rdx

%REP 100
%ENDREP


;dec dword [UserData]
;jnz LL


;##############################################################################
;#
;#                 Test code end
;#
;##############################################################################

        SERIALIZE

        ;;;;;;;;;;;;;;;;;;

        ; read time stamp counter
        rdtsc
        ;;;;;;; RESTORER STACK PTR ;;;;;;;;;;
        mov rsp, [OrigRsp]
        mov rbp, [OrigRbp]
        ;;;;;;; END RESTORE ;;;;;;;;;;;;;;;;
        sub     [r13 + (CountTemp-ThreadData)], eax        ; CountTemp[0]

        SERIALIZE

        ; Read counters
%assign i  0
%rep    NUM_COUNTERS
        mov     ecx, [Counters + i*4]
        rdpmc
        sub     [r13 + i*4 + 4 + (CountTemp-ThreadData)], eax  ; CountTemp[i+1]
%assign i  i+1
%endrep

        SERIALIZE

        ; subtract counts before from counts after
        mov     eax, [r13 + (CountTemp-ThreadData)]            ; -count
        neg     eax
%if     SUBTRACT_OVERHEAD
        sub     eax, [r13+(CountOverhead-ThreadData)]   ; overhead clock count        
%endif  ; SUBTRACT_OVERHEAD        
        mov     [r13+r14*4+(ClockResults-ThreadData)], eax      ; save clock count
        
%assign i  0
%rep    NUM_COUNTERS
        mov     eax, [r13 + i*4 + 4 + (CountTemp-ThreadData)]
        neg     eax
%if     SUBTRACT_OVERHEAD
        sub     eax, [r13+i*4+4+(CountOverhead-ThreadData)]   ; overhead pmc count        
%endif  ; SUBTRACT_OVERHEAD        
        mov     [r13+r14*4+i*4*REPETITIONS+(PMCResults-ThreadData)], eax      ; save count        
%assign i  i+1
%endrep
        
        ; end second test loop
        inc     r14d
        cmp     r14d, REPETITIONS
        jb      TEST_LOOP_2

        ; clean up
        mov     rsp, [r13+(RSPSave-ThreadData)]   ; restore stack pointer        
        finit
        cld
%if  USEAVX
        VZEROALL                       ; clear all ymm registers
%endif        

        ; return REPETITIONS;
        mov     eax, REPETITIONS
        
%if     WINDOWS                        ; Restore registers saved in Windows
        movaps  xmm6, [rsp]
        movaps  xmm7, [rsp+10H]
        movaps  xmm8, [rsp+20H]
        movaps  xmm9, [rsp+30H]
        movaps  xmm10, [rsp+40H]
        movaps  xmm11, [rsp+50H]
        movaps  xmm12, [rsp+60H]
        movaps  xmm13, [rsp+70H]
        movaps  xmm14, [rsp+80H]
        movaps  xmm15, [rsp+90H]
        add     rsp, 0A8H           ; Free space for saving xmm6 - 15
        pop     rdi
        pop     rsi
%endif
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbp
        pop     rbx
        ret
        
; End of TestLoop
