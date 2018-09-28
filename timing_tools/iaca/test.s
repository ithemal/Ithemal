	.file	"test.c"
	.text
	.globl	foo
	.type	foo, @function
foo:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -4(%rbp)
	jmp	.L2
.L3:
#APP
# 5 "test.c" 1
	
	  movl $111, %ebx
	  .byte 0x64, 0x67, 0x90
# 0 "" 2
#NO_APP
.L2:
.rept 100	
.endr
#APP
# 7 "test.c" 1
	
	  movl $222, %ebx
	  .byte 0x64, 0x67, 0x90
# 0 "" 2
#NO_APP
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	foo, .-foo
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
