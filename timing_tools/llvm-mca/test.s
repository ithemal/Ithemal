	.text
	.file	"time_test.cpp"
	.globl	_Z4testi                # -- Begin function _Z4testi
	.p2align	4, 0x90
	.type	_Z4testi,@function
_Z4testi:                               # @_Z4testi
	.cfi_startproc
# %bb.0:
                                        # kill: def $edi killed $edi def $rdi
	#APP
	# LLVM-MCA-BEGIN test
	#NO_APP
	#APP
	# LLVM-MCA-END
	#NO_APP
	retq
.Lfunc_end0:
	.size	_Z4testi, .Lfunc_end0-_Z4testi
	.cfi_endproc
                                        # -- End function

	.ident	"clang version 7.0.0 (https://github.com/llvm-mirror/clang.git 769fd39841d862d24f81ac3e968a6d007626ccda) (https://github.com/llvm-mirror/llvm.git 502e1bf614169af7dd73d8f28445bf397b9fc45b)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
