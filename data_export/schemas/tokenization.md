# x86 tokenization

### Instruction tokenization

`opcode,<D>,src1,src2,...,<D>,dst1,dst2,...,<D>`

### Memory operands 

There are different types of memory operands. At a high-level there are mainly three types of addressing modes.

* base-displacement addressing: `[base + scale * index + disp]`, where `base` and `index` are registers. `scale` and `disp`
are integer immediates.
* rip-relative addressing: the addresses are given as offsets relative to the program counter.
* absolute addressing: the addresses contain an absolute virtual memory address, these are changed by loader at runtime.

Apart from these there are near and far addressing modes, addressing using segment registers etc. which I will not go into detail
and will be ignored for the purposes of tokenization.

Any memory operand will be enclosed inside `<M>` tags. `<M>` tag is a special token. Within these tags the operands used to calculate
memory will be listed as follows.

* base-displacement addressing: `base,index,INT_IMMED` each component is optional, but there should be at least one of base, index or
INT_IMMED. e.g., `[eax]` will be tokenized to `<M>,eax,<M>`, where as `[eax + 4 * ebx + 0x55]` will be tokenized to `<M>,eax,ebx,INT_IMMED,<M>`.
* rip-relative: `<M>,INT_IMMED,<M>`
* absolute addressing: `<M>,INT_IMMED,<M>`

### Flags register

We will detect whether an instruction reads/writes to flag registers and emit it as a source/destination operand respectively.
