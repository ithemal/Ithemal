; Toassemble:	nasm -f elf64  -o a.out example.asm

	SECTION .text
        global main
main:

        ;;      START_MARKER
        mov ebx, 111
        db 0x64, 0x67, 0x90

        ;;      CODE GOES HERE

        ;;      END_MARKER
        mov ebx, 222
        db 0x64, 0x67, 0x90
