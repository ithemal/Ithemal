/**
        To assemble: gcc -c -o a.out example.S
        Or:          as -o a.out example.S
        */

        .text
        .global main
main:

        //      START_MARKER
        movl $111, %ebx
        .byte 0x64, 0x67, 0x90

        // CODE BLOCK GOES HERE

        //      END_MARKER
        movl $222, %ebx
        .byte 0x64, 0x67, 0x90
