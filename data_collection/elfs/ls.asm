read 10000 bytes
first byte 50
  0x00000000006020c0 push   %rax %rsp -> %rsp 0xfffffff8(%rsp)[8byte]
push   rax
mov    ecx, 0x00412c88
mov    edx, 0x00000ea6
mov    esi, 0x00413736
mov    edi, 0x00413c98
call   0x0000000000601c70
nop    dword ptr [rax+rax+0x00]
push   r15
push   r14
push   r13
push   r12
push   rbp
mov    rbp, rsi
push   rbx
mov    ebx, edi
sub    rsp, 0x00000388
mov    rdi, qword ptr [rsi]
mov    rax, qword ptr [fs:0x28]
mov    qword ptr [rsp+0x00000378], rax
xor    eax, eax
call   0x000000000060cec0
mov    esi, 0x00416919
mov    edi, 0x00000006
call   0x0000000000601f30
mov    esi, 0x0041381c
mov    edi, 0x00413800
call   0x0000000000601b60
mov    edi, 0x00413800
call   0x0000000000601b20
mov    edi, 0x0040a200
mov    <rel> dword ptr [0x0000000000819da0], 0x00000002
call   0x0000000000611700
mov    rax, 0x8000000000000000
mov    <rel> dword ptr [0x000000000081a850], 0x00000000
mov    <rel> byte ptr [0x000000000081a8f0], 0x01
mov    <rel> qword ptr [0x000000000081a9a0], rax
mov    eax, <rel> dword ptr [0x0000000000819d8c]
mov    <rel> qword ptr [0x000000000081a9b0], 0x00000000
mov    <rel> qword ptr [0x000000000081a9a8], 0xffffffff
mov    <rel> byte ptr [0x000000000081a910], 0x00
cmp    eax, 0x02
jz     0x0000000000602a23
cmp    eax, 0x03
jz     0x00000000006021cf
sub    eax, 0x01
jz     0x00000000006021aa
call   0x0000000000601a40
mov    edi, 0x00000001
call   0x0000000000601aa0
test   eax, eax
jz     0x000000000060300c
mov    <rel> dword ptr [0x000000000081a970], 0x00000002
mov    <rel> byte ptr [0x000000000081a910], 0x01
jmp    0x00000000006021e5
mov    esi, 0x00000005
xor    edi, edi
mov    <rel> dword ptr [0x000000000081a970], 0x00000000
call   0x000000000060de60
mov    edi, 0x0041382e
mov    <rel> dword ptr [0x000000000081a96c], 0x00000000
mov    <rel> dword ptr [0x000000000081a968], 0x00000000
mov    <rel> byte ptr [0x000000000081a967], 0x00
mov    <rel> byte ptr [0x000000000081a965], 0x00
mov    <rel> byte ptr [0x000000000081a964], 0x00
mov    <rel> dword ptr [0x000000000081a94c], 0x00000000
mov    <rel> byte ptr [0x000000000081a934], 0x00
mov    <rel> dword ptr [0x000000000081a930], 0x00000001
mov    <rel> byte ptr [0x000000000081a92e], 0x00
mov    <rel> byte ptr [0x000000000081a92d], 0x00
mov    <rel> dword ptr [0x000000000081a928], 0x00000000
mov    <rel> qword ptr [0x000000000081a920], 0x00000000
mov    <rel> qword ptr [0x000000000081a918], 0x00000000
mov    <rel> byte ptr [0x000000000081a99d], 0x00
call   0x00000000006019e0
test   rax, rax
mov    r12, rax
jz     0x000000000060229f
mov    ecx, 0x00000004
mov    edx, 0x00416460
mov    esi, 0x00416480
mov    rdi, rax
call   0x0000000000609670
test   eax, eax
js     0x0000000000602fa6
cwde
xor    edi, edi
mov    esi, dword ptr [rax*4+0x00416460]
call   0x000000000060de60
mov    edi, 0x0041383c
mov    <rel> qword ptr [0x000000000081a8e8], 0x00000050
call   0x00000000006019e0
mov    r12, rax
lea    rax, [rsp+0x40]
test   r12, r12
mov    qword ptr [rsp+0x20], rax
jz     0x00000000006022d1
cmp    byte ptr [r12], 0x00
jnz    0x0000000000602fd8
mov    rdx, qword ptr [rsp+0x20]
xor    eax, eax
mov    esi, 0x00005413
mov    edi, 0x00000001
call   0x0000000000601cd0
cmp    eax, 0xff
jz     0x00000000006022fd
movzx  eax, word ptr [rsp+0x42]
test   ax, ax
jz     0x00000000006022fd
mov    <rel> qword ptr [0x000000000081a8e8], rax
mov    edi, 0x00413844
call   0x00000000006019e0
test   rax, rax
mov    r12, rax
mov    <rel> qword ptr [0x000000000081a8f8], 0x00000008
jz     0x0000000000602342
mov    rcx, qword ptr [rsp+0x20]
xor    r8d, r8d
xor    edx, edx
xor    esi, esi
mov    rdi, rax
call   0x00000000006106b0
test   eax, eax
jnz    0x00000000006039db
mov    rax, qword ptr [rsp+0x40]
mov    <rel> qword ptr [0x000000000081a8f8], rax
xor    r14d, r14d
xor    r13d, r13d
xor    r12d, r12d
nop    dword ptr [rax+rax+0x00]
lea    r8, [rsp+0x38]
mov    ecx, 0x00413080
mov    edx, 0x00415bc8
mov    rsi, rbp
mov    edi, ebx
mov    dword ptr [rsp+0x38], 0xffffffff
call   0x0000000000601bd0
cmp    eax, 0xff
jz     0x0000000000602a3e
add    eax, 0x00000083
cmp    eax, 0x00000112
jnbe   0x0000000000602a19
jmp    qword ptr [rax*8+0x00412330]
mov    <rel> byte ptr [0x000000000081a965], 0x01
mov    <rel> dword ptr [0x000000000081a970], 0x00000000
jmp    0x0000000000602350
mov    r14d, 0x00000001
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a934], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a960], 0x000000b0
mov    <rel> dword ptr [0x000000000081a954], 0x000000b0
mov    <rel> qword ptr [0x000000000081a958], 0x00000001
mov    <rel> qword ptr [0x0000000000819d80], 0x00000001
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000000
mov    <rel> byte ptr [0x0000000000819d89], 0x00
jmp    0x0000000000602350
cmp    <rel> dword ptr [0x000000000081a970], 0x00
mov    <rel> dword ptr [0x000000000081a928], 0x00000002
mov    <rel> dword ptr [0x000000000081a968], 0xffffffff
jz     0x0000000000603456
mov    <rel> byte ptr [0x000000000081a964], 0x00
mov    <rel> byte ptr [0x000000000081a949], 0x00
mov    r13d, 0x00000001
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a92d], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a96c], 0x00000001
jmp    0x0000000000602350
mov    esi, 0x00000005
xor    edi, edi
call   0x000000000060de60
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a928], 0x00000002
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a99d], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a968], 0x00000001
mov    r13d, 0x00000001
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a968], 0xffffffff
mov    r13d, 0x00000001
jmp    0x0000000000602350
mov    rcx, qword ptr [rsp+0x20]
mov    rdi, <rel> qword ptr [0x0000000000819e60]
xor    r8d, r8d
xor    edx, edx
xor    esi, esi
call   0x00000000006106b0
test   eax, eax
jnz    0x000000000060341d
mov    rax, qword ptr [rsp+0x40]
mov    <rel> qword ptr [0x000000000081a8f8], rax
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a968], 0x00000002
mov    r13d, 0x00000001
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a92e], 0x01
jmp    0x0000000000602350
mov    esi, 0x00000003
xor    edi, edi
call   0x000000000060de60
jmp    0x0000000000602350
xor    esi, esi
xor    edi, edi
call   0x000000000060de60
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a930], 0x00000005
jmp    0x0000000000602350
mov    edi, 0x00000010
mov    r15, <rel> qword ptr [0x0000000000819e60]
call   0x0000000000610460
mov    rdx, <rel> qword ptr [0x000000000081a920]
mov    qword ptr [rax], r15
mov    qword ptr [rax+0x08], rdx
mov    <rel> qword ptr [0x000000000081a920], rax
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a930], 0x00000003
jmp    0x0000000000602350
mov    <rel> byte ptr [0x0000000000819d88], 0x00
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a94c], 0x00000003
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a950], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000002
jmp    0x0000000000602350
mov    edi, 0x00000010
call   0x0000000000610460
mov    rdx, <rel> qword ptr [0x000000000081a920]
mov    qword ptr [rax], 0x00413864
mov    edi, 0x00000010
mov    <rel> qword ptr [0x000000000081a920], rax
mov    qword ptr [rax+0x08], rdx
call   0x0000000000610460
mov    rdx, <rel> qword ptr [0x000000000081a920]
mov    qword ptr [rax], 0x00413863
mov    qword ptr [rax+0x08], rdx
mov    <rel> qword ptr [0x000000000081a920], rax
jmp    0x0000000000602350
cmp    <rel> dword ptr [0x000000000081a928], 0x00
jnz    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a928], 0x00000001
jmp    0x0000000000602350
cmp    <rel> dword ptr [0x000000000081a970], 0x00
jz     0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000001
jmp    0x0000000000602350
xor    edi, edi
call   0x0000000000608f70
mov    eax, <rel> dword ptr [0x0000000000819d8c]
mov    rcx, <rel> qword ptr [0x0000000000819d90]
cmp    eax, 0x01
jz     0x0000000000603413
cmp    eax, 0x02
mov    esi, 0x0041380f
mov    eax, 0x0041380e
cmovnz rsi, rax
mov    rdi, <rel> qword ptr [0x0000000000819e30]
mov    qword ptr [rsp], 0x00000000
mov    r9d, 0x004138bd
mov    r8d, 0x004138cd
mov    edx, 0x004137fc
xor    eax, eax
call   0x0000000000610350
xor    edi, edi
call   0x0000000000602010
mov    r12, <rel> qword ptr [0x0000000000819e60]
jmp    0x0000000000602350
mov    r9, <rel> qword ptr [0x0000000000819d98]
mov    rsi, <rel> qword ptr [0x0000000000819e60]
mov    r8d, 0x00000004
mov    ecx, 0x00412f50
mov    edx, 0x00412f80
mov    edi, 0x00413883
call   0x0000000000609940
mov    eax, dword ptr [rax*4+0x00412f50]
mov    <rel> dword ptr [0x000000000081a96c], eax
jmp    0x0000000000602350
mov    r9, <rel> qword ptr [0x0000000000819d98]
mov    rsi, <rel> qword ptr [0x0000000000819e60]
mov    r8d, 0x00000004
mov    ecx, 0x00412fb0
mov    edx, 0x00412fe0
mov    edi, 0x0041387c
mov    r13d, 0x00000001
call   0x0000000000609940
mov    eax, dword ptr [rax*4+0x00412fb0]
mov    <rel> dword ptr [0x000000000081a968], eax
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a960], 0x00000090
mov    <rel> dword ptr [0x000000000081a954], 0x00000090
mov    <rel> qword ptr [0x000000000081a958], 0x00000001
mov    <rel> qword ptr [0x0000000000819d80], 0x00000001
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a910], 0x00
jmp    0x0000000000602350
mov    r9, <rel> qword ptr [0x0000000000819d98]
mov    rsi, <rel> qword ptr [0x0000000000819e60]
mov    r8d, 0x00000004
mov    ecx, 0x00416460
mov    edx, 0x00416480
mov    edi, 0x004138ad
call   0x0000000000609940
mov    esi, dword ptr [rax*4+0x00416460]
xor    edi, edi
call   0x000000000060de60
jmp    0x0000000000602350
mov    r9, <rel> qword ptr [0x0000000000819d98]
mov    rsi, <rel> qword ptr [0x0000000000819e60]
mov    r8d, 0x00000004
mov    ecx, 0x004136b0
mov    edx, 0x004136c0
mov    edi, 0x0041389b
call   0x0000000000609940
mov    eax, dword ptr [rax*4+0x004136b0]
mov    <rel> dword ptr [0x000000000081a94c], eax
jmp    0x0000000000602350
mov    edi, 0x00000010
call   0x0000000000610460
mov    rdx, <rel> qword ptr [0x0000000000819e60]
mov    qword ptr [rax], rdx
mov    rdx, <rel> qword ptr [0x000000000081a918]
mov    <rel> qword ptr [0x000000000081a918], rax
mov    qword ptr [rax+0x08], rdx
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a92c], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000000
mov    r12d, 0x00413813
jmp    0x0000000000602350
mov    r9, <rel> qword ptr [0x0000000000819d98]
mov    rsi, <rel> qword ptr [0x0000000000819e60]
mov    r8d, 0x00000004
mov    ecx, 0x00413010
mov    edx, 0x00413040
mov    edi, 0x0041388a
call   0x0000000000609940
mov    eax, dword ptr [rax*4+0x00413010]
mov    <rel> dword ptr [0x000000000081a970], eax
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a94c], 0x00000002
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a930], 0x00000004
jmp    0x0000000000602350
mov    rsi, <rel> qword ptr [0x0000000000819e60]
test   rsi, rsi
jz     0x000000000060329c
mov    r9, <rel> qword ptr [0x0000000000819d98]
mov    r8d, 0x00000004
mov    ecx, 0x00412ec0
mov    edx, 0x00412f00
mov    edi, 0x00413893
call   0x0000000000609940
mov    eax, dword ptr [rax*4+0x00412ec0]
cmp    eax, 0x01
jz     0x000000000060329c
cmp    eax, 0x02
jz     0x000000000060328a
mov    <rel> byte ptr [0x000000000081a949], 0x00
jmp    0x0000000000602350
mov    rdi, <rel> qword ptr [0x0000000000819e60]
mov    edx, 0x0061b138
mov    esi, 0x0061b140
call   0x000000000060c030
test   eax, eax
jnz    0x0000000000603be0
mov    eax, <rel> dword ptr [0x000000000081a960]
mov    <rel> dword ptr [0x000000000081a954], eax
mov    rax, <rel> qword ptr [0x000000000081a958]
mov    <rel> qword ptr [0x0000000000819d80], rax
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a966], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000003
jmp    0x0000000000602350
mov    rcx, qword ptr [rsp+0x20]
mov    rdi, <rel> qword ptr [0x0000000000819e60]
xor    r8d, r8d
xor    edx, edx
xor    esi, esi
call   0x00000000006106b0
test   eax, eax
jnz    0x0000000000602920
cmp    qword ptr [rsp+0x40], 0x00
jnz    0x0000000000602954
mov    rdi, <rel> qword ptr [0x0000000000819e60]
call   0x000000000060e150
mov    edx, 0x00000005
mov    r15, rax
mov    esi, 0x0041384c
xor    edi, edi
call   0x0000000000601b80
mov    rcx, r15
mov    rdx, rax
xor    esi, esi
mov    edi, 0x00000002
xor    eax, eax
call   0x0000000000601f90
mov    rax, qword ptr [rsp+0x40]
mov    <rel> qword ptr [0x000000000081a8e8], rax
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a968], 0x00000003
mov    r13d, 0x00000001
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a96c], 0x00000002
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a968], 0x00000004
mov    r13d, 0x00000001
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a964], 0x01
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a967], 0x01
jmp    0x0000000000602350
mov    <rel> byte ptr [0x000000000081a910], 0x01
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a94c], 0x00000001
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000000
mov    <rel> byte ptr [0x0000000000819d88], 0x00
jmp    0x0000000000602350
mov    <rel> dword ptr [0x000000000081a970], 0x00000004
jmp    0x0000000000602350
mov    rbx, <rel> qword ptr [0x0000000000819e70]
mov    esi, 0x00415bf8
xor    edi, edi
mov    edx, 0x00000005
call   0x0000000000601b80
mov    rsi, rbx
mov    rdi, rax
call   0x0000000000601d40
mov    edi, 0x00000002
call   0x0000000000608f70
mov    esi, 0x00000005
xor    edi, edi
mov    <rel> dword ptr [0x000000000081a970], 0x00000002
call   0x000000000060de60
jmp    0x00000000006021e5
cmp    <rel> qword ptr [0x000000000081a958], 0x00
jz     0x00000000006031aa
mov    rdx, <rel> qword ptr [0x000000000081a8e8]
mov    eax, 0x00000001
cmp    rdx, 0x02
jnbe   0x000000000060301b
xor    edi, edi
mov    <rel> qword ptr [0x000000000081a840], rax
call   0x000000000060de20
mov    rdi, rax
mov    <rel> qword ptr [0x000000000081a908], rax
call   0x000000000060de50
cmp    eax, 0x05
jz     0x0000000000603a8c
mov    eax, <rel> dword ptr [0x000000000081a94c]
cmp    eax, 0x01
jbe    0x0000000000602ac9
lea    r14, [rax+0x004138ed]
sub    rax, 0x02
movzx  eax, byte ptr [rax+0x004138ef]
test   al, al
jz     0x0000000000602ac9
mov    rdi, <rel> qword ptr [0x000000000081a908]
add    r14, 0x01
movsx  esi, al
mov    edx, 0x00000001
call   0x000000000060de70
movzx  eax, byte ptr [r14]
test   al, al
jnz    0x0000000000602aa9
xor    edi, edi
call   0x000000000060de20
mov    edx, 0x00000001
mov    esi, 0x0000003a
mov    rdi, rax
mov    <rel> qword ptr [0x000000000081a900], rax
call   0x000000000060de70
cmp    <rel> byte ptr [0x000000000081a950], 0x00
jz     0x0000000000602b02
cmp    <rel> dword ptr [0x000000000081a970], 0x00
jz     0x0000000000602b02
mov    <rel> byte ptr [0x000000000081a950], 0x00
mov    eax, <rel> dword ptr [0x000000000081a96c]
sub    eax, 0x01
cmp    eax, 0x01
jbe    0x0000000000603185
cmp    <rel> dword ptr [0x000000000081a970], 0x00
jz     0x000000000060302d
cmp    <rel> byte ptr [0x000000000081a949], 0x00
mov    r12d, <rel> dword ptr [0x0000000000819e40]
jnz    0x00000000006032e5
cmp    <rel> dword ptr [0x000000000081a930], 0x01
jz     0x0000000000603259
cmp    <rel> byte ptr [0x000000000081a92e], 0x00
jnz    0x000000000060320f
mov    eax, <rel> dword ptr [0x000000000081a968]
cmp    eax, 0x04
jz     0x0000000000602f9a
cmp    eax, 0x02
jz     0x0000000000602f9a
cmp    <rel> dword ptr [0x000000000081a970], 0x00
jz     0x0000000000602f9a
cmp    <rel> byte ptr [0x000000000081a99d], 0x00
jnz    0x0000000000602f9a
cmp    <rel> byte ptr [0x000000000081a964], 0x00
jnz    0x0000000000602f9a
cmp    <rel> byte ptr [0x000000000081a92e], 0x00
mov    <rel> byte ptr [0x000000000081a8e1], 0x00
mov    eax, 0x00000001
jnz    0x0000000000602bc0
cmp    <rel> byte ptr [0x000000000081a949], 0x00
jnz    0x0000000000602bc0
cmp    <rel> dword ptr [0x000000000081a94c], 0x00
jnz    0x0000000000602bc0
cmp    <rel> byte ptr [0x000000000081a92c], 0x00
jnz    0x0000000000602bc0
xor    eax, eax
mov    <rel> byte ptr [0x000000000081a8e0], al
and    <rel> byte ptr [0x000000000081a8e0], 0x01
cmp    <rel> byte ptr [0x000000000081a950], 0x00
jz     0x0000000000602c08
mov    r8d, 0x004021f0
mov    ecx, 0x00402640
xor    edx, edx
xor    esi, esi
mov    edi, 0x0061afc0
call   0x0000000000601c10
mov    r8d, 0x004021f0
mov    ecx, 0x00402640
xor    edx, edx
xor    esi, esi
mov    edi, 0x0061af60
call   0x0000000000601c10
mov    r13d, ebx
mov    edi, 0x00004b00
mov    <rel> qword ptr [0x000000000081a9d8], 0x00000064
sub    r13d, r12d
call   0x0000000000610460
mov    <rel> qword ptr [0x000000000081a9d0], 0x00000000
mov    <rel> qword ptr [0x000000000081a9e0], rax
call   0x00000000006045f0
test   r13d, r13d
jle    0x0000000000603ac0
movsxd rax, r12d
lea    rbp, [rbp+rax*8+0x00]
mov    rdi, qword ptr [rbp+0x00]
xor    esi, esi
add    r12d, 0x01
mov    ecx, 0x00416919
mov    edx, 0x00000001
add    rbp, 0x08
call   0x00000000006076c0
cmp    ebx, r12d
jnle   0x0000000000602c4b
cmp    <rel> qword ptr [0x000000000081a9d0], 0x00
jnz    0x0000000000603a20
mov    rax, <rel> qword ptr [0x000000000081a9b0]
sub    r13d, 0x01
mov    qword ptr [rsp+0x18], rax
jnle   0x0000000000602cf2
jmp    0x0000000000603ba5
nop    dword ptr [rax+rax+0x00]
mov    edx, 0x00000005
mov    esi, 0x00415ce8
xor    edi, edi
call   0x0000000000601b80
movzx  edi, byte ptr [rsp+0x2f]
mov    rdx, r14
mov    rsi, rax
call   0x0000000000605030
mov    rdi, r13
call   0x0000000000601d00
mov    rbx, qword ptr [rsp+0x18]
mov    rdi, qword ptr [rbx]
call   0x0000000000601a10
mov    rdi, qword ptr [rbx+0x08]
call   0x0000000000601a10
mov    rdi, rbx
call   0x0000000000601a10
mov    <rel> byte ptr [0x000000000081a8f0], 0x01
mov    rax, <rel> qword ptr [0x000000000081a9b0]
mov    qword ptr [rsp+0x18], rax
cmp    qword ptr [rsp+0x18], 0x00
jz     0x00000000006038d5
mov    rcx, qword ptr [rsp+0x18]
cmp    <rel> qword ptr [0x000000000081a9e8], 0x00
mov    rax, qword ptr [rcx+0x18]
mov    <rel> qword ptr [0x000000000081a9b0], rax
jz     0x00000000006037d5
mov    r14, qword ptr [rcx]
test   r14, r14
jz     0x00000000006037e2
mov    rax, qword ptr [rsp+0x18]
movzx  ecx, byte ptr [rax+0x10]
mov    rbx, qword ptr [rax+0x08]
mov    byte ptr [rsp+0x2f], cl
call   0x0000000000601a50
mov    rdi, r14
mov    dword ptr [rax], 0x00000000
mov    r12, rax
call   0x0000000000601b40
test   rax, rax
mov    r13, rax
jz     0x00000000006039b5
cmp    <rel> qword ptr [0x000000000081a9e8], 0x00
jz     0x0000000000602e1f
mov    rdi, rax
call   0x0000000000601d90
test   eax, eax
mov    rdx, qword ptr [rsp+0x20]
js     0x00000000006036b5
mov    esi, eax
mov    edi, 0x00000001
call   0x0000000000601ea0
shr    eax, 0x1f
test   al, al
jnz    0x0000000000602c98
mov    rcx, qword ptr [rsp+0x48]
mov    rdx, qword ptr [rsp+0x40]
mov    edi, 0x00000010
mov    qword ptr [rsp+0x10], rcx
mov    qword ptr [rsp+0x08], rdx
call   0x0000000000610460
mov    rcx, qword ptr [rsp+0x10]
mov    rdx, qword ptr [rsp+0x08]
mov    rsi, rax
mov    rdi, <rel> qword ptr [0x000000000081a9e8]
mov    rbp, rax
mov    qword ptr [rax], rcx
mov    qword ptr [rax+0x08], rdx
call   0x000000000060b370
test   rax, rax
jz     0x0000000000603bdb
cmp    rbp, rax
jnz    0x0000000000603702
mov    rax, <rel> qword ptr [0x000000000081a738]
mov    rdx, <rel> qword ptr [0x000000000081a740]
sub    rdx, rax
cmp    rdx, 0x0f
jle    0x0000000000603884
lea    rdx, [rax+0x10]
mov    <rel> qword ptr [0x000000000081a738], rdx
mov    rdx, qword ptr [rsp+0x40]
mov    qword ptr [rax+0x08], rdx
mov    rdx, qword ptr [rsp+0x48]
mov    qword ptr [rax], rdx
cmp    <rel> byte ptr [0x000000000081a92e], 0x00
jnz    0x0000000000602e35
cmp    <rel> byte ptr [0x000000000081a8f0], 0x00
jz     0x0000000000602ef6
cmp    <rel> byte ptr [0x0000000000819be0], 0x00
jnz    0x0000000000602e66
mov    rdi, <rel> qword ptr [0x0000000000819e30]
mov    rax, qword ptr [rdi+0x28]
cmp    rax, qword ptr [rdi+0x30]
jnb    0x0000000000603e04
lea    rdx, [rax+0x01]
mov    qword ptr [rdi+0x28], rdx
mov    byte ptr [rax], 0x0a
add    <rel> qword ptr [0x000000000081a838], 0x01
cmp    <rel> byte ptr [0x000000000081a950], 0x00
mov    <rel> byte ptr [0x0000000000819be0], 0x00
jnz    0x000000000060374e
test   rbx, rbx
mov    rdx, <rel> qword ptr [0x000000000081a900]
mov    rdi, <rel> qword ptr [0x0000000000819e30]
cmovz  rbx, r14
xor    ecx, ecx
mov    rsi, rbx
call   0x0000000000604af0
add    <rel> qword ptr [0x000000000081a838], rax
cmp    <rel> byte ptr [0x000000000081a950], 0x00
jz     0x0000000000602ed3
mov    rax, <rel> qword ptr [0x000000000081a798]
lea    rdx, [rax+0x08]
cmp    <rel> qword ptr [0x000000000081a7a0], rdx
jb     0x00000000006038ba
mov    rdx, <rel> qword ptr [0x000000000081a838]
mov    qword ptr [rax], rdx
add    <rel> qword ptr [0x000000000081a798], 0x08
mov    rcx, <rel> qword ptr [0x0000000000819e30]
mov    edx, 0x00000002
mov    esi, 0x00000001
mov    edi, 0x0041393b
call   0x0000000000601ee0
add    <rel> qword ptr [0x000000000081a838], 0x02
call   0x00000000006045f0
movzx  eax, byte ptr [rsp+0x2f]
mov    qword ptr [rsp+0x08], 0x00000000
mov    dword ptr [rsp+0x10], eax
nop    dword ptr [rax]
mov    dword ptr [r12], 0x00000000
mov    rdi, r13
call   0x0000000000601e40
test   rax, rax
mov    rbp, rax
jz     0x00000000006034e0
lea    rbx, [rax+0x13]
mov    eax, <rel> dword ptr [0x000000000081a928]
cmp    eax, 0x02
jz     0x0000000000603498
cmp    byte ptr [rbp+0x13], 0x2e
jz     0x0000000000603478
test   eax, eax
jnz    0x0000000000603498
mov    r15, <rel> qword ptr [0x000000000081a918]
test   r15, r15
jnz    0x0000000000602f75
jmp    0x0000000000603498
nop    dword ptr [rax+rax+0x00]
mov    r15, qword ptr [r15+0x08]
test   r15, r15
jz     0x0000000000603498
mov    rdi, qword ptr [r15]
mov    edx, 0x00000004
mov    rsi, rbx
call   0x0000000000601c90
test   eax, eax
jnz    0x0000000000602f68
nop    dword ptr [rax+0x00000000]
call   0x0000000000605cb0
jmp    0x0000000000602f10
mov    <rel> byte ptr [0x000000000081a8e1], 0x01
jmp    0x0000000000602bbe
mov    rdi, r12
call   0x000000000060e150
xor    edi, edi
mov    r12, rax
mov    edx, 0x00000005
mov    esi, 0x00415b00
call   0x0000000000601b80
mov    rcx, r12
mov    rdx, rax
xor    esi, esi
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
jmp    0x000000000060229f
xor    r8d, r8d
xor    edx, edx
xor    esi, esi
mov    rcx, rax
mov    rdi, r12
call   0x00000000006106b0
test   eax, eax
jnz    0x00000000006032b3
mov    rax, qword ptr [rsp+0x40]
test   rax, rax
jz     0x00000000006032b3
mov    <rel> qword ptr [0x000000000081a8e8], rax
jmp    0x00000000006022d1
mov    <rel> dword ptr [0x000000000081a970], 0x00000001
jmp    0x00000000006021e5
mov    rax, rdx
mov    ecx, 0x00000003
xor    edx, edx
div    rdx, rax, rcx
jmp    0x0000000000602a62
test   r12, r12
jz     0x0000000000603de6
mov    r14d, 0x00412ca0
mov    r13d, 0x00000006
jmp    0x000000000060305e
nop    dword ptr [rax+0x00]
mov    edi, 0x00000002
call   0x000000000060a390
test   al, al
jz     0x0000000000602b21
add    r12, 0x06
mov    rsi, r12
mov    rdi, r14
mov    rcx, r13
rep cmpsb
jz     0x0000000000603048
cmp    byte ptr [r12], 0x2b
jz     0x0000000000603bf9
mov    ecx, 0x00000004
mov    edx, 0x004136f0
mov    esi, 0x00413700
mov    rdi, r12
call   0x0000000000609670
test   rax, rax
js     0x0000000000603d84
cmp    rax, 0x01
jz     0x0000000000603d69
jle    0x0000000000603c85
cmp    rax, 0x02
jz     0x0000000000603e13
cmp    rax, 0x03
jnz    0x00000000006030c8
mov    edi, 0x00000002
call   0x000000000060a390
test   al, al
jnz    0x0000000000603e2e
mov    rdi, <rel> qword ptr [0x0000000000819bf0]
mov    esi, 0x00413766
call   0x0000000000602080
test   rax, rax
jz     0x0000000000603ca4
mov    <rel> qword ptr [0x0000000000819f68], 0x00000005
mov    r14, <rel> qword ptr [0x0000000000819f68]
mov    r13d, 0x0061a760
mov    <rel> qword ptr [0x0000000000819f68], 0x00000000
mov    r12d, 0x0002000e
mov    edi, r12d
mov    qword ptr [rsp+0x40], r14
call   0x0000000000601e80
mov    rcx, qword ptr [rsp+0x20]
xor    r9d, r9d
xor    r8d, r8d
mov    edx, 0x000000a1
mov    rsi, r13
mov    rdi, rax
call   0x000000000060c5e0
cmp    rax, 0x000000a0
jnbe   0x0000000000603b76
mov    rax, qword ptr [rsp+0x40]
cmp    <rel> qword ptr [0x0000000000819f68], rax
cmovnb rax, <rel> qword ptr [0x0000000000819f68]
add    r12d, 0x01
add    r13, 0x000000a1
cmp    r12d, 0x0002001a
mov    <rel> qword ptr [0x0000000000819f68], rax
jnz    0x000000000060310b
cmp    rax, r14
jb     0x00000000006030ed
test   rax, rax
jnz    0x0000000000602b21
jmp    0x0000000000603b81
test   r13l, r13l
jnz    0x0000000000602b14
cmp    <rel> dword ptr [0x000000000081a970], 0x00
jz     0x000000000060302d
mov    <rel> dword ptr [0x000000000081a968], 0x00000004
jmp    0x0000000000602b14
mov    edi, 0x004138e1
call   0x00000000006019e0
mov    edx, 0x0061b138
mov    r15, rax
mov    esi, 0x0061b140
mov    rdi, rax
call   0x000000000060c030
test   r15, r15
jz     0x0000000000603d51
mov    eax, <rel> dword ptr [0x000000000081a960]
mov    <rel> dword ptr [0x000000000081a954], eax
mov    rax, <rel> qword ptr [0x000000000081a958]
mov    <rel> qword ptr [0x0000000000819d80], rax
test   r14l, r14l
jz     0x0000000000602a4c
mov    <rel> dword ptr [0x000000000081a960], 0x00000000
mov    <rel> qword ptr [0x000000000081a958], 0x00000400
jmp    0x0000000000602a4c
xor    esi, esi
mov    r8d, 0x004049d0
mov    ecx, 0x00404990
mov    edx, 0x00404980
mov    edi, 0x0000001e
call   0x000000000060ac20
test   rax, rax
mov    <rel> qword ptr [0x000000000081a9e8], rax
jz     0x0000000000603bdb
mov    r8d, 0x004021f0
mov    ecx, 0x00402640
xor    edx, edx
xor    esi, esi
mov    edi, 0x0061af00
call   0x0000000000601c10
jmp    0x0000000000602b4f
cmp    <rel> byte ptr [0x000000000081a92d], 0x00
mov    eax, 0x00000002
jnz    0x000000000060327f
cmp    <rel> dword ptr [0x000000000081a94c], 0x03
jz     0x000000000060327f
cmp    <rel> dword ptr [0x000000000081a970], 0x01
sbb    eax, eax
and    eax, 0xfe
add    eax, 0x04
mov    <rel> dword ptr [0x000000000081a930], eax
jmp    0x0000000000602b42
mov    edi, 0x00000001
call   0x0000000000601aa0
test   eax, eax
jz     0x0000000000602898
mov    <rel> byte ptr [0x000000000081a949], 0x01
mov    <rel> qword ptr [0x000000000081a8f8], 0x00000000
jmp    0x0000000000602350
mov    rdi, r12
call   0x000000000060e150
xor    edi, edi
mov    r12, rax
mov    edx, 0x00000005
mov    esi, 0x00415b48
call   0x0000000000601b80
mov    rcx, r12
mov    rdx, rax
xor    esi, esi
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
jmp    0x00000000006022d1
mov    edi, 0x0041397f
call   0x00000000006019e0
test   rax, rax
mov    qword ptr [rsp+0x38], rax
jz     0x0000000000603302
cmp    byte ptr [rax], 0x00
jnz    0x0000000000603d1a
cmp    <rel> byte ptr [0x000000000081a949], 0x00
jz     0x0000000000602b35
mov    edi, 0x0000000d
call   0x00000000006044f0
test   al, al
jnz    0x000000000060334b
mov    edi, 0x0000000e
call   0x00000000006044f0
test   al, al
jz     0x0000000000603334
cmp    <rel> byte ptr [0x000000000081a9b8], 0x00
jnz    0x000000000060334b
mov    edi, 0x0000000c
call   0x00000000006044f0
test   al, al
jz     0x0000000000603352
cmp    <rel> dword ptr [0x000000000081a970], 0x00
jnz    0x0000000000603352
mov    <rel> byte ptr [0x000000000081a935], 0x01
mov    edi, 0x00000001
call   0x0000000000601e20
test   eax, eax
js     0x0000000000602b35
mov    edi, 0x0061b040
xor    r13d, r13d
call   0x0000000000601dc0
mov    r14d, dword ptr [r13+0x00412cc0]
mov    rdx, qword ptr [rsp+0x20]
xor    esi, esi
mov    edi, r14d
call   0x0000000000601ab0
cmp    qword ptr [rsp+0x40], 0x01
jz     0x000000000060339c
mov    esi, r14d
mov    edi, 0x0061b040
call   0x0000000000602070
add    r13, 0x04
cmp    r13, 0x30
jnz    0x0000000000603371
lea    rdi, [rsp+0x48]
mov    esi, 0x0061b040
mov    ecx, 0x00000020
rep movsd
mov    dword ptr [rsp+0xc8], 0x10000000
xor    r13l, r13l
mov    r14d, 0x004049b0
mov    r15d, dword ptr [r13+0x00412cc0]
mov    edi, 0x0061b040
mov    esi, r15d
call   0x0000000000602000
test   eax, eax
jz     0x0000000000603404
mov    rsi, qword ptr [rsp+0x20]
cmp    r15d, 0x14
mov    eax, 0x004057f0
cmovnz rax, r14
mov    edi, r15d
xor    edx, edx
mov    qword ptr [rsp+0x40], rax
call   0x0000000000601ab0
add    r13, 0x04
cmp    r13, 0x30
jnz    0x00000000006033cb
jmp    0x0000000000602b35
mov    esi, 0x00413807
jmp    0x0000000000602646
mov    rdi, <rel> qword ptr [0x0000000000819e60]
call   0x000000000060e150
xor    edi, edi
mov    r15, rax
mov    edx, 0x00000005
mov    esi, 0x00413867
call   0x0000000000601b80
mov    rcx, r15
mov    rdx, rax
xor    esi, esi
mov    edi, 0x00000002
xor    eax, eax
call   0x0000000000601f90
jmp    0x00000000006024c5
mov    edi, 0x00000001
call   0x0000000000601aa0
cmp    eax, 0x01
sbb    eax, eax
add    eax, 0x02
mov    <rel> dword ptr [0x000000000081a970], eax
jmp    0x000000000060241b
nop    dword ptr [rax+rax+0x00]
test   eax, eax
jz     0x0000000000602f90
xor    eax, eax
cmp    byte ptr [rbp+0x14], 0x2e
setz   al
cmp    byte ptr [rbp+rax+0x14], 0x00
jz     0x0000000000602f90
nop    dword ptr [rax+0x00]
mov    r15, <rel> qword ptr [0x000000000081a920]
test   r15, r15
jnz    0x00000000006034bd
jmp    0x0000000000603640
nop    dword ptr [rax+0x00000000]
mov    r15, qword ptr [r15+0x08]
test   r15, r15
jz     0x0000000000603640
mov    rdi, qword ptr [r15]
mov    edx, 0x00000004
mov    rsi, rbx
call   0x0000000000601c90
test   eax, eax
jnz    0x00000000006034b0
jmp    0x0000000000602f90
nop    dword ptr [cs:rax+rax+0x00000000]
mov    edx, dword ptr [r12]
test   edx, edx
jz     0x0000000000603513
xor    edi, edi
mov    edx, 0x00000005
mov    esi, 0x004139b1
call   0x0000000000601b80
mov    edi, dword ptr [rsp+0x10]
mov    rdx, r14
mov    rsi, rax
call   0x0000000000605030
cmp    dword ptr [r12], 0x4b
jz     0x0000000000602f90
mov    rdi, r13
call   0x0000000000601d00
test   eax, eax
jnz    0x00000000006036dc
call   0x00000000006046a0
cmp    <rel> byte ptr [0x000000000081a92e], 0x00
jnz    0x00000000006036ca
mov    eax, <rel> dword ptr [0x000000000081a970]
test   eax, eax
jz     0x000000000060354c
cmp    <rel> byte ptr [0x000000000081a964], 0x00
jz     0x0000000000603620
cmp    <rel> byte ptr [0x000000000081a950], 0x00
jnz    0x00000000006037ad
mov    edx, 0x00000005
xor    edi, edi
mov    esi, 0x004139db
call   0x0000000000601b80
mov    rsi, <rel> qword ptr [0x0000000000819e30]
mov    rbx, rax
mov    rdi, rax
call   0x0000000000601d40
mov    rdi, rbx
call   0x0000000000601ba0
mov    rdi, <rel> qword ptr [0x0000000000819e30]
add    <rel> qword ptr [0x000000000081a838], rax
mov    rax, qword ptr [rdi+0x28]
cmp    rax, qword ptr [rdi+0x30]
jnb    0x0000000000603d0b
lea    rdx, [rax+0x01]
mov    qword ptr [rdi+0x28], rdx
mov    byte ptr [rax], 0x20
mov    r8, <rel> qword ptr [0x000000000081a958]
mov    edx, <rel> dword ptr [0x000000000081a960]
lea    rsi, [rsp+0xe0]
mov    rdi, qword ptr [rsp+0x08]
mov    ecx, 0x00000200
add    <rel> qword ptr [0x000000000081a838], 0x01
call   0x000000000060b590
mov    rsi, <rel> qword ptr [0x0000000000819e30]
mov    rbx, rax
mov    rdi, rax
call   0x0000000000601d40
mov    rdi, rbx
call   0x0000000000601ba0
mov    rdi, <rel> qword ptr [0x0000000000819e30]
add    <rel> qword ptr [0x000000000081a838], rax
mov    rax, qword ptr [rdi+0x28]
cmp    rax, qword ptr [rdi+0x30]
jnb    0x0000000000603cfc
lea    rdx, [rax+0x01]
mov    qword ptr [rdi+0x28], rdx
mov    byte ptr [rax], 0x0a
add    <rel> qword ptr [0x000000000081a838], 0x01
cmp    <rel> qword ptr [0x000000000081a9d0], 0x00
jz     0x0000000000602cc1
call   0x0000000000607210
jmp    0x0000000000602cc1
nop    dword ptr [rax+rax+0x00000000]
movzx  eax, byte ptr [rbp+0x12]
xor    esi, esi
sub    eax, 0x01
cmp    al, 0x0d
jnbe   0x0000000000603657
movzx  eax, al
mov    esi, dword ptr [rax*4+0x00412c00]
xor    edx, edx
mov    rcx, r14
mov    rdi, rbx
call   0x00000000006076c0
add    qword ptr [rsp+0x08], rax
cmp    <rel> dword ptr [0x000000000081a970], 0x01
jnz    0x0000000000602f90
cmp    <rel> dword ptr [0x000000000081a968], 0xff
jnz    0x0000000000602f90
cmp    <rel> byte ptr [0x000000000081a964], 0x00
jnz    0x0000000000602f90
cmp    <rel> byte ptr [0x000000000081a92e], 0x00
jnz    0x0000000000602f90
call   0x00000000006046a0
call   0x0000000000607210
call   0x00000000006045f0
nop    dword ptr [rax+0x00]
jmp    0x0000000000602f90
mov    rsi, r14
mov    edi, 0x00000001
call   0x0000000000601e30
shr    eax, 0x1f
jmp    0x0000000000602d8d
movzx  esi, byte ptr [rsp+0x2f]
mov    rdi, r14
call   0x00000000006048b0
jmp    0x0000000000603535
xor    edi, edi
mov    edx, 0x00000005
mov    esi, 0x004139c6
call   0x0000000000601b80
movzx  edi, byte ptr [rsp+0x2f]
mov    rdx, r14
mov    rsi, rax
call   0x0000000000605030
jmp    0x0000000000603523
mov    rdi, rbp
call   0x0000000000601a10
mov    rdi, r14
call   0x000000000060e2d0
mov    edx, 0x00000005
mov    rbx, rax
mov    esi, 0x00415d10
xor    edi, edi
call   0x0000000000601b80
mov    rcx, rbx
mov    rdx, rax
xor    esi, esi
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
mov    rdi, r13
call   0x0000000000601d00
mov    <rel> dword ptr [0x000000000081a850], 0x00000002
jmp    0x0000000000602cc1
mov    rcx, <rel> qword ptr [0x0000000000819e30]
mov    edx, 0x00000002
mov    esi, 0x00000001
mov    edi, 0x00413771
call   0x0000000000601ee0
add    <rel> qword ptr [0x000000000081a838], 0x02
cmp    <rel> byte ptr [0x000000000081a950], 0x00
jz     0x0000000000602e7a
mov    rax, <rel> qword ptr [0x000000000081a798]
lea    rdx, [rax+0x08]
cmp    <rel> qword ptr [0x000000000081a7a0], rdx
jb     0x000000000060389f
mov    rdx, <rel> qword ptr [0x000000000081a838]
mov    qword ptr [rax], rdx
add    <rel> qword ptr [0x000000000081a798], 0x08
jmp    0x0000000000602e7a
mov    rcx, <rel> qword ptr [0x0000000000819e30]
mov    edx, 0x00000002
mov    esi, 0x00000001
mov    edi, 0x00413771
call   0x0000000000601ee0
add    <rel> qword ptr [0x000000000081a838], 0x02
jmp    0x0000000000603559
mov    rax, qword ptr [rsp+0x18]
mov    r14, qword ptr [rax]
jmp    0x0000000000602d28
mov    rax, <rel> qword ptr [0x000000000081a738]
mov    rdx, rax
sub    rdx, <rel> qword ptr [0x000000000081a730]
cmp    edx, 0x0f
jbe    0x0000000000603ce3
mov    rdx, <rel> qword ptr [0x000000000081a740]
sub    rdx, rax
cmp    rdx, 0xf0
jnl    0x0000000000603822
mov    esi, 0xfffffff0
mov    edi, 0x0061af00
call   0x0000000000601f40
mov    rax, <rel> qword ptr [0x000000000081a738]
lea    rdx, [rax-0x10]
mov    rsi, qword ptr [rsp+0x20]
mov    rdi, <rel> qword ptr [0x000000000081a9e8]
mov    <rel> qword ptr [0x000000000081a738], rdx
mov    rdx, qword ptr [rax-0x10]
mov    rax, qword ptr [rax-0x08]
mov    qword ptr [rsp+0x40], rdx
mov    qword ptr [rsp+0x48], rax
call   0x000000000060b3b0
test   rax, rax
jz     0x0000000000603aa7
mov    rdi, rax
call   0x0000000000601a10
mov    rbx, qword ptr [rsp+0x18]
mov    rdi, qword ptr [rbx]
call   0x0000000000601a10
mov    rdi, qword ptr [rbx+0x08]
call   0x0000000000601a10
mov    rdi, rbx
call   0x0000000000601a10
jmp    0x0000000000602ce6
mov    esi, 0x00000010
mov    edi, 0x0061af00
call   0x0000000000601f40
mov    rax, <rel> qword ptr [0x000000000081a738]
jmp    0x0000000000602e03
mov    esi, 0x00000008
mov    edi, 0x0061af60
call   0x0000000000601f40
mov    rax, <rel> qword ptr [0x000000000081a798]
jmp    0x0000000000603796
mov    esi, 0x00000008
mov    edi, 0x0061af60
call   0x0000000000601f40
mov    rax, <rel> qword ptr [0x000000000081a798]
jmp    0x0000000000602ec1
cmp    <rel> byte ptr [0x000000000081a949], 0x00
jz     0x0000000000603972
cmp    <rel> byte ptr [0x000000000081a948], 0x00
jz     0x000000000060390d
cmp    <rel> qword ptr [0x0000000000819c00], 0x02
jz     0x0000000000603aeb
mov    edi, 0x0061a3e0
call   0x0000000000605c60
mov    edi, 0x0061a3f0
call   0x0000000000605c60
mov    rdi, <rel> qword ptr [0x0000000000819e30]
mov    ebx, 0x00412cc0
call   0x0000000000602040
jmp    0x000000000060392d
add    rbx, 0x04
cmp    rbx, 0x00412cf0
jz     0x000000000060394a
mov    ebp, dword ptr [rbx]
mov    edi, 0x0061b040
mov    esi, ebp
call   0x0000000000602000
test   eax, eax
jz     0x0000000000603920
xor    esi, esi
mov    edi, ebp
call   0x0000000000601d80
jmp    0x0000000000603920
mov    ebx, <rel> dword ptr [0x000000000081a854]
test   ebx, ebx
jz     0x0000000000603963
mov    edi, 0x00000013
call   0x0000000000601a00
sub    ebx, 0x01
jnz    0x0000000000603954
mov    edi, <rel> dword ptr [0x000000000081a858]
test   edi, edi
jz     0x0000000000603972
call   0x0000000000601a00
cmp    <rel> byte ptr [0x000000000081a950], 0x00
jnz    0x0000000000603b2c
mov    rbx, <rel> qword ptr [0x000000000081a9e8]
test   rbx, rbx
jz     0x0000000000603a15
mov    rdi, rbx
call   0x000000000060a7d0
test   rax, rax
jz     0x0000000000603a0d
mov    ecx, 0x00412ca7
mov    edx, 0x000005dc
mov    esi, 0x00413736
mov    edi, 0x00415d68
call   0x0000000000601c70
xor    edi, edi
mov    edx, 0x00000005
mov    esi, 0x00413998
call   0x0000000000601b80
movzx  edi, byte ptr [rsp+0x2f]
mov    rdx, r14
mov    rsi, rax
call   0x0000000000605030
jmp    0x0000000000602cc1
mov    rdi, r12
call   0x000000000060e150
xor    edi, edi
mov    r12, rax
mov    edx, 0x00000005
mov    esi, 0x00415b88
call   0x0000000000601b80
mov    rcx, r12
mov    rdx, rax
xor    esi, esi
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
jmp    0x0000000000602342
mov    rdi, rbx
call   0x000000000060ae60
mov    edi, <rel> dword ptr [0x000000000081a850]
call   0x0000000000602010
call   0x00000000006046a0
cmp    <rel> byte ptr [0x000000000081a92d], 0x00
jz     0x0000000000603c74
cmp    <rel> qword ptr [0x000000000081a9d0], 0x00
jz     0x0000000000602c7b
call   0x0000000000607210
cmp    <rel> qword ptr [0x000000000081a9b0], 0x00
jz     0x0000000000603cd5
mov    rdi, <rel> qword ptr [0x0000000000819e30]
mov    rax, qword ptr [rdi+0x28]
cmp    rax, qword ptr [rdi+0x30]
jnb    0x0000000000603cc3
lea    rdx, [rax+0x01]
mov    qword ptr [rdi+0x28], rdx
mov    byte ptr [rax], 0x0a
mov    rax, <rel> qword ptr [0x000000000081a9b0]
add    <rel> qword ptr [0x000000000081a838], 0x01
mov    qword ptr [rsp+0x18], rax
jmp    0x0000000000602cf2
mov    rdi, <rel> qword ptr [0x000000000081a908]
mov    edx, 0x00000001
mov    esi, 0x00000020
call   0x000000000060de70
jmp    0x0000000000602a88
mov    ecx, 0x00412ca7
mov    edx, 0x0000059d
mov    esi, 0x00413736
mov    edi, 0x00413992
call   0x0000000000601c70
cmp    <rel> byte ptr [0x000000000081a92d], 0x00
jz     0x0000000000603bc5
mov    ecx, 0x00416919
mov    edx, 0x00000001
mov    esi, 0x00000003
mov    edi, 0x00413990
call   0x00000000006076c0
jmp    0x0000000000602c6d
mov    rdi, <rel> qword ptr [0x0000000000819c08]
mov    edx, 0x00000002
mov    esi, 0x004139e1
call   0x0000000000601d20
test   eax, eax
jnz    0x00000000006038f9
cmp    <rel> qword ptr [0x0000000000819c10], 0x01
jnz    0x00000000006038f9
mov    rax, <rel> qword ptr [0x0000000000819c18]
cmp    byte ptr [rax], 0x6d
jnz    0x00000000006038f9
jmp    0x000000000060390d
mov    esi, 0x0061afc0
mov    edi, 0x004139e4
call   0x0000000000604e50
mov    esi, 0x0061af60
mov    edi, 0x004139ee
call   0x0000000000604e50
mov    rdi, <rel> qword ptr [0x000000000081a908]
call   0x000000000060de50
mov    eax, eax
mov    esi, 0x00415d40
mov    edi, 0x00000001
mov    rdx, qword ptr [rax*8+0x00416480]
xor    eax, eax
call   0x0000000000601f50
jmp    0x000000000060397f
mov    <rel> qword ptr [0x0000000000819f68], 0x00000000
xor    edi, edi
mov    edx, 0x00000005
mov    esi, 0x00415c30
call   0x0000000000601b80
xor    esi, esi
mov    rdx, rax
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
jmp    0x0000000000602b21
test   rax, rax
jz     0x0000000000602cf2
cmp    qword ptr [rax+0x18], 0x00
jnz    0x0000000000602cf2
mov    <rel> byte ptr [0x000000000081a8f0], 0x00
jmp    0x0000000000602cf2
mov    edx, 0x00000001
xor    esi, esi
mov    edi, 0x00413990
call   0x0000000000604540
jmp    0x0000000000602c6d
call   0x0000000000610670
mov    r8, <rel> qword ptr [0x0000000000819e60]
mov    esi, dword ptr [rsp+0x38]
mov    ecx, 0x00413080
xor    edx, edx
mov    edi, eax
call   0x0000000000610af0
add    r12, 0x01
mov    esi, 0x0000000a
mov    rdi, r12
call   0x0000000000601bf0
test   rax, rax
mov    r14, rax
jz     0x0000000000603c6f
lea    r13, [rax+0x01]
mov    esi, 0x0000000a
mov    rdi, r13
call   0x0000000000601bf0
test   rax, rax
jz     0x0000000000603c58
mov    rdi, r12
call   0x000000000060e430
mov    edx, 0x00000005
mov    r15, rax
mov    esi, 0x00413900
xor    edi, edi
call   0x0000000000601b80
mov    rcx, r15
mov    rdx, rax
xor    esi, esi
mov    edi, 0x00000002
xor    eax, eax
call   0x0000000000601f90
mov    byte ptr [r14], 0x00
mov    <rel> qword ptr [0x0000000000819bf0], r12
mov    <rel> qword ptr [0x0000000000819bf8], r13
jmp    0x00000000006030c8
mov    r13, r12
jmp    0x0000000000603c5c
mov    esi, 0x00000001
xor    edi, edi
call   0x00000000006048b0
jmp    0x0000000000603a32
test   rax, rax
jnz    0x00000000006030c8
mov    <rel> qword ptr [0x0000000000819bf8], 0x0041394e
mov    <rel> qword ptr [0x0000000000819bf0], 0x0041394e
mov    rdi, <rel> qword ptr [0x0000000000819bf8]
mov    esi, 0x00413766
call   0x0000000000602080
test   rax, rax
jnz    0x00000000006030e2
jmp    0x0000000000602b21
mov    esi, 0x0000000a
call   0x0000000000601c20
nop    dword ptr [rax]
jmp    0x0000000000603a73
mov    qword ptr [rsp+0x18], 0x00000000
jmp    0x0000000000602cf2
mov    ecx, 0x00412c38
mov    edx, 0x000003d5
mov    esi, 0x00413736
mov    edi, 0x00415c58
call   0x0000000000601c70
mov    esi, 0x0000000a
call   0x0000000000601c20
jmp    0x0000000000603618
mov    esi, 0x00000020
call   0x0000000000601c20
jmp    0x00000000006035ab
mov    rdi, rax
mov    word ptr [rsp+0x30], 0x3f3f
mov    byte ptr [rsp+0x32], 0x00
xor    r13d, r13d
call   0x0000000000610650
xor    edx, edx
mov    <rel> qword ptr [0x000000000081a938], rax
mov    qword ptr [rsp+0x40], rax
cmp    edx, 0x05
jnbe   0x00000000006021a5
mov    eax, edx
jmp    qword ptr [rax*8+0x00412bc8]
mov    edi, 0x004138e4
call   0x00000000006019e0
test   rax, rax
jnz    0x00000000006031d2
jmp    0x00000000006031ec
mov    <rel> qword ptr [0x0000000000819bf8], 0x00413966
mov    <rel> qword ptr [0x0000000000819bf0], 0x00413966
jmp    0x0000000000603ca4
mov    rdx, rax
mov    rsi, r12
mov    edi, 0x0041391d
call   0x00000000006097a0
mov    rbx, <rel> qword ptr [0x0000000000819e70]
mov    edx, 0x00000005
mov    esi, 0x00413928
xor    edi, edi
call   0x0000000000601b80
mov    rsi, rbx
mov    rdi, rax
mov    ebx, 0x00413700
call   0x0000000000601d40
mov    rcx, qword ptr [rbx]
test   rcx, rcx
jz     0x00000000006029f6
mov    rdi, <rel> qword ptr [0x0000000000819e70]
mov    edx, 0x0041393e
mov    esi, 0x00000001
xor    eax, eax
add    rbx, 0x08
call   0x0000000000602030
jmp    0x0000000000603dbc
mov    edi, 0x004138f5
call   0x00000000006019e0
mov    r12, rax
test   rax, rax
mov    eax, 0x00413827
cmovz  r12, rax
jmp    0x0000000000603036
mov    esi, 0x0000000a
call   0x0000000000601c20
jmp    0x0000000000602e5e
mov    <rel> qword ptr [0x0000000000819bf0], 0x00413975
mov    <rel> qword ptr [0x0000000000819bf8], 0x00413969
jmp    0x00000000006030c8
mov    rsi, <rel> qword ptr [0x0000000000819bf0]
mov    edx, 0x00000002
xor    edi, edi
call   0x0000000000601b80
mov    rsi, <rel> qword ptr [0x0000000000819bf8]
mov    edx, 0x00000002
xor    edi, edi
mov    <rel> qword ptr [0x0000000000819bf0], rax
call   0x0000000000601b80
mov    <rel> qword ptr [0x0000000000819bf8], rax
jmp    0x00000000006030c8
mov    edx, 0x00000005
mov    esi, 0x00415d98
xor    edi, edi
call   0x0000000000601b80
xor    esi, esi
mov    rdx, rax
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
mov    rdi, <rel> qword ptr [0x000000000081a938]
call   0x0000000000601a10
mov    rdi, <rel> qword ptr [0x000000000081a940]
test   rdi, rdi
jz     0x0000000000603fa7
mov    r13, qword ptr [rdi+0x20]
call   0x0000000000601a10
mov    rdi, r13
jmp    0x0000000000603e99
mov    rax, qword ptr [rsp+0x38]
lea    rdx, [rax+0x01]
mov    qword ptr [rsp+0x38], rdx
cmp    byte ptr [rax], 0x3d
mov    edx, 0x00000005
jnz    0x0000000000603d48
mov    rax, qword ptr [rsp+0x40]
mov    rdi, qword ptr [rsp+0x20]
lea    rcx, [r13+0x10]
lea    rsi, [rsp+0x38]
xor    dl, dl
mov    qword ptr [r13+0x18], rax
call   0x0000000000604200
cmp    al, 0x01
sbb    edx, edx
and    edx, 0x05
jmp    0x0000000000603d3f
mov    rax, qword ptr [rsp+0x38]
mov    edx, 0x00000005
cmp    byte ptr [rax], 0x00
jz     0x0000000000603d48
lea    rdx, [rax+0x01]
mov    qword ptr [rsp+0x38], rdx
movzx  eax, byte ptr [rax]
mov    edx, 0x00000002
mov    byte ptr [rsp+0x31], al
jmp    0x0000000000603d48
mov    rax, qword ptr [rsp+0x38]
movzx  ecx, byte ptr [rax]
cmp    cl, 0x2a
jz     0x0000000000603ff4
cmp    cl, 0x3a
jz     0x0000000000603fe6
test   cl, cl
jz     0x0000000000603fae
lea    rdx, [rax+0x01]
mov    qword ptr [rsp+0x38], rdx
movzx  eax, byte ptr [rax]
mov    edx, 0x00000001
mov    byte ptr [rsp+0x30], al
jmp    0x0000000000603d48
mov    rax, qword ptr [rsp+0x38]
xor    r15d, r15d
lea    rdx, [rax+0x01]
mov    qword ptr [rsp+0x38], rdx
cmp    byte ptr [rax], 0x3d
mov    edx, 0x00000005
jnz    0x0000000000603d48
jmp    0x0000000000603f92
lea    rdi, [rsp+0x30]
add    r15, 0x01
call   0x0000000000601d70
test   eax, eax
jz     0x0000000000604048
mov    rsi, qword ptr [r15*8+0x004135e0]
movsxd r14, r15d
test   rsi, rsi
jnz    0x0000000000603f7c
jmp    0x0000000000604077
mov    <rel> byte ptr [0x000000000081a949], 0x00
cmp    <rel> qword ptr [0x0000000000819c70], 0x06
jnz    0x0000000000603302
mov    rdi, <rel> qword ptr [0x0000000000819c78]
mov    edx, 0x00000006
mov    esi, 0x00413989
call   0x0000000000601a60
test   eax, eax
jnz    0x0000000000603302
mov    <rel> byte ptr [0x000000000081a9b8], 0x01
jmp    0x0000000000603302
add    rax, 0x01
mov    qword ptr [rsp+0x38], rax
jmp    0x0000000000603d3f
mov    edi, 0x00000028
call   0x0000000000610460
mov    r13, rax
mov    rax, <rel> qword ptr [0x000000000081a940]
mov    rdi, qword ptr [rsp+0x20]
lea    rsi, [rsp+0x38]
mov    edx, 0x00000001
mov    rcx, r13
add    qword ptr [rsp+0x38], 0x01
mov    <rel> qword ptr [0x000000000081a940], r13
mov    qword ptr [r13+0x20], rax
mov    rax, qword ptr [rsp+0x40]
mov    qword ptr [r13+0x08], rax
call   0x0000000000604200
cmp    al, 0x01
sbb    edx, edx
and    edx, 0x02
add    edx, 0x03
jmp    0x0000000000603d3f
shl    r14, 0x04
mov    rax, qword ptr [rsp+0x40]
mov    rdi, qword ptr [rsp+0x20]
lea    rcx, [r14+0x0061a3e0]
lea    rsi, [rsp+0x38]
xor    edx, edx
mov    qword ptr [rcx+0x08], rax
call   0x0000000000604200
xor    edx, edx
test   al, al
jnz    0x0000000000603d48
lea    rdi, [rsp+0x30]
call   0x000000000060e150
mov    edx, 0x00000005
mov    r14, rax
mov    esi, 0x004139fb
xor    edi, edi
call   0x0000000000601b80
mov    rcx, r14
mov    rdx, rax
xor    esi, esi
xor    edi, edi
xor    eax, eax
call   0x0000000000601f90
mov    edx, 0x00000005
jmp    0x0000000000603d48
xor    ebp, ebp
mov    r9, rdx
pop    rsi
mov    rdx, rsp
and    rsp, 0xf0
push   rax
push   rsp
mov    r8, 0x00411ed0
mov    rcx, 0x00411e60
mov    rdi, 0x004028c0
call   0x0000000000601d10
hlt
nop    dword ptr [rax+rax+0x00]
mov    eax, 0x0061a5ff
push   rbp
sub    rax, 0x0061a5f8
cmp    rax, 0x0e
mov    rbp, rsp
jnbe   0x00000000006040f7
pop    rbp
ret
mov    eax, 0x00000000
test   rax, rax
jz     0x00000000006040f5
pop    rbp
mov    edi, 0x0061a5f8
jmp    rax
nop    dword ptr [rax+0x00000000]
mov    eax, 0x0061a5f8
push   rbp
sub    rax, 0x0061a5f8
sar    rax, 0x03
mov    rbp, rsp
mov    rdx, rax
shr    rdx, 0x3f
add    rax, rdx
sar    rax, 0x01
jnz    0x0000000000604134
pop    rbp
ret
mov    edx, 0x00000000
test   rdx, rdx
jz     0x0000000000604132
pop    rbp
mov    rsi, rax
mov    edi, 0x0061a5f8
jmp    rdx
nop    dword ptr [rax+0x00000000]
cmp    <rel> byte ptr [0x0000000000819e78], 0x00
jnz    0x000000000060416a
push   rbp
mov    rbp, rsp
call   0x00000000006040e0
pop    rbp
mov    <rel> byte ptr [0x0000000000819e78], 0x01
ret
nop    dword ptr [rax+0x00]
cmp    <rel> qword ptr [0x0000000000819620], 0x00
jz     0x0000000000604198
mov    eax, 0x00000000
test   rax, rax
jz     0x0000000000604198
push   rbp
mov    edi, 0x00619e00
mov    rbp, rsp
call   rax
pop    rbp
jmp    0x0000000000604110
nop    dword ptr [rax]
jmp    0x0000000000604110
nop    dword ptr [rax]
mov    rax, qword ptr [rdi]
xor    edx, edx
div    rdx, rax, rsi
mov    rax, rdx
ret
nop    dword ptr [rax+0x00]
xor    eax, eax
mov    rdx, qword ptr [rsi]
cmp    qword ptr [rdi], rdx
jz     0x00000000006041c0
ret
nop    dword ptr [rax+0x00]
mov    rax, qword ptr [rsi+0x08]
cmp    qword ptr [rdi+0x08], rax
setz   al
ret
nop    dword ptr [rax+0x00]
mov    eax, <rel> dword ptr [0x000000000081a858]
test   eax, eax
jnz    0x00000000006041e0
mov    <rel> dword ptr [0x000000000081a858], edi
ret
nop    dword ptr [cs:rax+rax+0x00000000]
jmp    0x0000000000601a10
nop    dword ptr [cs:rax+rax+0x00000000]
push   r14
mov    r8, qword ptr [rsi]
xor    eax, eax
mov    r9, qword ptr [rdi]
xor    r10d, r10d
xor    r11d, r11d
push   r12
mov    r12, 0x007e000000000000
push   rbp
mov    rbp, rcx
push   rbx
mov    ebx, 0x00000001
cmp    eax, 0x02
jz     0x0000000000604279
jbe    0x0000000000604340
cmp    eax, 0x03
jz     0x0000000000604300
cmp    eax, 0x04
nop    dword ptr [rax]
jnz    0x0000000000604290
movzx  eax, byte ptr [r8]
lea    ecx, [rax-0x40]
cmp    cl, 0x3e
jbe    0x00000000006042c0
cmp    al, 0x3f
jz     0x00000000006044d0
xor    eax, eax
mov    qword ptr [rdi], r9
mov    qword ptr [rsi], r8
mov    qword ptr [rbp+0x00], r10
pop    rbx
pop    rbp
pop    r12
pop    r14
ret
nop    dword ptr [rax+0x00000000]
lea    r11d, [rax+r11*8-0x30]
add    r8, 0x01
movzx  eax, byte ptr [r8]
lea    ecx, [rax-0x30]
cmp    cl, 0x07
jbe    0x0000000000604270
mov    byte ptr [r9], r11l
add    r10, 0x01
add    r9, 0x01
movzx  eax, byte ptr [r8]
cmp    al, 0x3d
jz     0x00000000006042da
jle    0x0000000000604360
cmp    al, 0x5c
jz     0x00000000006044c0
cmp    al, 0x5e
jnz    0x0000000000604380
add    r8, 0x01
movzx  eax, byte ptr [r8]
lea    ecx, [rax-0x40]
cmp    cl, 0x3e
jnbe   0x000000000060424e
nop
and    eax, 0x1f
add    r8, 0x01
add    r10, 0x01
mov    byte ptr [r9], al
movzx  eax, byte ptr [r8]
add    r9, 0x01
cmp    al, 0x3d
jnz    0x0000000000604298
test   dl, dl
jz     0x0000000000604380
mov    eax, 0x00000001
jmp    0x0000000000604258
nop    dword ptr [rax+0x00]
shl    r11d, 0x04
add    r8, 0x01
lea    r11d, [rax+r11-0x57]
nop    dword ptr [rax]
movzx  eax, byte ptr [r8]
lea    ecx, [rax-0x30]
cmp    cl, 0x36
jnbe   0x0000000000604285
mov    r14, rbx
shl    r14, cl
test   r14d, 0x007e0000
jnz    0x0000000000604398
test   r14, r12
jnz    0x00000000006042f0
test   r14d, 0x000003ff
jz     0x0000000000604285
shl    r11d, 0x04
add    r8, 0x01
lea    r11d, [rax+r11-0x30]
jmp    0x0000000000604300
cmp    eax, 0x01
jnz    0x0000000000604290
movzx  eax, byte ptr [r8]
cmp    al, 0x78
jnbe   0x00000000006044b0
movzx  ecx, al
jmp    qword ptr [rcx*8+0x00411f40]
nop
test   al, al
jz     0x0000000000604368
cmp    al, 0x3a
jnz    0x0000000000604380
mov    eax, 0x00000005
cmp    eax, 0x06
setnz  al
jmp    0x0000000000604258
nop    dword ptr [rax+rax+0x00000000]
mov    byte ptr [r9], al
add    r8, 0x01
add    r10, 0x01
add    r9, 0x01
jmp    0x0000000000604290
nop    dword ptr [rax+0x00]
add    r8, 0x01
shl    r11d, 0x04
lea    r11d, [rax+r11-0x37]
movzx  eax, byte ptr [r8]
lea    ecx, [rax-0x30]
cmp    cl, 0x36
jnbe   0x0000000000604285
jmp    0x0000000000604310
nop    dword ptr [rax+rax+0x00]
mov    eax, 0x00000003
xor    r11d, r11d
nop    dword ptr [rax+rax+0x00000000]
add    r8, 0x01
cmp    eax, 0x04
jbe    0x0000000000604226
jmp    0x000000000060436d
nop
mov    r11d, 0x0000001b
nop    dword ptr [cs:rax+rax+0x00000000]
mov    byte ptr [r9], r11l
add    r10, 0x01
add    r9, 0x01
xor    eax, eax
jmp    0x00000000006043d0
nop
mov    r11d, 0x00000020
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00000000]
mov    eax, 0x00000006
jmp    0x00000000006043d0
nop    dword ptr [rax+rax+0x00000000]
lea    r11d, [rax-0x30]
mov    eax, 0x00000002
jmp    0x00000000006043d0
nop    dword ptr [rax+rax+0x00]
mov    r11d, 0x0000007f
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00000000]
mov    r11d, 0x00000007
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00000000]
mov    r11d, 0x00000008
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00000000]
mov    r11d, 0x00000009
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00000000]
mov    r11d, 0x0000000b
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00]
mov    r11d, 0x0000000a
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00]
mov    r11d, 0x0000000d
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00]
mov    r11d, 0x0000000c
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00]
mov    r11d, eax
jmp    0x00000000006043f0
nop    dword ptr [rax+rax+0x00000000]
add    r8, 0x01
jmp    0x0000000000604349
nop    dword ptr [rax+0x00000000]
mov    byte ptr [r9], 0x7f
add    r10, 0x01
add    r9, 0x01
jmp    0x0000000000604290
nop    dword ptr [cs:rax+rax+0x00000000]
mov    edi, edi
xor    eax, eax
shl    rdi, 0x04
mov    rdx, qword ptr [rdi+0x0061a3e0]
mov    rsi, qword ptr [rdi+0x0061a3e8]
test   rdx, rdx
jz     0x000000000060451c
cmp    rdx, 0x01
jz     0x0000000000604530
cmp    rdx, 0x02
mov    eax, 0x00000001
jz     0x0000000000604520
ret
nop
mov    edi, 0x00413733
mov    ecx, 0x00000002
rep cmpsb
setnz  al
ret
cmp    byte ptr [rsi], 0x30
setnz  al
ret
nop    dword ptr [rax+rax+0x00000000]
push   r13
mov    r13d, edx
push   r12
mov    r12, rsi
push   rbp
mov    rbp, rdi
mov    edi, 0x00000020
push   rbx
sub    rsp, 0x08
call   0x0000000000610460
mov    rbx, rax
xor    eax, eax
test   r12, r12
jz     0x000000000060456f
mov    rdi, r12
call   0x0000000000610650
mov    qword ptr [rbx+0x08], rax
xor    eax, eax
test   rbp, rbp
jz     0x0000000000604582
mov    rdi, rbp
call   0x0000000000610650
mov    qword ptr [rbx], rax
mov    rax, <rel> qword ptr [0x000000000081a9b0]
mov    byte ptr [rbx+0x10], r13l
mov    <rel> qword ptr [0x000000000081a9b0], rbx
mov    qword ptr [rbx+0x18], rax
add    rsp, 0x08
pop    rbx
pop    rbp
pop    r12
pop    r13
ret
nop    dword ptr [cs:rax+rax+0x00000000]
push   rbx
mov    rbx, rdi
mov    rdi, qword ptr [rdi]
call   0x0000000000601a10
mov    rdi, qword ptr [rbx+0x08]
call   0x0000000000601a10
mov    rdi, qword ptr [rbx+0xa8]
cmp    rdi, 0x0061a56a
jz     0x00000000006045e0
pop    rbx
jmp    0x0000000000601ff0
nop    dword ptr [rax+rax+0x00]
pop    rbx
ret
nop    dword ptr [cs:rax+rax+0x00000000]
push   rbx
xor    ebx, ebx
cmp    <rel> qword ptr [0x000000000081a9d0], 0x00
jz     0x000000000060461d
nop    dword ptr [rax]
mov    rax, <rel> qword ptr [0x000000000081a9c8]
mov    rdi, qword ptr [rax+rbx*8]
add    rbx, 0x01
call   0x00000000006045b0
cmp    <rel> qword ptr [0x000000000081a9d0], rbx
jnbe   0x0000000000604600
mov    <rel> qword ptr [0x000000000081a9d0], 0x00000000
mov    <rel> byte ptr [0x000000000081a99c], 0x00
mov    <rel> dword ptr [0x000000000081a998], 0x00000000
mov    <rel> dword ptr [0x000000000081a994], 0x00000000
mov    <rel> dword ptr [0x000000000081a990], 0x00000000
mov    <rel> dword ptr [0x000000000081a988], 0x00000000
mov    <rel> dword ptr [0x000000000081a984], 0x00000000
mov    <rel> dword ptr [0x000000000081a980], 0x00000000
mov    <rel> dword ptr [0x000000000081a98c], 0x00000000
mov    <rel> dword ptr [0x000000000081a97c], 0x00000000
mov    <rel> dword ptr [0x000000000081a978], 0x00000000
mov    <rel> dword ptr [0x000000000081a974], 0x00000000
pop    rbx
ret
nop    dword ptr [cs:rax+rax+0x00000000]
push   rbp
push   rbx
sub    rsp, 0x08
mov    rbx, <rel> qword ptr [0x000000000081a9d0]
mov    rax, rbx
mov    rbp, rbx
shr    rax, 0x01
add    rax, rbx
cmp    rax, <rel> qword ptr [0x000000000081a9c0]
jnbe   0x00000000006047b8
test   rbp, rbp
jz     0x00000000006046f3
mov    rax, <rel> qword ptr [0x000000000081a9c8]
mov    rdx, <rel> qword ptr [0x000000000081a9e0]
lea    rcx, [rax+rbp*8]
nop    dword ptr [rax]
mov    qword ptr [rax], rdx
add    rax, 0x08
add    rdx, 0x000000c0
cmp    rax, rcx
jnz    0x00000000006046e0
cmp    <rel> dword ptr [0x000000000081a968], 0xff
jz     0x00000000006047ac
mov    edi, 0x0061a680
call   0x0000000000601d30
test   eax, eax
jz     0x0000000000604760
mov    r8d, <rel> dword ptr [0x000000000081a968]
cmp    r8d, 0x03
jz     0x00000000006047ff
mov    rsi, <rel> qword ptr [0x000000000081a9d0]
mov    rdi, <rel> qword ptr [0x000000000081a9c8]
test   rsi, rsi
jz     0x0000000000604753
mov    rdx, <rel> qword ptr [0x000000000081a9e0]
lea    rcx, [rdi+rsi*8]
mov    rax, rdi
mov    qword ptr [rax], rdx
add    rax, 0x08
add    rdx, 0x000000c0
cmp    rax, rcx
jnz    0x0000000000604740
mov    ecx, r8d
mov    eax, 0x00000001
jmp    0x0000000000604774
nop    dword ptr [rax]
mov    ecx, <rel> dword ptr [0x000000000081a968]
mov    rsi, <rel> qword ptr [0x000000000081a9d0]
mov    rdi, <rel> qword ptr [0x000000000081a9c8]
xor    edx, edx
cmp    ecx, 0x04
cmovz  edx, <rel> dword ptr [0x000000000081a96c]
cwde
movzx  r8d, <rel> byte ptr [0x000000000081a92c]
add    edx, ecx
lea    rdx, [rax+rdx*2]
movzx  eax, <rel> byte ptr [0x000000000081a967]
lea    rax, [rax+rdx*2]
lea    rax, [r8+rax*2]
mov    rdx, qword ptr [rax*8+0x00412d00]
call   0x000000000060ceb0
add    rsp, 0x08
pop    rbx
pop    rbp
ret
nop    dword ptr [rax+rax+0x00]
mov    rdi, <rel> qword ptr [0x000000000081a9c8]
call   0x0000000000601a10
mov    rax, 0x0aaaaaaaaaaaaaaa
cmp    qword ptr [rax], rax
