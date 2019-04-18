#!/usr/bin/env bash

set -e

function template_head_att() {
    cat <<"EOF"
        .text
        .att_syntax
        .globl          main
main:
	movl $111, %ebx
	.byte 0x64, 0x67, 0x90

EOF
}
function template_tail_att() {
    cat <<"EOF"
	movl $222, %ebx
	.byte 0x64, 0x67, 0x90
EOF
}

function template_head_hex() {
    echo -n "7f454c4602010100000000000000000001003e000100000000000000000000000000000000000000100100000000000000000000400000000000400004000100bb6f000000646790"
}

function template_tail_hex() {
    echo -n "bbde000000646790"
}

function to_att() {
    ${ITHEMAL_HOME}/data_collection/disassembler/build/disassemble -att
}

function assemble() {
    as -o $1 -
}

if [ "$#" != "3" ]; then
    echo "Usage: $0 arch kind_id code_id"
    exit 1
fi

arch=$1; shift
kind_id=$1; shift
code_id=$1; shift

code_raw=$(echo "SELECT code_raw FROM code WHERE code_id=${code_id}" | mysql -N)

if [ -z "$code_raw" ]; then
    exit 1
fi

full_code=$(template_head_hex; echo -n $code_raw; template_tail_hex)

speed=$(iaca -arch $arch -reduceout <(echo -n ${full_code} | xxd -r -p) | \
            grep 'Block Throughput:' | \
            awk '{print 100 * $3}'
     )
echo "INSERT INTO time (code_id, arch_id, kind_id, cycle_count) VALUES (${code_id}, 1, ${kind_id}, ${speed});" | mysql
