#!/usr/bin/env bash

dir=$(dirname -- "$0")

set -e

function template_head_att() {
    cat <<"EOF"
        .text
        .att_syntax
        .globl          main
main:
        # LLVM-MCA-BEGIN test

EOF
}
function template_tail_att() {
    cat <<"EOF"
        # LLVM-MCA-END test
EOF
}

function to_att() {
    ${ITHEMAL_HOME}/data_collection/disassembler/build/disassemble -att
}


if [ "$#" != "3" ]; then
    echo "Usage: $0 arch kind_id code_id"
    exit 1
fi

arch=$1; shift
kind_id=$1; shift
code_id=$1; shift

code=$(echo "SELECT code_raw FROM code WHERE code_id=${code_id}" | mysql -N | to_att | tr '\t' ' ' | tr '\n' '\t' | sed 's/\t/\\n/g')

if [ -z "$code" ]; then
    exit 1
fi

speed=$((template_head_att; echo -e $code; template_tail_att) | $dir/../llvm-build/bin/llvm-mca -march=x86 -mcpu $arch | grep 'Block RThroughput:' | awk '{print 100 * $3}')
echo "INSERT INTO time (code_id, arch_id, kind_id, cycle_count) VALUES (${code_id}, 1, ${kind_id}, ${speed});" | mysql
