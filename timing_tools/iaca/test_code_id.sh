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

function to_att() {
    ${ITHEMAL_HOME}/data_collection/build/bin/tokenizer $(cat) --att
}

if [ "$#" != "2" ]; then
    echo "Usage: $0 arch code_id"
    exit 1
fi

arch=$1; shift
code_id=$1; shift

code=$(echo "SELECT code_raw FROM code WHERE code_id=${code_id}" | mysql -N | to_att | sed 's/0xf[a-fA-F0-9]+\?\([a-fA-F0-9]\{6\}\)/0xf\1/g' | sed 's/\n/\\n/g')

if [ -z "$code" ]; then
    exit 1
fi

tmpfile=$(mktemp)
(template_head_att; echo -e $code; template_tail_att) | as --64 -o $tmpfile -
speed=$(iaca -arch $arch -reduceout $tmpfile | grep 'Block Throughput:' | awk '{print 100 * $3}')
rm $tmpfile
echo "INSERT INTO time (code_id, arch_id, kind_id, cycle_count) VALUES (${code_id}, 1, 2, ${speed});" | mysql
