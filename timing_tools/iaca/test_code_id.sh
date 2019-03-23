#!/usr/bin/env bash

set -e

function template_head() {
    cat <<"EOF"
        .text
        .globl          main
main:
	movl $111, %ebx
	.byte 0x64, 0x67, 0x90

EOF
}
function template_tail() {
    cat <<"EOF"
	movl $222, %ebx
	.byte 0x64, 0x67, 0x90
EOF
}

if [ "$#" == "0" ]; then
    echo "Usage: $0 code_id"
    exit 1
fi

code_id=$1; shift
code=$(echo "SELECT code_att FROM code WHERE code_id=${code_id}" | mysql -N | sed 's/0xf[a-fA-F0-9]+\?\([a-fA-F0-9]\{6\}\)/0xf\1/g')
if [ -z "$code" ]; then
    exit 1
fi

tmpfile=$(mktemp)
(template_head; echo -e $code; template_tail) | as --64 -o $tmpfile -
speed=$(iaca -reduceout $tmpfile | grep 'Block Throughput:' | awk '{print 100 * $3}')
echo $code_id $speed
