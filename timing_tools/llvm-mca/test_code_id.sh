#!/usr/bin/env bash

set -e

function template_head() {
    cat <<"EOF"
        .text
        .globl          main
main:
        # LLVM-MCA-BEGIN test

EOF
}
function template_tail() {
    cat <<"EOF"
	# LLVM-MCA-END test
EOF
}

if [ "$#" != "2" ]; then
    echo "Usage: $0 arch code_id"
    exit 1
fi

arch=$1; shift
code_id=$1; shift

code=$(echo "SELECT code_att FROM code WHERE code_id=${code_id}" | mysql -N | sed 's/0xf[a-fA-F0-9]+\?\([a-fA-F0-9]\{6\}\)/0xf\1/g')
if [ -z "$code" ]; then
    exit 1
fi

speed=$((template_head; echo -e $code; template_tail) | ../llvm-build/bin/llvm-mca -mcpu $arch | grep 'Block RThroughput:' | awk '{print 100 * $3}')
echo $code_id $speed
