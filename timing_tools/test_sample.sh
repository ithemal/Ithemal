#!/usr/bin/env bash

dir=$(dirname -- "$0")

if [[ "$#" -lt "2" ]]; then
    echo "Usage: $0 tool arch [num_to_sample]"
    exit 1
fi

tool=$1; shift
arch=$1; shift
sample_size=${1:-99999999999}; shift

if [[ $tool = "iaca" ]]; then
    kind_id=2
elif [[ $tool = "llvm-cycles" ]]; then
    kind_id=3
elif [[ $tool = "llvm-rthroughput" ]]; then
    kind_id=5
else
    echo "unknown tool $tool"
    exit 1
fi


if [[ $arch = "haswell" ]]; then
    arch_id=1
elif [[ $arch = "skylake" ]]; then
    arch_id=2
elif [[ $arch = "broadwell" ]]; then
    arch_id=3
elif [[ $arch = "nehalem" ]]; then
    arch_id=4
elif [[ $arch = "ivybridge" ]]; then
    arch_id=5
else
    echo "unknown arch $arch"
    exit 1
fi

echo "SELECT DISTINCT(reals.code_id) FROM (SELECT code_id FROM (SELECT code_id, COUNT(1) AS size FROM (SELECT t.time_id, t.code_id FROM time AS t JOIN time_metadata AS tm ON t.time_id=tm.time_id WHERE l1drmisses<=0 AND l1dwmisses<=0 AND l1imisses<=0 AND conswitch<=0) AS good_times GROUP BY good_times.code_id) AS grouped_times WHERE grouped_times.size > 1) as reals LEFT JOIN (SELECT code_id FROM time WHERE kind_id=${kind_id} AND arch_id=${arch_id}) as already_collected ON reals.code_id=already_collected.code_id WHERE already_collected.code_id IS NULL ORDER BY rand() LIMIT ${sample_size};" | \
    mysql -N | \
    xargs -P4 -n 256 "${dir}/test_code_id.py" $arch $tool --insert
