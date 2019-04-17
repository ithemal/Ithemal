#!/usr/bin/env bash

dir=$(dirname -- "$0")

if [ "$#" != "3" ]; then
    echo "Usage: $0 tool arch [num_to_sample]"
    exit 1
fi

tool=$1; shift
arch=$1; shift
sample_size=${1:-99999999999}; shift

echo "SELECT code_id FROM (SELECT code_id, COUNT(1) AS size FROM (SELECT t.time_id, t.code_id FROM time AS t JOIN time_metadata AS tm ON t.time_id=tm.time_id WHERE l1drmisses<=0 AND l1dwmisses<=0 AND l1imisses<=0 AND conswitch<=0) AS ts GROUP BY ts.code_id) AS q WHERE q.size > 1 ORDER BY RAND() LIMIT ${sample_size};" | \
    mysql -N | \
    xargs -P $(nproc) -n 1 timeout 60s "${dir}/${tool}/test_code_id.sh" $arch
