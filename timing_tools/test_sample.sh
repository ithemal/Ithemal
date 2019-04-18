#!/usr/bin/env bash

dir=$(dirname -- "$0")

if [[ "$#" -lt "3" ]]; then
    echo "Usage: $0 tool arch kind_id [num_to_sample]"
    exit 1
fi

tool=$1; shift
arch=$1; shift
kind_id=$1; shift
sample_size=${1:-99999999999}; shift

echo "SELECT DISTINCT(reals.code_id) FROM (SELECT code_id FROM (SELECT code_id, COUNT(1) AS size FROM (SELECT t.time_id, t.code_id FROM time AS t JOIN time_metadata AS tm ON t.time_id=tm.time_id WHERE l1drmisses<=0 AND l1dwmisses<=0 AND l1imisses<=0 AND conswitch<=0) AS ts GROUP BY ts.code_id) AS q WHERE q.size > 1) as reals LEFT JOIN (SELECT code_id FROM time WHERE kind_id=${kind_id}) as already_collected ON reals.code_id=already_collected.code_id WHERE already_collected.code_id IS NULL ORDER BY rand() LIMIT ${sample_size};" | \
    mysql -N | \
    xargs -P 4 -n 1 timeout 60s "${dir}/${tool}/test_code_id.sh" $arch $kind_id
