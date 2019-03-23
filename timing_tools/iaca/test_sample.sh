#!/usr/bin/env bash

if [ "$#" != "2" ]; then
    echo "Usage: $0 arch num_to_sample"
    exit 1
fi

arch=$1; shift
sample_size=$1; shift
echo "select code_id from times group by code_id order by rand() limit $sample_size;" | mysql -N | xargs -P 4 -n 1 ./test_code_id.sh $arch
