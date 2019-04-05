#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Requires a batch size parameter"
    exit 1
fi

BATCH_SIZE=$1; shift

bash "${ITHEMAL_HOME}/learning/pytorch/experiments/default_train.sh" "batch_size_${BATCH_SIZE}" --mode train --batch-size "${BATCH_SIZE}" "${@}"
