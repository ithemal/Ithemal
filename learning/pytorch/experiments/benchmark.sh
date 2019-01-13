#!/usr/bin/env bash

if [ "$#" -lt 3 ]; then
    echo "Requires a parameter for num trainers, num threads, and batch size"
    exit 1
fi

NUM_TRAINERS=$1; shift
NUM_THREADS=$1; shift
BATCH_SIZE=$1; shift

NAME="benchmark_${NUM_TRAINERS}_${NUM_THREADS}_${BATCH_SIZE}"

RUNTIME=$(python "${ITHEMAL_HOME}/learning/pytorch/ithemal/run_ithemal.py" \
               --data "${ITHEMAL_HOME}/learning/pytorch/inputs/data/time_skylake_test.data" \
               benchmark \
               --trainers "${NUM_TRAINERS}" --threads "${NUM_THREADS}" --batch-size "${BATCH_SIZE}" \
               "${@}" \
              | tail -n 1 \
              | cut -d' ' -f6
       )
"${ITHEMAL_HOME}/aws/ping_slack.py" "${NUM_TRAINERS} trainers, ${NUM_THREADS} threads, batch size ${BATCH_SIZE} -- ${RUNTIME} seconds"
