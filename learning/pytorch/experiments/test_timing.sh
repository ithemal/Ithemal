#!/usr/bin/env bash

TRAINERS="$1"; shift
THREADS="$1"; shift

cd "${ITHEMAL_HOME}/learning/pytorch"

bash experiments/train.sh "speed_test_${TRAINERS}_${THREADS}" --no-decay-procs --trainers "${TRAINERS}" --threads "${THREADS}"
