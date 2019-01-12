#!/usr/bin/env bash

set -ex

CHECKPOINT_FILE=$1; shift

cd "${ITHEMAL_HOME}/learning/pytorch"

experiments/download_data.sh

MODEL="${ITHEMAL_HOME}/learning/pytorch/saved/${CHECKPOINT_FILE}"

aws s3 cp "s3://ithemal-checkpoints/checkpoint_models/${CHECKPOINT_FILE}" "${MODEL}"

REPORT_FILE="${ITHEMAL_HOME}/learning/pytorch/saved/${CHECKPOINT_FILE}.report"

python ithemal/run_ithemal.py --mode validate --embmode none --embedfile inputs/embeddings/code_delim.emb --savedatafile saved/time_skylake_1217.data --loadfile "${MODEL}" \
    | tee "${REPORT_FILE}"

aws s3 cp "${REPORT_FILE}" "s3://ithemal-checkpoints/checkpoint_reports/"

LOSS=$(tail -n 1 "${REPORT_FILE}" | cut -d' ' -f6)
