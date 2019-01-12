#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${DIR}"

mkdir -p train_loss_reports
aws s3 sync s3://ithemal-checkpoints/loss_reports train_loss_reports

mkdir -p train_loss_checkpoints

aws s3 sync s3://ithemal-checkpoints/checkpoint_reports/ test_loss_checkpoint_reports
