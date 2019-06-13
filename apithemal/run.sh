#!/usr/bin/env bash

mkdir -p "${ITHEMAL_HOME}"/learning/pytorch/saved
aws s3 cp s3://ithemal-experiments/043019_lstm_initial-lr-0.1-decay-1.2-weird-lr/2019-04-30T15:53:04.407209/trained.mdl "${ITHEMAL_HOME}"/learning/pytorch/saved/trained.mdl
aws s3 cp s3://ithemal-predictors/skylake_predictor.dump "${ITHEMAL_HOME}"/learning/pytorch/saved/predictor.dump

pip install --user flask gunicorn
~/.local/bin/gunicorn apithemal:app -w 16 -b 0.0.0.0:5000
