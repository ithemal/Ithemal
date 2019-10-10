#!/usr/bin/env bash

mkdir -p "${ITHEMAL_HOME}"/learning/pytorch/saved


pushd "${ITHEMAL_HOME}/learning/pytorch/saved"
wget https://www.dropbox.com/sh/5or5t3vfgz70p8n/AADD0rH-bfoDVepLnpg4JBmka -O "models.zip"
unzip -o models.zip
popd

pip install --user flask gunicorn
~/.local/bin/gunicorn apithemal:app -w 16 -b 0.0.0.0:5000
