#!/usr/bin/env bash

set -ex

if [[ -z "${ITHEMAL_HOME}" ]]; then
    echo "ITHEMAL_HOME environment variable must be set!"
    exit 1
fi

"${ITHEMAL_HOME}"/data_collection/build_dynamorio.sh

cd "${ITHEMAL_HOME}"/common
pip install --user -e .
