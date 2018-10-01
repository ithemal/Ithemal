#!/usr/bin/env bash

set -ex

if [[ -z "${ITHEMAL_HOME}" ]]; then
    echo "ITHEMAL_HOME environment variable must be set!"
    exit 1
fi

if [[ -z "${DYNAMORIO_HOME}" ]]; then
    echo "DYNAMORIO_HOME environment variable must be set!"
    exit 1
fi

mkdir "${ITHEMAL_HOME}/data_collection/build"
cd "${ITHEMAL_HOME}/data_collection/build"
cmake -D -DDynamoRIO_DIR="${DYNAMORIO_HOME}/cmake" ..
make -j"$(nproc --all)"
