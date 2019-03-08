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

BUILD_DIR="${ITHEMAL_HOME}/data_collection/build"

if [ ! -d "${BUILD_DIR}" ]; then
    mkdir "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"
cmake -DDynamoRIO_DIR="${DYNAMORIO_HOME}/cmake" -DCMAKE_BUILD_TYPE=Debug ..
make -j"$(nproc --all)"
