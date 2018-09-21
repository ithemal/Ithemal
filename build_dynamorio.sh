#!/usr/bin/env bash

set -ex

mkdir -p /home/ithemal/ithemal/data_collection/build
cd /home/ithemal/ithemal/data_collection/build
cmake -D -DDynamoRIO_DIR=/home/ithemal/DynamoRIO-Linux-7.0.0-RC1/cmake ..
make -j"$(nproc --all)"
