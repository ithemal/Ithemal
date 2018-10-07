#!/usr/bin/env bash

set -ex

sudo docker build --build-arg HOST_UID=$(id -u) -t ithemal:latest "$(dirname $0)"
