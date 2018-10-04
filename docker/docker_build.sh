#!/usr/bin/env bash

sudo docker build --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) -t ithemal:latest "$(dirname $0)"
