#!/usr/bin/env bash

source "$(dirname $0)/_docker_utils.sh"
get_sudo
sudo docker build --build-arg HOST_UID=$(id -u) -t ithemal:latest "$(dirname $0)"
