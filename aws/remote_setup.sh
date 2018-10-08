#!/usr/bin/env bash

set -ex

sudo yum install -y docker
sudo service docker start

# TODO: this doesn't successfully pull from the container yet -- we should do something along the lines of `aws ecr get-login --no-include-email --region us-east-2`
# TODO: we also need to set up database connection
# TODO: we also need to make it easy to SCP stuff in.

sudo docker pull 654586875650.dkr.ecr.us-east-2.amazonaws.com/ithemal:latest
HOST_PATH=$(readlink -f $(dirname $0)/..)
sudo docker run -dit --name ithemal -v ${HOST_PATH}:/home/ithemal/ithemal ithemal:latest
