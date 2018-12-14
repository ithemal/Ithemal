#!/usr/bin/env bash

set -ex

sudo $(aws ecr get-login --no-include-email --region us-east-2)

AWS_HOST_UID=500

sudo docker build --build-arg "HOST_UID=${AWS_HOST_UID}" -t ithemal-aws:latest "$(dirname $0)"
sudo docker tag ithemal-aws:latest 654586875650.dkr.ecr.us-east-2.amazonaws.com/ithemal:latest
sudo docker push 654586875650.dkr.ecr.us-east-2.amazonaws.com/ithemal:latest
