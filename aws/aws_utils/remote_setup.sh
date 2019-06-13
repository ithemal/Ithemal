#!/usr/bin/env bash

set -ex

AWS_DOCKER_USER="$1"; shift
AWS_DOCKER_PASSWORD="$1"; shift
AWS_DOCKER_ENDPOINT="$1"; shift
MYSQL_USER="$1"; shift
MYSQL_PASSWORD="$1"; shift
MYSQL_HOST="$1"; shift
MYSQL_PORT="$1"; shift
REGION="$1"; shift

sudo yum install -y docker tmux
sudo service docker start

IMAGE_ID="654586875650.dkr.ecr.us-east-2.amazonaws.com/ithemal:latest"

sudo docker login -u "${AWS_DOCKER_USER}" -p "${AWS_DOCKER_PASSWORD}" "${AWS_DOCKER_ENDPOINT}"
sudo docker pull "${IMAGE_ID}"

sudo docker run -dit \
     --name ithemal \
     -v /home/ec2-user/ithemal:/home/ithemal/ithemal \
     -e ITHEMAL_HOME=/home/ithemal/ithemal \
     -p 8888:8888 \
     -p 80:5000 \
     -p 443:5443 \
     "${IMAGE_ID}"

sudo docker exec -u ithemal ithemal bash -lc '/home/ithemal/ithemal/build_all.sh'
sudo docker exec -d -u ithemal ithemal bash -lc "jupyter notebook --ip 0.0.0.0 --port 8888 /home/ithemal/ithemal/learning/pytorch/notebooks --no-browser --NotebookApp.token='ithemal'"
sudo docker exec -i -u ithemal ithemal bash -lc 'cat > ~/.my.cnf' <<EOF
[client]
host=${MYSQL_HOST}
port=${MYSQL_PORT}
user=${MYSQL_USER}
password=${MYSQL_PASSWORD}
database=ithemal
EOF

sudo docker exec -i -u ithemal ithemal bash -lc 'mkdir ~/.aws; cat > ~/.aws/config' <<EOF
[default]
region=${REGION}
EOF
