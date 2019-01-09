#!/usr/bin/env bash

set -ex

AWS_DOCKER_USER="$1"; shift
AWS_DOCKER_PASSWORD="$1"; shift
AWS_DOCKER_ENDPOINT="$1"; shift
MYSQL_USER="$1"; shift
MYSQL_PASSWORD="$1"; shift
MYSQL_HOST="$1"; shift
MYSQL_PORT="$1"; shift
IAM_CREDENTIAL="$1"; shift

sudo yum install -y docker tmux
sudo service docker start

IMAGE_ID="654586875650.dkr.ecr.us-east-2.amazonaws.com/ithemal:latest"

sudo docker login -u "${AWS_DOCKER_USER}" -p "${AWS_DOCKER_PASSWORD}" "${AWS_DOCKER_ENDPOINT}"
sudo docker pull "${IMAGE_ID}"

function dict_to_environ() {
    PREFIX=$1; shift
    python -c "import sys, json; print('\n'.join('${PREFIX}_{}=\'{}\''.format(*i) for i in json.load(sys.stdin).items()))"
}

# get the token secret configuration
source <(
    curl http://169.254.169.254/latest/meta-data/iam/security-credentials/"${IAM_CREDENTIAL}" \
        | dict_to_environ CRED \
        | grep 'AccessKeyId\|SecretAccessKey\|Token'

    curl http://169.254.169.254/latest/dynamic/instance-identity/document \
        | dict_to_environ REGION \
        | grep 'region'
)

sudo docker run -dit \
     --name ithemal \
     -v /home/ec2-user/ithemal:/home/ithemal/ithemal \
     -e ITHEMAL_HOME=/home/ithemal/ithemal \
     -e AWS_ACCESS_KEY_ID="${CRED_AccessKeyId}" \
     -e AWS_SECRET_ACCESS_KEY="${CRED_SecretAccessKey}" \
     -e AWS_SESSION_TOKEN="${CRED_Token}" \
     -e AWS_DEFAULT_REGION="${REGION_region}" \
     -p 8888:8888 \
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
