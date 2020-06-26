FROM debian:stretch

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        build-essential             \
        cmake                       \
        coreutils                   \
        curl                        \
        git                         \
        gnupg                       \
        libsqlite3-dev              \
        locales                     \
        man                         \
        nasm                        \
        pv                          \
        python-dev                  \
        qt5-default                 \
        sqlite3                     \
        sudo                        \
        tmux                        \
        unzip                       \
        vim                         \
        wget                        \
        zip

# install mysql
RUN wget -O /tmp/RPM-GPG-KEY-mysql https://repo.mysql.com/RPM-GPG-KEY-mysql && \
        apt-key add /tmp/RPM-GPG-KEY-mysql && \
        /bin/echo -e "deb http://repo.mysql.com/apt/debian/ stretch mysql-5.7\ndeb-src http://repo.mysql.com/apt/debian/ stretch mysql-5.7" > /etc/apt/sources.list.d/mysql.list && \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server

COPY docker_my.cnf /etc/my.cnf

RUN wget --quiet -O /tmp/install_conda.sh https://repo.continuum.io/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh && \
        sh /tmp/install_conda.sh -b -p /opt/conda && \
        rm /tmp/install_conda.sh

COPY env /tmp/docker_env
RUN /opt/conda/bin/conda create --name ithemal --file /tmp/docker_env

RUN curl -sL https://github.com/DynamoRIO/dynamorio/releases/download/release_7_0_0_rc1/DynamoRIO-Linux-7.0.0-RC1.tar.gz | tar xz -C /opt
ENV DYNAMORIO_HOME "/opt/DynamoRIO-Linux-7.0.0-RC1"
# tar will not give you `755 & ~umask` because tar is evil
RUN chown -R root:root "${DYNAMORIO_HOME}" && \
        find "${DYNAMORIO_HOME}" -type d -exec chmod 755 {} \; && \
        find "${DYNAMORIO_HOME}" -type f -exec chmod 644 {} \;

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ARG HOST_UID=1000
ENV HOST_UID $HOST_UID

RUN groupadd -g 1000 ithemal && useradd -m -s /bin/bash -r -u $HOST_UID -g ithemal ithemal
USER ithemal
WORKDIR /home/ithemal

# non-login shell
RUN /bin/echo 'export PATH=/opt/conda/bin:$PATH' >> /home/ithemal/.bash_profile && \
        /bin/echo 'source activate ithemal' >> /home/ithemal/.bash_profile && \
        /bin/echo 'export PYTHONPATH="/home/ithemal/ithemal/learning/pytorch"' >> /home/ithemal/.bash_profile

# login shell
RUN /bin/echo 'export PATH=/opt/conda/bin:$PATH' >> /home/ithemal/.bashrc && \
        /bin/echo 'source activate ithemal' >> /home/ithemal/.bashrc && \
        /bin/echo 'export PYTHONPATH="/home/ithemal/ithemal/learning/pytorch"' >> /home/ithemal/.bashrc

RUN bash -lc 'pip install --upgrade --user pyhamcrest pip; jupyter notebook --generate-config'

COPY notebook_config.patch /tmp/_docker_notebook_conf.patch
RUN patch .jupyter/jupyter_notebook_config.py < /tmp/_docker_notebook_conf.patch
