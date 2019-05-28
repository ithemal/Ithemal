FROM debian:stretch

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libsqlite3-dev              \
        locales                     \
        build-essential             \
        cmake                       \
        coreutils                   \
        curl                        \
        git                         \
        gnupg                       \
        man                         \
        nasm                        \
        python-dev                  \
        pv                          \
        qt5-default                 \
        sqlite3                     \
        sudo                        \
        tmux                        \
        vim                         \
        wget

# install mysql
RUN wget -O /tmp/RPM-GPG-KEY-mysql https://repo.mysql.com/RPM-GPG-KEY-mysql && \
        apt-key add /tmp/RPM-GPG-KEY-mysql && \
        /bin/echo -e "deb http://repo.mysql.com/apt/debian/ stretch mysql-5.7\ndeb-src http://repo.mysql.com/apt/debian/ stretch mysql-5.7" > /etc/apt/sources.list.d/mysql.list && \
        apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server

COPY docker_my.cnf /etc/my.cnf

RUN wget --quiet -O /tmp/install_conda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
        sh /tmp/install_conda.sh -b -p /opt/conda && \
        rm /tmp/install_conda.sh

RUN /opt/conda/bin/conda create --name ithemal python=2.7
RUN /opt/conda/bin/conda install -q -n ithemal -c pytorch pytorch=1.1.0
RUN /opt/conda/bin/conda install -q -n ithemal \
        cython=0.29 \
        graphviz=2.40.1 \
        ipython=5.8.0 \
        ipython-notebook=4.0.4 \
        matplotlib=2.2.3 \
        mysql-connector-python=2 \
        numpy=1.15 \
        pandas=0.23.4 \
        psutil=5.4.7 \
        pyqt=5.9.2 \
        pytest=3.8.1 \
        python-graphviz=0.8.4 \
        requests=2.21.0 \
        scikit-learn=0.19 \
        scipy=1.1.0 \
        seaborn=0.9.0 \
        statistics=1.0.3.5 \
        tqdm=4.28.1 \
        typing=3.6.6 \
        zeromq=4.2.5

RUN /opt/conda/bin/conda install -q -n ithemal -c conda-forge awscli

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
