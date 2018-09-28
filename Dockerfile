FROM debian:stretch

RUN groupadd -g 999 ithemal && \
    useradd -m -r -u 999 -g ithemal ithemal

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libsqlite3-dev              \
        build-essential             \
        cmake                       \
        coreutils                   \
        curl                        \
        git                         \
        gnupg                       \
        man                         \
        python-dev                  \
        qt5-default                 \
        sqlite3                     \
        sudo                        \
        vim                         \
        wget

# install mysql
RUN wget -O /tmp/RPM-GPG-KEY-mysql https://repo.mysql.com/RPM-GPG-KEY-mysql
RUN apt-key add /tmp/RPM-GPG-KEY-mysql
RUN /bin/echo -e "deb http://repo.mysql.com/apt/debian/ stretch mysql-5.7\ndeb-src http://repo.mysql.com/apt/debian/ stretch mysql-5.7" > /etc/apt/sources.list.d/mysql.list
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server
COPY docker_my.cnf /etc/my.cnf

RUN wget --quiet -O /tmp/install_conda.sh https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
    sh /tmp/install_conda.sh -b -p /opt/conda && \
    rm /tmp/install_conda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile

USER ithemal
WORKDIR /home/ithemal

RUN /opt/conda/bin/conda create --name ithemal python=2.7
RUN /opt/conda/bin/conda install -q -n ithemal -c pytorch pytorch torchvision
RUN /opt/conda/bin/conda install -q -n ithemal mysql-connector-python=2 matplotlib=2.2.3 psutil=5.4.7 tqdm scikit-learn=0.19 numpy=1.15 scipy=1.1.0 statistics=1.0.3.5 ipython-notebook ipython pandas pyqt

RUN curl -sL https://github.com/DynamoRIO/dynamorio/releases/download/release_7_0_0_rc1/DynamoRIO-Linux-7.0.0-RC1.tar.gz | tar xz

RUN bash -lc 'source activate ithemal; jupyter notebook --generate-config'
RUN echo 'export DYNAMORIO_HOME=/home/ithemal/DynamoRIO-Linux-7.0.0-RC1' >> /home/ithemal/.bashrc
RUN echo 'source activate ithemal' >> /home/ithemal/.bashrc
