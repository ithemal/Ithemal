
# Overview

Ithemal is the first data driven model for predicting throughput of a basic block of x86-64 instructions.
More details about Ithemal's approach can be found in our [paper](https://arxiv.org/abs/1808.07412).

# Dependencies

We personally recommend downloading the required sources (for the versions mentioned) and building dependencies from scratch, but you are welcome to install using your favourite package manager. We have always built from sources and have not tested the versions included with package managers (except for packages annotated with PM). Download link to source distributions of dependencies are provided (in most cases).

* Common
  * Boost 1.59 [link](https://www.boost.org/users/download/)
  * cmake - version > 3.1 [link](https://cmake.org/download/)
  * libncurses5-dev (PM)

* Data Collection
  * DynamoRIO [link](https://github.com/DynamoRIO/dynamorio/wiki/Downloads).
  * SQLite [link](https://www.sqlite.org/download.html).

* Data Export
  * MySQL server 5.7 [link](https://dev.mysql.com/downloads/mysql/5.7.html) - We do not support MySQL 8; build with -DWITH_BOOST=/path/to/boost, note that higher versions of boost doesn't work.

* Training and Inference
  * Python 2.7
  * Python packages - we recommend using a python virtual environment when installing these packages 
    * MySQL python connector 2.1 [link](https://dev.mysql.com/downloads/connector/python/) - We do not support connector 8 API
    * PyTorch 0.4 or higher [link](https://pytorch.org).

# Organization

This repo contains software to generate basic blocks from existing binary programs, time them using Agner Fog's timing scripts
(should be separately downloaded), populate databases and neural network models to learn throughput prediction.

## Basic Block collection

Ithemal dynamically collects all the basic blocks run by binaries using the dynamic binary instrumentation framework, [DynamoRIO](http://dynamorio.org). DynamoRIO clients needed to perform basic block collection are located within drclients folder.