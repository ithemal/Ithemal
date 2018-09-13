
# Overview

Ithemal is the first data driven model for predicting throughput of a basic block of x86-64 instructions.
More details about Ithemal's approach can be found in our [paper](https://arxiv.org/abs/1808.07412).

# Dependencies

* Common
  * cmake - version > 3.1

* Data Collection
  * DynamoRIO - download and build the latest DyanmoRIO version from [here](https://github.com/DynamoRIO/dynamorio/wiki/Downloads).
  * SQLite - We recommend downloading the latest source from [here](https://www.sqlite.org/download.html) and building it.

* Data Export
  * MySQL server 5.7 - download and build from [here](https://dev.mysql.com/downloads/mysql/5.7.html) - we do not support MySQL 8

* Training and Inference
  * Python 2.7
  * Python packages - we recommend using a python virtual environment when installing these packages 
    * MySQL python connector 2.1 - Download from [here](https://dev.mysql.com/downloads/connector/python/). We do not support connector 8 API
    * PyTorch 0.4 or higher - download the latest source from [here](https://pytorch.org) and build.

# Organization

This repo contains software to generate basic blocks from existing binary programs, time them using Agner Fog's timing scripts
(should be separately downloaded), populate databases and neural network models to learn throughput prediction.

## Basic Block collection

Ithemal dynamically collects all the basic blocks run by binaries using the dynamic binary instrumentation framework, [DynamoRIO](http://dynamorio.org). DynamoRIO clients needed to perform basic block collection are located within drclients folder.