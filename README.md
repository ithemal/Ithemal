
# Overview

Ithemal is the first data driven model for predicting throughput of a basic block of x86-64 instructions.
More details about Ithemal's approach can be found in our [paper](https://arxiv.org/abs/1808.07412).

# Dependencies

We personally recommend downloading the required sources (for the versions mentioned) and building dependencies from scratch, but you are welcome to install using your favourite package manager. We have always built from sources and have not tested the versions included with package managers (except for packages annotated with PM). Download link to source distributions of dependencies are provided (in most cases).

* Common
  * Boost 1.59 [(link)](https://www.boost.org/users/download/)
  * cmake 3.1 or higher [(link)](https://cmake.org/download/)
  * libncurses5-dev (PM)
  * python-dev (PM)

* Data Collection
  * DynamoRIO 7.0.0 [(link)](https://github.com/DynamoRIO/dynamorio/wiki/Downloads).
  * SQLite3 [(link)](https://www.sqlite.org/download.html).

* Data Export
  * MySQL server 5.7 [(link)](https://dev.mysql.com/downloads/mysql/5.7.html) - We do not support MySQL 8; build with -DWITH_BOOST=/path/to/boost, note that higher versions of boost doesn't work.

* Training and Inference
  * Python 2.7
  * virtualenv
  * Python packages - We recommend using a python virtual environment when installing these packages; you can use pip install. 
    * MySQL python connector 2.1 [(link)](https://dev.mysql.com/downloads/connector/python/) - We do not support connector 8 API
    * PyTorch 0.4 or higher [(link)](https://pytorch.org).
    * matplotlib 2.2.3
    * psutil 5.4.7
    * tqdm 4.26
    * scikit-learn 0.19.2
    * numpy 1.15
    * scipy 1.1.0
    * statistics 1.0.3

# Organization

This repo contains software to extract basic blocks from existing binary programs, time them using Agner Fog's timing scripts
(should be separately downloaded), populate databases and train neural network models to learn throughput prediction.

## Data Collection

Ithemal dynamically collects all the basic blocks run by binaries using the dynamic binary instrumentation framework, [DynamoRIO](http://dynamorio.org). DynamoRIO clients needed to perform basic block collection are located within `data_collection` folder. DynamoRIO `static` client dumps collected basic block data into a collection of SQL files which can be exported into a database.

## Data Export

Folder `data_export` contains tools used for exporting collected SQL files into a database. Subfolder `schemas` contains the SQL schemas which describe the database structure of the dumped SQL files. They describe the composition of each table and field of the database. Subfolder `scripts` contains convenience scripts which can be used to export SQL files to a database.

## Learning

Ithemal's core learning routines are located in the `learning` folder. It contains code to build neural network models, preprocess data to be fed into those models, training scripts as well as validation scripts. Ithemal uses Pytorch Neural Network framework for building the DAG-RNN used for prediction.

## Comparison Tools

We compare Ithemal with llvm-mca and Intel's IACA. `timing_tools` folder contains scripts we used for collecting throughput predictions using these systems. Further, we include scripts we used to collect ground truth throughput values.

## Testing

Folder `testing` contains some sanity checks that should pass before you run any component of Ithemal.

# Building

Only parts which need explicit building are the DynamoRIO clients located within `data_collection` folder. The instructions are located in the README file within its folder.

# Running

Before running any Ithemal related code, please ensure you do the following in the Ithemal root folder.

`source setup.sh`

This sets up some environment variables needed to run components of Ithemal successfully.