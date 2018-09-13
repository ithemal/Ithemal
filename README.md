
# Overview

Ithemal is the first data driven model for predicting throughput of a basic block of x86-64 instructions.
More details about Ithemal's approach can be found in our [paper](https://arxiv.org/abs/1808.07412).

# Dependencies

* Common
  * cmake - version > 3.1

* Data Collection
  * DynamoRIO - download and build the latest DyanmoRIO version from [here](https://github.com/DynamoRIO/dynamorio/wiki/Downloads).
  * SQLite - download the latest build from [here](https://www.sqlite.org/download.html).

# Organization

This repo contains software to generate basic blocks from existing binary programs, time them using Agner Fog's timing scripts
(should be separately downloaded), populate databases and neural network models to learn throughput prediction.

## Basic Block collection

Ithemal dynamically collects all the basic blocks run by binaries using the dynamic binary instrumentation framework, [DynamoRIO](http://dynamorio.org). DynamoRIO clients needed to perform basic block collection are located within drclients folder.