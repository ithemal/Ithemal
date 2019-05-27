
# Overview

Ithemal is a data driven model for predicting throughput of a basic block of x86-64 instructions.
More details about Ithemal's approach can be found in our [paper](https://arxiv.org/abs/1808.07412).

# Usage

## Environment

You first need to install [docker](https://github.com/docker/docker-install) and [docker-compose](https://docs.docker.com/compose/install/).

It is easiest to run Ithemal within the provided Docker environment.
To build the Docker environment, run `docker/docker_build.sh`.
No user interaction is required during the build, despite the various prompts seemingly asking for input.

Once the docker environment is built, connect to it with `docker/docker_connect.sh`.
This will drop you into a tmux shell in the container.
The container will continue running in the background, even if you exit.
The container can be stopped with `docker/docker_stop.sh` from the host machine.
To detach from the container while keeping jobs running, use the normal tmux detach command of `Control-b d`; running `docker/docker_connect.sh` will drop you back into the same session.

## Prediction

Models can be downloaded from [the Ithemal models repository](https://www.github.com/psg-mit/Ithemal-models).

TODO: Model download instructions

### Command Line

Ithemal can be used as a drop-in replacement

### Python API

TODO: Notebook showing how to generate predictions (essentially copy/paste of predict.py)

## Training

### Data

TODO: how to download and format data

### Model Training

TODO: How to train a model using our hyperparameters

### Model Export

TODO: how to dump a model

# Documentation

TODO: explain that Ithemal-related code is in learning/pytorch and common

# Data Collection

TODO: what do we want to say about this?
