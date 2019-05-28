
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
This will drop you into a tmux shell in the container (additionally starting a Jupyter notebook exposed on port 8888; nothing depends on this except for your own convenience, so feel free to disable exposing the notebook by removing the port forwarding on lines 37 and 38 in `docker/docker-compose.yml`).
The container will continue running in the background, even if you exit.
The container can be stopped with `docker/docker_stop.sh` from the host machine.
To detach from the container while keeping jobs running, use the normal tmux detach command of `Control-b d`; running `docker/docker_connect.sh` will drop you back into the same session.

## Prediction

Models can be downloaded from [the Ithemal models repository](https://www.github.com/psg-mit/Ithemal-models).
Models are split into two parts: the model architecture and the model data (this is an unfortunate historical artifact more than a good design decision).
The model architecture contains the code and the token embedding map for the model, and the model data contains the learned model tensors.
The versions of the model reported in the paper are:
- The [Haswell model](https://github.com/psg-mit/Ithemal-models/blob/master/paper/haswell/)
- The [Skylake model](https://github.com/psg-mit/Ithemal-models/blob/master/paper/skylake/)
- The [Ivy Bridge model](https://github.com/psg-mit/Ithemal-models/blob/master/paper/ivybridge/)

### Command Line

Ithemal can be used as a drop-in replacement for the throughput prediction capabilities of `IACA` or `llvm-mca` via the `learning/pytorch/ithemal/predict.py` script.
Ithemal uses the same convention as `IACA` to denote what code should be predicted; this can be achieved with any of the files in `learning/pytorch/examples`, or by consulting [the IACA documentation](https://software.intel.com/en-us/articles/intel-architecture-code-analyzer).

Once you have downloaded one of the models from the previous section, and you have compiled a piece of code to some file, you can generate prediction for this code with something like:
```
python learning/pytorch/ithemal/predict.py --verbose --model predictor.dump --model-data trained.mdl --file a.out
```

### Python API

TODO: Notebook showing how to generate predictions (essentially copy/paste of predict.py)

## Training

### Data

#### Canonicalization

To canonicalize a basic block so that it can be used as input for Ithemal, first get the hex representation of the basic block you want to predict (i.e. via `xxd`, `objdump`, or equivalent).
For instance, the instruction `push   0x000000e3` is represented in hex as `68e3000000`.
Next, run the tokenizer as follows:
```
data_collection/build/bin/tokenizer {HEX_CODE} --token
```
which will output an XML representation of the basic block, with all implicit operands expanded.
This is necessary to store in the dataset.

#### Representation

Raw data is represented as a list of tuples containing `(code_id, timing, code_intel, code_xml)`, where `code_id` is a unique identifier for the code, `timing` is the float representing the timing of the block, `code_intel` is the human-readable block string (this gets split on newlines and attached to instructions in the block. This is only for debugging and can be empty or `None`), and `code_xml` is the result of the tokenizer on that block.
To store datasets, we `torch.save` and `torch.load` this list of tuples.
The first 80% of a dataset is loaded as the train set, and the last 20% is the test set.
For an example of one of these datasets, look at [a small sample of our training data](TODO THIS IS BROKEN).

### Model Training

To train a model, pick a suitable `EXPERIMENT_NAME` and `EXPERIMENT_TIME`, and run the following command:
```
python /home/ithemal/ithemal/learning/pytorch/ithemal/run_ithemal.py --data {DATA_FILE} --use-rnn train --experiment-name {EXPERIMENT_NAME} --experiment-time {EXPERIMENT_TIME} --sgd --threads 4 --trainers 6 --weird-lr --decay-lr --epochs 100
```
which will train a model with the parameters reported in the paper.
The results of this are saved into `learning/pytorch/saved/EXPERIMENT_NAME/EXPERIMENT_TIME/`, which we will refer to as `RESULT_DIR` from now on.
The training loss is printed live as the model is trained, and also saved into `RESULT_DIR/loss_report.log`, which is a tab-separated list of `epoch, elapsed time, training loss, number of active parallel trainers`.
The results of the trained model on the test set are stored in `RESULT_DIR/validation_results.txt`, which consists of a list of of the `predicted,actual` value of each item in the test set, followed by the overall loss of the trained model on the test set at the end.
Finally, the resulting trained models and predictor dumps (for use in the command line API above) are saved in `RESULT_DIR/trained.mdl` and `RESULT_DIR/predictor.dump` respectively.

# Documentation

TODO: explain that Ithemal-related code is in learning/pytorch and common

# Data Collection

TODO: what do we want to say about this?
