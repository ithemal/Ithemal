# Phases of running Ithemal

## Prerequisites

You should have a running instance of `mysqld` and should have already exported the timing database to the MySQL server. Please fill in database, user, password and port values in the commands that follow corresponding to your database.

## Serializing timing data

First, serialize timing of a given architecture to a pickle object. This is done for increasing I/O performance and to reduce data loading time during training. For example, let's serialize actual timing numbers for the Skylake architecture (arch=2). Use the save mode for this task.

`python run_ithemal.py --mode=save --database=[db_name] --user=[username] --password=[password] --port=[port_number] --savedatafile=../saved/time_skylake.data --arch=2`

## Training

Next, train Ithemal's neural network model using the train mode.

`python run_ithemal.py --mode=train --savedatafile=../inputs/data/time_skylake.data --embedfile=../inputs/embeddings/code_delim.emb --embmode=none --savefile=../inputs/models/graph_skylake.mdl`

This will train Ithemal and validate its performance on a test set.

## Validating

If you want to validate the performance of a pretrained model, you can use the validate mode.

`python run_ithemal.py --mode=validate --savedatafile=../inputs/data/time_skylake.data --embedfile=../inputs/embeddings/code_delim.emb --embmode=none --loadfile=../inputs/models/graph_skylake.mdl`

## Updating the database

If you are satisfied with the results of the trained model, you can use it to update the database with predictions for throughput for each basic block. Note that, this mode updates predictions for all basic blocks (not only for the test set).

`python run_ithemal.py --mode=predict --database=[db_name] --user=[username] --password=[password] --port=[port_number]  --format=text --savedatafile=../inputs/data/timing_skylake.data --embedfile=../inputs/embeddings/code_delim.emb --loadfile=../inputs/models/graph_skylake.mdl --embmode=none --arch=2`