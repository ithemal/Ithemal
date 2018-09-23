# Phases of running Ithemal

## Prerequisites

You should have a running instance of `mysqld` and should have already exported the timing database to the MySQL server. Please fill in database, user, password and port values in the commands that follow corresponding to your database.

## Serializing timing data

First, serialize timing of a given architecture to a pickle object. This is done for increasing I/O performance and to reduce data loading time during training. For example, let's serialize actual timing numbers for the Skylake architecture (arch=2).

`python run_ithemal.py --mode=save --database=[db_name] --user=[username] --password=[password] --port=[port_number] --savedatafile=../saved/time_skylake.data --arch=2`
