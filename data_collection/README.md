
# Build Instructions

* Make sure LIBRAY_PATH is prepended to point to the SQLite lib folder and CPATH is prepended to point to the SQLite include folder.
* mkdir build; cd build
* cmake -DDynamoRIO_DIR=/path/to/dynamorio/cmake/folder ..
* make

# Running Instructions

## sample invocation of DynamoRIO under a client

* export DYNAMORIO_HOME=/path/to/dynamorio/home
* $DYNAMORIO_HOME/build/drrun -c /path/to/client \<client_arguments\> \-\- \<binary\>

## static DR client

This is used for collecting textual (Intel or AT&T or both) and tokenized representations of the basic blocks. Output can directly
populate a SQLite database or can dump SQL files.

* arguments (in order)
  * mode - memory mapped snooping by a separate database, raw SQL file dump, SQLite
  * Intel, AT&T and token outputs - bitwise ORed value that controls which versions are outputted
  * inserting or updating - whether the SQL files should be dumped for inserting into an empty database or for updating an existing
  database
  * compiler name - what compiler was used to compile the binary
  * compiler flags
  * data folder - where the SQL dump should be created


## timing DR client

This client is used for timing basic blocks while running under DynamoRIO control. Due to inaccuracies in the timing
measurements this method is not preferred and not used by Ithemal.
   