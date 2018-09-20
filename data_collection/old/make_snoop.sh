gcc -c snoop.c -I/data/scratch/charithm/libraries/install/mysql/include -L/data/scratch/charithm/libraries/install/mysql/lib -lmysqlclient -o build/snoop.o
gcc -c mysql_impl.c -I/data/scratch/charithm/libraries/install/mysql/include -L/data/scratch/charithm/libraries/install/mysql/lib -lmysqlclient -o build/mysql_impl.o
gcc build/snoop.o build/mysql_impl.o -o build/snoop -L/data/scratch/charithm/libraries/install/mysql/lib -lmysqlclient
