cur=$(pwd)
cd /data/scratch/charithm/libraries/install/mysql
for file in /data/scratch/charithm/projects/cmodel/data/test/*
do
 # do something on $file
 ./bin/mysql --defaults-file=my.cnf --database=costmodel -u root -pmysql7788# --max_allowed_packet=32M -f < $file
done
echo $cur
cd $cur
