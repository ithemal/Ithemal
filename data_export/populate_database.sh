cur=$(pwd)
cd /data/scratch/charithm/libraries/install/mysql

for file in /data/scratch/charithm/projects/cmodel/data/$2/static_*
do
 # do something on $file
 ./bin/mysql --defaults-file=my.cnf --database=$1 -u root -pmysql7788# --max_allowed_packet=32M -f < $file
done

for file in /data/scratch/charithm/projects/cmodel/data/$2/dyn_*
do
 # do something on $file
 ./bin/mysql --defaults-file=my.cnf --database=$1 -u root -pmysql7788# --max_allowed_packet=32M -f < $file
done

echo $cur
cd $cur

