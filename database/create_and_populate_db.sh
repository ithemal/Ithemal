cur=$(pwd)
cd /data/scratch/charithm/libraries/install/mysql

 ./bin/mysql --defaults-file=my.cnf -u root -pmysql7788# -e"DROP DATABASE IF EXISTS $1; CREATE DATABASE IF NOT EXISTS $1;"

 ./bin/mysql --defaults-file=my.cnf --database=$1 -u root -pmysql7788# --max_allowed_packet=32M -f < /data/scratch/charithm/projects/cmodel/database/mysql_schema.sql

static_files=$(find /data/scratch/charithm/projects/cmodel/data/$2 -type f -name '*.sql' -not -name 'static_perlbench_r_*' -not -name 'static_perlbench_s_*' -name 'static_*')
dyn_files=$(find /data/scratch/charithm/projects/cmodel/data/$2 -type f -name '*.sql' -not -name 'dyn_perlbench_r_*' -not -name 'dyn_perlbench_s_*' -name 'dyn_*' )


for file in $static_files
do
 # do something on $file
 ./bin/mysql --defaults-file=my.cnf --database=$1 -u root -pmysql7788# --max_allowed_packet=32M -f < $file
done

for file in $dyn_files
do
 # do something on $file
 ./bin/mysql --defaults-file=my.cnf --database=$1 -u root -pmysql7788# --max_allowed_packet=32M -f < $file
done

echo $cur
cd $cur

