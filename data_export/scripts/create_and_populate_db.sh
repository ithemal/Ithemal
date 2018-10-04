CFG_FILE=$1
DB=$2
SCHEMA=$3
SQL_FOLDER=$4

mysql --defaults-file=$CFG_FILE -e"DROP DATABASE IF EXISTS $DB; CREATE DATABASE IF NOT EXISTS $DB;"
mysql --defaults-file=$CFG_FILE --database=$DB --max_allowed_packet=32M -f < $SCHEMA

static_files=$(find $SQL_FOLDER -type f -name '*.sql' -not -name 'static_perlbench_r_*' -not -name 'static_perlbench_s_*' -name 'static_*')
dyn_files=$(find $SQL_FOLDER -type f -name '*.sql' -not -name 'dyn_perlbench_r_*' -not -name 'dyn_perlbench_s_*' -name 'dyn_*' )


for file in $static_files
do
 echo $file
 mysql --defaults-file=$CFG_FILE --database=$DB --max_allowed_packet=32M -f < $file
done

for file in $dyn_files
do
 # do something on $file
 echo $file
 mysql --defaults-file=$CFG_FILE --database=$DB --max_allowed_packet=32M -f < $file
done

