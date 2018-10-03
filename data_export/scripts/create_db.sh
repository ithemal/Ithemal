CFG_FILE=$1
DB=$2
SCHEMA=$3

mysql --defaults-file=$CFG_FILE -e"DROP DATABASE IF EXISTS $DB; CREATE DATABASE IF NOT EXISTS $DB;"
mysql --defaults-file=$CFG_FILE --database=$DB --max_allowed_packet=32M -f < $SCHEMA
