cur=$(pwd)
cd /data/scratch/charithm/libraries/install/mysql

files=$(find /data/scratch/charithm/projects/cmodel/data/$1 -type f -name '*.sql' -not -name 'dyn_perlbench_r_*' -not -name 'dyn_perlbench_s_*' -name 'dyn_*')

for file in $files
do
 # do something on $file
 echo $file
done

echo $cur
cd $cur

