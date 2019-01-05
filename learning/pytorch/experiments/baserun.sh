cd "${ITHEMAL_HOME}/learning/pytorch"

if [ ! -f saved/time_skylake_1217.data ]; then
   mkdir -p saved
   pushd saved
   wget https://www.dropbox.com/s/qjcjje5hjrljd5a/time_skylake_1217.data
   popd
fi

python ithemal/run_ithemal.py --embmode none --embedfile inputs/embeddings/code_delim.emb --savedatafile saved/time_skylake_1217.data --arch 2 --epochs 5 --batch-size 4 "${@}"
