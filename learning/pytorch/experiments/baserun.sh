#!/usr/bin/env python3

set -ex

if [ "$#" -lt 1 ]; then
    echo "Requires a name parameter"
    exit 1
fi

NAME=$1; shift

cd "${ITHEMAL_HOME}/learning/pytorch"

if [ ! -f saved/time_skylake_1217.data ]; then
   mkdir -p saved
   pushd saved
   wget https://www.dropbox.com/s/qjcjje5hjrljd5a/time_skylake_1217.data
   popd
fi

NAMEDATE="${NAME}_$(date '+%m-%d-%y_%H:%M:%S')"
SAVEFILE="saved/${NAMEDATE}.mdl"
LOSS_REPORT_FILE="loss_reports/${NAMEDATE}.log"

python ithemal/run_ithemal.py --embmode none --embedfile inputs/embeddings/code_delim.emb --savedatafile saved/time_skylake_1217.data --arch 1 --epochs 5 --savefile "${SAVEFILE}" --loss-report-file "${LOSS_REPORT_FILE}" "${@}"

"${ITHEMAL_HOME}/aws/ping_slack.py" "Experiment ${NAME} complete"
