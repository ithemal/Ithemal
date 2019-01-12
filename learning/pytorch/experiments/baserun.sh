set -ex

if [ "$#" -lt 1 ]; then
    echo "Requires a name parameter"
    exit 1
fi

NAME=$1; shift

cd "${ITHEMAL_HOME}/learning/pytorch"

experiments/download_data.sh

NAMEDATE="${NAME}_$(date '+%m-%d-%y_%H:%M:%S')"
SAVEFILE="saved/${NAMEDATE}.mdl"
LOSS_REPORT_FILE="loss_reports/${NAMEDATE}.log"

mkdir -p saved/checkpoints

python ithemal/run_ithemal.py --embmode none --embedfile inputs/embeddings/code_delim.emb --savedatafile saved/time_skylake_1217.data --arch 2 --epochs 5 --savefile "${SAVEFILE}" --loss-report-file "${LOSS_REPORT_FILE}" --checkpoint-dir saved/checkpoints "${@}" &
ITHEMAL_PID=$!

# while train process is still running...
while kill -0 $ITHEMAL_PID > /dev/null 2>&1; do

    # get the list of new checkpoints
    FILES=($(aws s3 sync --dryrun saved/checkpoints/ s3://ithemal-checkpoints/checkpoint_models/ | grep upload | rev | cut -d'/' -f1 | rev))
    aws s3 sync saved/checkpoints/ s3://ithemal-checkpoints/checkpoint_models/

    for f in "${FILES[@]}"; do
        echo '${ITHEMAL_HOME}/learning/pytorch/experiments/benchmark_checkpoint.sh' "${f}" | "${ITHEMAL_HOME}/aws/queue.py" send checkpoint_queue >/dev/null 2>&1 || :
    done

    aws s3 cp "${LOSS_REPORT_FILE}" s3://ithemal-checkpoints/loss_reports/

    sleep 100
done

aws s3 cp "${SAVEFILE}" s3://ithemal-checkpoints/trained_models/
aws s3 cp "${LOSS_REPORT_FILE}" s3://ithemal-checkpoints/loss_reports/

"${ITHEMAL_HOME}/aws/ping_slack.py" "Experiment ${NAME} complete"
