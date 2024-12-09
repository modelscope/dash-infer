#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 [-h <host>] [-p <port>] [-m] -- --model-path <model_path_in_huggingface> --revision <revision>"
    echo "Example:
    $0 -h 127.0.0.1 -p 8088 -- --model-path qwen/Qwen-7B-Chat --revision master"
}

cleanup() {
    echo "stopping all processes..."
    pkill -P $1
    rm -f *.log
}


host="127.0.0.1"
port="8000"

OPTIND=1
while getopts ":h:p:m" opt; do
    case "$opt" in
        h )
            host="$OPTARG"
            ;;
        p )
            port="$OPTARG"
            ;;
        m )
            export FASTCHAT_USE_MODELSCOPE=True
            ;;
        * )
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))


# start fastchat controller
python3 -m fastchat.serve.controller --host 127.0.0.1 &
# start fastchat openai_api_server
python3 -m fastchat.serve.openai_api_server --host $host --port $port &
# start fastchat worker: allspark_worker
python3 allspark_worker.py $@ &



# stop all services
pgid=$$
trap "cleanup $pgid" SIGINT SIGTERM EXIT

while true; do
    sleep 1
done
