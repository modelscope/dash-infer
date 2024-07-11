#!/usr/bin/env bash

set -e

OPTIND=1
host="localhost"
port="8000"
model_path="qwen/Qwen-7B-Chat"
model_revision=""
conv_template=""
dashinfer_config=""

while getopts ":h:p:m:r" opt; do
    case "$opt" in
        h ) host="$OPTARG";;
        p ) port="$OPTARG";;
        m ) model_path="$OPTARG";;
        r ) model_revision="$OPTARG";;
        t ) conv_template="$OPTARG";;
        \? ) echo -e "Invalid Options: -$OPTARG" >&2
            exit 1;;
        : ) echo -e "Option -$OPTARG requires an argument." >&2
            exit 1;;
    esac
done
shift $((OPTIND-1))


dashinfer_config=$1
if [ -z "$dashinfer_config" ]; then
    echo -e "You must provide a dashinfer json config file. \nusage: \n$0 CONFIG_FILE" >&2
    exit 1
fi

if [ ! -f "$dashinfer_config" ]; then
    echo -e "The dashinfer json config file $dashinfer_config is not existed." >&2
    exit 1
fi

echo -e "using config file: $dashinfer_config"
    


# start fastchat controller
python3 -m fastchat.serve.controller &
# start fastchat openai_api_server
python3 -m fastchat.serve.openai_api_server --host $host --port $port &
# start fastchat worker: dashinfer_worker
python3 dashinfer_worker.py \
    --model-path ${model_path} \
    ${model_revision:+"--revision ${model_revision}"} \
    ${conv_template:+"--conv-template ${conv_template}"} \
    ${dashinfer_config} &


# stop all services
pgid=$$
trap "pkill -P $pgid" SIGINT SIGTERM EXIT

wait


# clean logs
rm -f *.log
