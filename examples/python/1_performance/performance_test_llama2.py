#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    performance_test_llama2.py
#
import os
import sys
import copy
import time
import queue
import random
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse

from dashinfer.helper import EngineHelper, ConfigManager


def download_model(model_id, revision, source="modelscope"):
    print(f"Downloading model {model_id} (revision: {revision}) from {source}")
    if source == "modelscope":
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id, revision=revision)
    elif source == "huggingface":
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_id)
    else:
        raise ValueError("Unknown source")

    print(f"Save model to path {model_dir}")

    return model_dir


def create_random_prompt(batch_size, input_len, default_gen_cfg=None):
    prompt_list = np.random.randint(low=1, high=5000, size=(batch_size, input_len)).astype(np.int64).tolist()

    gen_cfg_list = []
    for i in range(len(prompt_list)):
        if default_gen_cfg != None:
            gen_cfg = copy.deepcopy(default_gen_cfg)
            gen_cfg["seed"] = random.randint(0, 10000)
            gen_cfg_list.append(gen_cfg)

    return prompt_list, gen_cfg_list


def process_request(request_list, engine_helper: EngineHelper):

    def done_callback(future):
        request = future.argument
        future.result()
        engine_helper.print_inference_result(request)

    # create a threadpool
    executor = ThreadPoolExecutor(
        max_workers=engine_helper.engine_config["engine_max_batch"])

    try:
        # submit all tasks to the threadpool
        futures = []
        for request in request_list:
            future = executor.submit(engine_helper.process_one_request, request)
            future.argument = request
            future.add_done_callback(done_callback)
            futures.append(future)
    finally:
        executor.shutdown(wait=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='config_llama2_7b.json')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--multinode_mode', action='store_true')

    args = parser.parse_args()

    config_file = "../model_config/" + args.config_file

    config = ConfigManager.get_config_from_json(config_file)
    config["generation_config"]["early_stopping"] = False
    config["generation_config"]["stop_words_ids"] = []
    config["device_ids"] = args.device_ids
    config["multinode_mode"] = args.multinode_mode

    cmd = f"pip show dashinfer | grep 'Location' | cut -d ' ' -f 2"
    package_location = subprocess.run(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      shell=True,
                                      text=True)
    package_location = package_location.stdout.strip()
    os.environ["AS_DAEMON_PATH"] = package_location + "/dashinfer/allspark/bin"
    os.environ["AS_NUMA_NUM"] = str(len(config["device_ids"]))
    os.environ["AS_NUMA_OFFSET"] = str(config["device_ids"][0])

    '''
    # Turn on the following options to get operator-level profiling results
    config["engine_config"]["do_profiling"] = True
    os.environ["AS_PROFILE"] = "ON"
    '''

    ## download original model
    ## download model from huggingface
    original_model = {
        "source": "modelscope",
        "model_id": "modelscope/Llama-2-7b-chat-ms",
        "revision": "v1.0.5",
        "model_path": ""
    }
    original_model["model_path"] = download_model(original_model["model_id"],
                                                  original_model["revision"],
                                                  original_model["source"])

    ## init EngineHelper class
    engine_helper = EngineHelper(config)
    engine_helper.verbose = True
    engine_helper.init_tokenizer(original_model["model_path"])

    ## convert huggingface model to dashinfer model
    ## only one conversion is required
    if engine_helper.check_model_exist() == False:
        engine_helper.convert_model(original_model["model_path"])

    batch_size_list = [1, 2, 4, 8]
    output_len_list = [128]
    input_len_list = [128, 1200]

    for output_len in output_len_list:
        for input_len in input_len_list:
            for batch_size in batch_size_list:
                print(f"### batch_size: {batch_size}, output_len: {output_len}, input_len: {input_len}")
                sys.stdout.flush()

                engine_helper.init_engine()
                engine_helper.verbose = False

                engine_helper.default_gen_cfg["max_length"] = output_len + input_len
                prompt_list, gen_cfg_list = create_random_prompt(
                    batch_size, input_len, engine_helper.default_gen_cfg)
                request_list = engine_helper.create_request(prompt_list, gen_cfg_list)

                global_start = time.time()
                process_request(request_list, engine_helper)
                global_end = time.time()

                total_timecost = global_end - global_start
                engine_helper.print_profiling_data(request_list, total_timecost)
                print(f"total timecost: {total_timecost} s")

                engine_helper.uninit_engine()
