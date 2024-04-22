#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    basic_example_qwen_v10.py
#
import os
import sys
import copy
import time
import queue
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

from dashinfer.helper import EngineHelper


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


def create_test_prompt(default_gen_cfg=None):
    prompt_list = [
        "浙江的省会在哪",
        "Where is the capital of Zhejiang?",
        "将“温故而知新”翻译成英文，并解释其含义",
    ]
    gen_cfg_list = []
    for i in range(len(prompt_list)):
        prompt_list[i] = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" \
                         + prompt_list[i] + "<|im_end|>\n<|im_start|>assistant\n"
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

    # submit all tasks to the threadpool
    futures = []
    for request in request_list:
        future = executor.submit(engine_helper.process_one_request, request)
        future.argument = request
        future.add_done_callback(done_callback)
        futures.append(future)

    # wait, until all tasks finish
    for future in futures:
        future.result()

    return


if __name__ == '__main__':
    config_file = "../model_config/config_qwen_v10_1_8b.json"
    config = EngineHelper.get_config_from_json(config_file)

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

    ## download original model
    ## download model from huggingface
    # original_model = {
    #     "source": "huggingface",
    #     "model_id": "Qwen/Qwen-1_8B-Chat",
    #     "revision": "",
    #     "model_path": ""
    # }

    ## download model from modelscope
    original_model = {
        "source": "modelscope",
        "model_id": "qwen/Qwen-1_8B-Chat",
        "revision": "v1.0.0",
        "model_path": ""
    }
    original_model["model_path"] = download_model(original_model["model_id"],
                                                  original_model["revision"],
                                                  original_model["source"])

    ## init EngineHelper class
    engine_helper = EngineHelper(config)
    engine_helper.verbose = True
    engine_helper.init_tokenizer(original_model["model_path"])
    engine_helper.init_torch_model(original_model["model_path"])

    ## convert huggingface model to dashinfer model
    ## only one conversion is required
    engine_helper.convert_model(original_model["model_path"])

    ## inference
    engine_helper.init_engine()

    prompt_list, gen_cfg_list = create_test_prompt(
        engine_helper.default_gen_cfg)
    request_list = engine_helper.create_request(prompt_list, gen_cfg_list)

    global_start = time.time()
    process_request(request_list, engine_helper)
    global_end = time.time()

    total_timecost = global_end - global_start
    # engine_helper.print_inference_result_all(request_list)
    engine_helper.print_profiling_data(request_list, total_timecost)
    print(f"total timecost: {total_timecost} s")

    engine_helper.uninit_engine()
