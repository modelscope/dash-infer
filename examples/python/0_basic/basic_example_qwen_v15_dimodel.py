#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    basic_example_qwen_v15_dimodel.py
#
import os
import copy
import time
import random
import argparse
import subprocess
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor

from modelscope import snapshot_download
from dashinfer.helper import EngineHelper, ConfigManager


def create_test_prompt(default_gen_cfg=None):
    input_list = [
        "浙江的省会在哪",
        "Where is the capital of Zhejiang?",
        "将“温故而知新”翻译成英文，并解释其含义",
    ]

    start_text = "<|im_start|>"
    end_text = "<|im_end|>"
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {"role": "user", "content": ""}
    assistant_msg = {"role": "assistant", "content": ""}

    prompt_template = Template(
        "{{start_text}}" + "{{system_role}}\n" + "{{system_content}}" + "{{end_text}}\n" +
        "{{start_text}}" + "{{user_role}}\n" + "{{user_content}}" + "{{end_text}}\n" +
        "{{start_text}}" + "{{assistant_role}}\n")

    gen_cfg_list = []
    prompt_list = []
    for i in range(len(input_list)):
        user_msg["content"] = input_list[i]
        prompt = prompt_template.render(start_text=start_text, end_text=end_text,
                                        system_role=system_msg["role"], system_content=system_msg["content"],
                                        user_role=user_msg["role"], user_content=user_msg["content"],
                                        assistant_role=assistant_msg["role"])
        prompt_list.append(prompt)
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
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()

    model_path = snapshot_download("dash-infer/Qwen1.5-1.8B-Chat-DI")

    config_file = model_path + "/" + "di_config.json"
    config = ConfigManager.get_config_from_json(config_file)
    config["convert_config"]["do_dynamic_quantize_convert"] = args.quantize
    config["model_path"] = model_path

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

    ## init EngineHelper class
    engine_helper = EngineHelper(config)
    engine_helper.verbose = True
    engine_helper.init_tokenizer(model_path)

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
