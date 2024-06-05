#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    basic_example_chatglm4_dimodel_simple.py
#
import copy
import random

from modelscope import snapshot_download
from dashinfer.helper import EngineHelper, ConfigManager

model_path = snapshot_download("dash-infer/glm-4-9b-chat-DI")

config_file = model_path + "/" + "di_config.json"
config = ConfigManager.get_config_from_json(config_file)
config["model_path"] = model_path

## init EngineHelper class
engine_helper = EngineHelper(config)
engine_helper.verbose = True
engine_helper.init_tokenizer(model_path)

## init engine
engine_helper.init_engine()

## prepare inputs and generation configs
user_input = "浙江的省会在哪"
prompt = "[gMASK] <sop> " + "<|user|>\n" + user_input + "<|assistant|>\n"
gen_cfg = copy.deepcopy(engine_helper.default_gen_cfg)
gen_cfg["seed"] = random.randint(0, 10000)
request_list = engine_helper.create_request([prompt], [gen_cfg])

## inference
engine_helper.process_one_request(request_list[0])
engine_helper.print_inference_result_all(request_list)

engine_helper.uninit_engine()
