'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    env.py
'''
from os import environ

BFC_ENV = {
    "BFC_ALLOCATOR": "ON",
    "BFC_MEM_RATIO": "0.76",
    "BFC_ALLOW_GROWTH": "OFF",
    "VL_PRECISION": "fp16",
    "VL_WORKER_NUMS": "1",
    "HIE_DISABLE_CUDNNV8_CONV": "1",
    "HIE_OPT_DISABLE_GLOBALLAYOUT": "1",
    "LOCAL_TEST": "1",
    "VL_LOCAL_ENABLE": "1",
    "VL_REDIS_ENABLE": "0",
    "VL_LOCAL_MAX_CACHE": "1024",
    "DS_LLM_MAX_IN_TOKENS": "30000",
    "VL_MAX_OUTPUT_LEN": "1024",
    # "QWEN_MODEL_TYPE": "QWEN2-AL",
}


def setenv() -> None:
    def dict_env_setter(kv: dict) -> None:
        for k, v in kv.items():
            if k not in environ.keys():
                environ[k] = v
        return

    dict_env_setter(BFC_ENV)
    return


def getenv(key, value):
    return environ.get(key, value)


# dashscope参数
DS_SVC_ID = getenv("DS_SVC_ID", "")
DS_SVC_NAME = getenv("DS_SVC_NAME", "")
