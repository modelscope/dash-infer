'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    __init__.py
'''
from .engine import Engine
from .quant.quantization_config_gptq import GPTQSettings
from .quant.quantization_config_iq import IQSettings
from .quant.quantization_config import QuantizationSettings
from .runtime_config import AsModelRuntimeConfigBuilder
from .model_loader import ModelSerializerException
from .model_loader import HuggingFaceModel

from ._allspark import *

__all__ = [
    "AsStatus",
    "AsModelConfig",
    "ModelRuntimeConfig",
    "Engine",
    "GenerateRequestStatus",
    "MultiMediaInfo",
    "HuggingFaceModel",
    # functions
    "save_allsparky_dltensor_tofile",
    "set_global_header",
]

print("AllSpark python package start init.")
try:
    from .client import ClientEngine
    __all__.append("ClientEngine")
except ImportError:
    print("[Info] No Multi-NUMA support on CUDA Version.")

QUANT_CONFIG_MAP = {
    "gptq" : GPTQSettings,
    "iq" : IQSettings,
    "instant_quant": IQSettings

}

def get_quant_settings_cls(method : str, quant_cfg = None) -> QuantizationSettings :
    try:
        return QUANT_CONFIG_MAP[method](quant_cfg)
    except KeyError as e:
        print(e.with_traceback())
        raise NotImplementedError(f"not implement quant config method: {method}")
