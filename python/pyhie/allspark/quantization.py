'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    quantization.py
'''
from enum import Enum, IntEnum
import re

from .quant.quantization_config import SubChannel


class QuantizeConfig:

    class QuantMode(Enum):
        # Dynamic = 1 # already deprecated
        A16W8 = 2
        A16W4 = 3
        A8W8  = 4
        FP8A8W8 = 5

    def load_GPTQ_config(self, quan_json):
        """ deprecated function. """
        bits = quan_json["bits"]
        if bits == 8:
            self.weight_type = "INT8"
            # maybe UINT8?
        elif bits == 4:
            self.weight_type = "UINT4"

        group_size = quan_json["group_size"]
        if (group_size == -1):
            self.extra_option["SubChannel"] = False
        else:
            self.extra_option["SubChannel"] = True
            self.extra_option["GroupSize"] = group_size
            # group_settings: first configure the special ones and then configure the general ones
            if "group_settings" in quan_json:
                self.extra_option["GroupSettings"] = {}
                group_settings = quan_json["group_settings"]
                for key, value in group_settings.items():
                    if not (isinstance(key, str) and isinstance(value, int)):
                        raise Exception("Here group_settings must be Dict of (str, int). Please check GPTQ quant config")
                    self.extra_option["GroupSettings"].update({key: value})

        if quan_json["desc_act"] and self.extra_option["SubChannel"]:
            raise Exception("GPTQ not support desc_act=True in SubChannel")

    def __init__(
        self,
        activation_type,
        weight_type,
        quant_settings: "QuantizationSettings" = None,
        quan_json=None,
        extra_option=None,
    ):
        """
        AllSpark quant config from user pass to model, don't use this function to config quantization related config,
        use QuantizationSettings.

        The quantization will convert weight type to activation type and call inference device do calculation,
        the lower activation bits, the higher compute power.

        Args:
            activation_type: for weight only quant, it's the activation data type, like A16W8 can select bfloat16 or
                             float16, this value should align with runtime_cfg 's data type in serialized model.
                             option : [float16, bfloat16]
            quant_settings:  quantization settings object, new API, see @.allspark.QuantizationSettings document

            weight_type: data type of weight [int8, uint8, int4, uint4]
            quan_json: [deprecated] : huggingface quantization_config json.
            extra_option:  [deprecated] detail setting for IQ quantization, it's a python dict:

              SubChannel = True/False:
                If you want to use SubChannel, please set True
              GroupSize = 512
                SubChannel`s GroupSize, Default 512. Supported 64, 128, 256, 512.
              GroupSettings = None or Dict(key=re, value=groupsize)
                Use multiple regular expressions to configure GroupSize of different layers.
                First configure the special ones and then configure the general ones. Such as {r"\.c_proj.weight": 32, r".*": 128}.
                If Groupsettings is set, the GroupSize parameter is ignored
              AdaptedQuantMethod
                [None, "GPTQ", "AWQ", "SQUEEZELLM", ...]

        """

        # Support Quantize Op
        quantize_op_type = ["GEMM", "GEMMCAPSULE"]
        # Valid Extra Option Keys

        #######################################
        self.op_type_to_quantize = quantize_op_type
        self.activation_type = activation_type
        self.weight_type = weight_type or None
        self.extra_option = extra_option or {}
        self.quantize_mode = None
        #######################################

        if quant_settings is None:
            self.init_with_config_derecated(quan_json, quantize_op_type)
            return
        # use new setting to init.

        self.weight_type = str(weight_type).upper()
        self.activation_type = str(activation_type).upper()

        ## this sub-channel means average continues split(calculate scale in same stride)

        # TODO: add activation data type.
        if quant_settings.get_subchannel_method() == SubChannel.ContinuousSplit:
            self.extra_option["SubChannel"] = quant_settings.get_group_num() != -1
            self.extra_option["GroupSize"] = quant_settings.get_group_num()
        else:
            raise NotImplementedError("TP Split not supported by now, will add later.")

        self.extra_option["AdaptedQuantMethod"] = (
            None if quant_settings.get_name() == "instant_quant"
            else quant_settings.get_name().upper()
        )

        self.extra_option["GroupSettings"] = quant_settings.get_group_settings()

        if self.weight_type == self.activation_type == "INT8":
            self.quantize_mode = QuantizeConfig.QuantMode.A8W8
        elif self.weight_type == self.activation_type == "FP8_E4M3":
            self.quantize_mode = QuantizeConfig.QuantMode.FP8A8W8
        elif self.weight_type in ["INT8", "UINT8"] and self.activation_type in ["BFLOAT16", "FLOAT16"]:
            self.quantize_mode = QuantizeConfig.QuantMode.A16W8
        elif self.weight_type == "UINT4" and self.activation_type in ["BFLOAT16", "FLOAT16"]:
            self.quantize_mode = QuantizeConfig.QuantMode.A16W4
        else:
            raise NotImplementedError(
                f"Not implemented mode: activate:{self.activation_type}, weight:{self.weight_type}")



        # TODO: if some sub-channel kernel not supported, convert them into per-channel with TP-Split.from
        # weight process will be in model's init function

    def init_with_config_derecated(self, quan_json, quantize_op_type):
        valid_extra_option = {
            "SubChannel": False,
            "GroupSize": 512,
            "GroupSettings": None,
            "AdaptedQuantMethod": None,  # [None, "GPTQ", "AWQ", "SQUEEZELLM", ...]
        }
        if quan_json:
            if self.extra_option["AdaptedQuantMethod"] not in [None, "GPTQ"]:
                raise Exception(
                    "AdaptedQuantMethod only support [None, 'GPTQ']")
            if self.extra_option["AdaptedQuantMethod"] == "GPTQ":
                self.load_GPTQ_config(quan_json)
            # TODO other method
        # Check OpType
        for idx, op in enumerate(self.op_type_to_quantize):
            op = op.upper()
            self.op_type_to_quantize[idx] = op
            if op not in quantize_op_type:
                raise Exception(
                    "The [{}] Op does not support quantization.".format(op))
            else:
                print("Quantize Op [{}]".format(op))
        if self.activation_type.upper() in [
            "FLOAT16", "BFLOAT16"
        ] and self.weight_type.upper() in ["UINT8", "INT8"]:
            self.quantize_mode = QuantizeConfig.QuantMode.A16W8
        elif self.activation_type.upper() == "INT8" and self.weight_type.upper(
        ) == "INT8":
            self.quantize_mode = QuantizeConfig.QuantMode.A8W8
        elif self.activation_type.upper() in [
            "FLOAT16", "BFLOAT16"
        ] and self.weight_type.upper() in ["UINT4"]:
            self.quantize_mode = QuantizeConfig.QuantMode.A16W4
        else:
            raise Exception(
                "AllSpark does not support your quantized configuration.")
        # Check Extra Option
        for key, val in self.extra_option.items():
            if key not in valid_extra_option.keys():
                raise Exception(
                    "extra_option [{}] key is invalid.".format(key))
        for key, default_val in valid_extra_option.items():
            if key not in self.extra_option.keys():
                self.extra_option[key] = default_val

        # check if group setting for sub-channel continuous mode is valid.
        self.check_group_settings()

        if self.extra_option["AdaptedQuantMethod"] not in [None, "GPTQ", "GPTQ_NO_PACK"]:
            raise Exception("AdaptedQuantMethod only support [None, 'GPTQ', GPTQ_NO_PACK']")

    def check_group_settings(self):
        # Check SubChannel GroupSize
        if self.extra_option["SubChannel"] and self.extra_option[
            "GroupSize"] not in [64, 128, 256, 512]:
            raise Exception(
                "SubChannel GroupSize only support [64, 128, 256, 512]")
        if self.extra_option["SubChannel"] and self.extra_option[
            "GroupSettings"] is not None:
            for key, val in self.extra_option["GroupSettings"].items():
                pattern = re.compile(key)
                if not isinstance(pattern, re.Pattern):
                    raise Exception(
                        "Key of GroupSettings must be regex, but got {}.".format(key))
                if not isinstance(val, int) or val % 32 != 0:
                    raise Exception(
                        "SubChannel GroupSize now only support multiples of 32, bur got {}.".format(val))

    def __repr__(self) -> str:
        return f"QuantizeConfig(mode={self.quantize_mode}, activation={self.activation_type}, weight={self.weight_type}, extra={self.extra_option})"

    # def init_from_json(self, quan_json):
    # if self.extra_option["AdaptedQuantMethod"] not in [None, "GPTQ"]:
    #     raise Exception("AdaptedQuantMethod only support [None, 'GPTQ']")
    # if self.extra_option["AdaptedQuantMethod"] == "GPTQ":
    #     bits = quan_json["bits"]
    #     if bits == 8:
    #         self.weight_type = "INT8"
    #         # maybe UINT8?
    #     elif bits == 4:
    #         self.weight_type = "UINT4"
    #     group_size = quan_json["group_size"]
    #     if (group_size == -1):
    #         self.extra_option["SubChannel"] = False
    #     else:


    #     if self.activation_type.upper() in ["FLOAT16", "BFLOAT16"] and self.weight_type.upper() in ["UINT8", "INT8"]:
    #         self.quantize_mode = QuantizeConfig.QuantMode.A16W8
    #     elif self.activation_type.upper() == "INT8" and self.weight_type.upper() == "INT8":
    #         self.quantize_mode = QuantizeConfig.QuantMode.Dynamic
class KVCacheConfig:
    ''' Allspark KV Cache Config and store params for engine-build and generation.
        constructor params kv_cache_type, currently support only "int8" and "uint4"
    '''

    class QuantType(IntEnum):
        INT8 = 0
        UINT4 = 1

    def __init__(self, kv_cache_type: str = "int8") -> None:
        '''
        AllSpark KV Cache Config.
        :param kv_cache_type: "int8", "uint4"
        '''
        self.quantize_type = None
        self.quantize_type_alias = kv_cache_type
        if kv_cache_type.upper() == "INT8":
            self.quantize_type = KVCacheConfig.QuantType.INT8
        elif kv_cache_type.upper() == "UINT4":
            self.quantize_type = KVCacheConfig.QuantType.UINT4
        else:
            raise Exception(
                f"AllSpark does not support incoming kv-cache {kv_cache_type} configuration."
            )
        return

    def __repr__(self) -> str:
        return f"KVCacheConfig({self.quantize_type_alias})"
