#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    quantization.py
#
from enum import Enum, IntEnum


class QuantizeConfig:

    class QuantMode(Enum):
        Dynamic = 1
        A16W8 = 2

    def __init__(
        self,
        activation_type,
        weight_type=None,
        quan_json=None,
        op_type_to_quantize=None,
        extra_option=None,
    ):
        '''
        AllSpark Quantization Config.
        :param activation_type: "float16", "bfloat16", "int8", "uint8"
        :param weight_type: "int8", "uint8"
        :param op_type_to_quantize: ["GEMM",...]
        :param extra_option: {key:value}
            SubChannel = True/False:
                If you want to use SubChannel, please set True
            GroupSize = 512
                SubChannel`s GroupSize, Default 512. Supported 64, 128, 256, 512.
        '''
        # Support Quantize Op
        quantize_op_type = ["GEMM"]
        # Valid Extra Option Keys
        valid_extra_option = {
            "SubChannel": False,
            "GroupSize": 512,
        }
        #######################################
        self.activation_type = activation_type
        self.weight_type = weight_type or None
        self.op_type_to_quantize = op_type_to_quantize or quantize_op_type
        self.extra_option = extra_option or {}
        self.quantize_mode = None
        #######################################
        # Load From config.json
        if quan_json:
            if self.extra_option["AdaptedQuantMethod"] not in [None]:
                raise Exception("AdaptedQuantMethod only support [None]")
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
        # Check SubChannel GroupSize
        if self.extra_option["SubChannel"] and self.extra_option[
                "GroupSize"] not in [64, 128, 256, 512]:
            raise Exception(
                "SubChannel GroupSize only support [64, 128, 256, 512]")
        print(self.quantize_mode)

    def __repr__(self) -> str:
        return f"QuantizeConfig(mode={self.quantize_mode}, activation={self.activation_type}, weight={self.weight_type}, extra={self.extra_option})"
