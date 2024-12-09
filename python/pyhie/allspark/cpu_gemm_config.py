'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    cpu_gemm_config.py
'''


class CpuGemmConfig:
    from enum import Enum

    class GemmMode(Enum):
        WEIGHT_BIAS_BF16 = 1
        WEIGHT_BF16 = 2
        WEIGHT_BIAS_FP16 = 3
        WEIGHT_FP16 = 4

    def __init__(self, gemm_dict):
        '''
        AllSpark CPU Gemm Config. It is used to config GEMM weight and bias data type.
        :param gemm_dict:{ "weight_type": "float16/bfloat16", "bias_type":"None/float16/bfloat16"}
        '''
        #######################################
        self.weight_type = gemm_dict.get("weight_type")
        self.bias_type = gemm_dict.get("bias_type")
        self.gemm_mode = None
        #######################################

        if self.weight_type.upper() in ["FLOAT16", "BFLOAT16"]:
            if self.weight_type.upper() == "FLOAT16":
                if self.bias_type is not None:
                    if self.bias_type.upper() == "FLOAT16":
                        self.gemm_mode = CpuGemmConfig.GemmMode.WEIGHT_BIAS_FP16
                    else:
                        raise Exception(
                            "AllSpark does not support bias_type config")
                else:
                    self.gemm_mode = CpuGemmConfig.GemmMode.WEIGHT_FP16
            else:
                if self.bias_type is not None:
                    if self.bias_type.upper() == "BFLOAT16":
                        self.gemm_mode = CpuGemmConfig.GemmMode.WEIGHT_BIAS_BF16
                    else:
                        raise Exception(
                            "AllSpark does not support bias_type config")
                else:
                    self.gemm_mode = CpuGemmConfig.GemmMode.WEIGHT_BF16
        else:
            raise Exception(
                "AllSpark does not support your cpu gemm configuration.")
        print(self.gemm_mode)
