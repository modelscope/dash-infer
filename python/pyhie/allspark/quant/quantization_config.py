'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    quantization_config.py
'''
from enum import Enum
from collections import defaultdict



class SubChannel(Enum):
    """ SubChannel method  """
    TPSplit = "TPSplit",
    ContinuousSplit = "CPContinuousSplit",


class SignType(Enum):
    """ The Sign flag for quantization weight, like 8bit have sign 8bit [-127,128], unsigned int8 [0,255]
        and also can auto select by settings.
    """
    AUTO = "AutoSign"
    Signed = "Signed"
    Unsigned = "Unsigned"
class QuantComputeMode:
    """
    quant compute mode:
    #  - weight_only: means the activation use fp16 and weight stored by `quant_format`
    #  - activate_quant:  means use compute quantization, the activation and compute use the `quant_format`, **if choose compute, only `-1` is supported in group_size setting**
    compute_method: weight_only # choose between ["weight_only", "activate_quant"]
    """
    WeightOnly = "weight_only"
    ActivateQuant = "activate_quant"


class QuantizationSettings:
    """
    The helper class to parse the quantization config for different quantization method, like gptq and Allspark's IQ
    """

    def __init__(self):
        self.group_size = -1
        self.weight_bits = 8
        self.desc_act = False
        self.sym = True
        self.quant_compute_mode = "weight_only"

        self.activate_dtypes = ['int8', 'uint8', 'float16', 'bfloat16']
        self.weight_dtypes = ['int8', 'uint8', 'int4', 'uint4']
        self.support_matrix = defaultdict(lambda: defaultdict(bool))

        self.sub_group_list = [-1, 32, 64, 128, 256, 512]
        self.compute_method_list = ["weight_only", "activate_quant"]

        self.group_compute_support_matrix = defaultdict(lambda: defaultdict(bool))

        # some group size not support by some compute mode.
        for grp in [-1, 32, 64, 128, 256, 512]:
            self.set_group_compute_matrix_support("weight_only", grp)

        self.set_group_compute_matrix_support("activate_quant", -1)


    def set_support_matrix_support(self, activate_dtype, weight_dtype):
        self.support_matrix[activate_dtype][weight_dtype] = True

    def set_group_compute_matrix_support(self, compute_method, sub_group):
        self.group_compute_support_matrix[compute_method][sub_group] = True

    def get_support_status(self, activate_dtype: str, weight_dtype: str) -> bool:

        weight_type_support = self.support_matrix[activate_dtype][weight_dtype]
        group_compute_support = self.group_compute_support_matrix[self.quant_compute_mode][self.group_size]

        ret = weight_type_support and group_compute_support
        if not group_compute_support:
            print(f"group and compute is not supported ({self.quant_compute_mode},{self.group_size}), "
                  f"supported combo is {self.group_compute_support_matrix}")
        if not weight_type_support:
            print(f"type support is not support for ({activate_dtype},{weight_dtype}), "
                  f"supported matrix is {self.support_matrix}")

        print(f"Quant support status {ret}, because weight:{weight_type_support} group: {group_compute_support}")

        return ret

    def __str__(self):
        return (f"{self.__class__.__name__} (group_size={self.group_size}, "
                f"weight_bits={self.weight_bits}, "
                f"desc_act={self.desc_act}, "
                f"sym={self.sym})")

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_hf_config(config_dict: dict) -> "QuantizationSettings":
        from ...allspark import get_quant_settings_cls
        return get_quant_settings_cls(config_dict['quant_method'], config_dict)

    def to_as_quant_config(self, activation_data_type: str, weight_data_type: str = None) -> "QuantizeConfig":
        """
        generate this setting to an allspark engine's QuantizationConfig class.
        Args:
            weight_data_type:
            activation_data_type:

        Returns:

        """
        raise NotImplementedError("no impl.")

    def get_name(self):
        """
        Quantization method name, like "gptq"
        """
        raise NotImplementedError("no impl.")

    def get_weight_bits(self):
        """
        Quantization weight bits number, 4 bit return 4
        """
        raise NotImplementedError("no impl.")

    def is_quant_symmetric(self) -> bool:
        """
        If the quant is symmetric
        """
        raise NotImplementedError("no impl")

    def get_group_settings(self) -> dict:
        """
        For different gemm, use different group size , express by a dict.

        Returns: GroupSettings = None or Dict(key=re, value=groupsize)
                Use multiple regular expressions to configure GroupSize of different layers.
                First configure the special ones and then configure the general ones. Such as {r"\.c_proj.weight": 32, r".*": 128}.
                If Groupsettings is set, the GroupSize parameter is ignored
        """
        raise NotImplementedError("no impl")


    def get_subchannel_method(self):
        """
        Get SubChannel method,

        TPSplit  means split channel by device(s) number, aka, Tensor Parallel number, like
           8 TP, split the full channel(token) into 8 Channel, still per-channel in each device.
        ContinuesSplit means split channel by continuous tokens into one group and calculate one quant scale parameter.
        Returns:

        """

        raise NotImplementedError("no impl")

    def get_activate_data_setting(self):
        raise NotImplementedError("no imp.")

    def get_group_num(self):
        """
        Get group number if you use ContinuesSplit quant.
        Returns:

        """
        raise NotImplementedError("no imp.")
    def guess_weight_data_type(self):
        raise NotImplementedError("no imp.")

