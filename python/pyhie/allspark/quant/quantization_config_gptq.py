'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    quantization_config_gptq.py
'''
from .quantization_config import QuantizationSettings
from .quantization_config import SubChannel, SignType


class GPTQSettings(QuantizationSettings):
    def __init__(self, quant_config: dict):
        super().__init__()
        assert(quant_config['quant_method'] == "gptq")
        self.hf_quant_config = quant_config
        self.group_size = quant_config.get("group_size", -1)

        self.weight_bits = quant_config["bits"]

        if "weight_sign" in quant_config:
            self.weight_sign_hf = quant_config.get("weight_sign")
            if self.weight_sign_hf:
                self.weight_sign = SignType.Signed
            else:
                self.weight_sign = SignType.Unsigned
        else:
            self.weight_sign = SignType.AUTO

        self.desc_act = quant_config.get("desc_act", False)
        if self.desc_act:
            raise NotImplementedError("not support desc act == True")
        self.sym = quant_config.get("sym", True)

        # add all supported feature into support table.
        # it's only means support status in model convert phase,
        # different device(like GPU,CPU) may have different support status.
        super().set_support_matrix_support("float16", "int8")
        super().set_support_matrix_support("bfloat16", "int8")
        super().set_support_matrix_support("float16", "uint4")
        super().set_support_matrix_support("bfloat16", "uint4")
        super().set_support_matrix_support("int8", "int8")
        # super().set_support_matrix_support("int8", "uint4")

        # gptq use dequant model to support sub-channel quant
        for grp in [-1, 32, 64, 128, 256, 512]:
            self.set_group_compute_matrix_support("activate_quant", grp)



    def get_support_status(self, activation_data_type: str, weight_data_type: str):
        """
        Get support status of quantization, return true if this quantization combo is supported.
        Args:
            activation_data_type:
            weight_data_type:

        Returns:

        """

        self.quant_compute_mode = "activate_quant" if activation_data_type in ["int8", "fp8_e4m3"] else "weight_only"

        return super().get_support_status(activation_data_type, weight_data_type)

    def get_name(self):
        return "gptq"

    def guess_weight_data_type(self):
        if self.weight_bits == 8:
            return "int8"
        elif self.weight_bits == 4:
            return "uint4"
        else:
            raise NotImplementedError(f"{self.weight_bits} is not supported")

    def to_as_quant_config(self,  activation_data_type: str, weight_data_type: str = None) -> "QuantizeConfig":
        """
        convert GPTQ setting to allspark's quant config.

        Args:
            activation_data_type: data type of activation side data type, eg, A8W8 use int8/uint8 as data type. option: [float16,bfloat16,int8]
            weight_data_type: string for weight data type, like [int8,int4,uint4]

        Returns: generate an allspark QuantConfig.
        """
        from ...allspark.quantization import QuantizeConfig

        if weight_data_type == None:
            weight_data_type = self.guess_weight_data_type()


        if not self.get_support_status(activation_data_type, weight_data_type):
            raise ValueError("quantization combo is not supported by engine.")


        as_quant_config = QuantizeConfig(activation_data_type, weight_data_type, self)
        return as_quant_config

    def get_group_num(self):
        return self.group_size

    def get_group_settings(self):
        if "group_settings" in self.hf_quant_config:
            return self.hf_quant_config["group_settings"]
        else:
            return None

    def get_weight_bits(self):
        return self.weight_bits

    def is_quant_symmetric(self) -> bool:
        return self.sym

    def get_subchannel_method(self):
        return SubChannel.ContinuousSplit
