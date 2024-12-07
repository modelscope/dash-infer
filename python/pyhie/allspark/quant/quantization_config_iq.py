'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    quantization_config_iq.py
'''
from .quantization_config import QuantizationSettings
from .quantization_config import SubChannel, SignType


class IQSettings(QuantizationSettings):
    """ Instant Quant Config"""

    def get_name(self):
        return "instant_quant"

    def __init__(self, quant_config: dict):
        super().__init__()
        assert (quant_config['quant_method'] == "instant_quant")
        self.di_quant_config = quant_config
        self.group_size = quant_config.get("group_size", -1)
        if "weight_format" not in quant_config:
            raise ValueError(
                "weight_format is missing in DIConfig config, this parameter is required, try weight_format: int8")

        signed_format = ["int8", "int4", "fp8_e4m3"]
        unsigned_format = ["uint8", "uint4"]


        full_format_support_list = signed_format + unsigned_format
        self.weight_format = quant_config.get("weight_format")
        if self.weight_format not in full_format_support_list:
            raise ValueError(f"IQSettings:Quant: not supported format: {self.weight_format}, "
                             f"supported format are: {full_format_support_list}")

        if self.weight_format in signed_format:
            self.weight_sign = SignType.Signed
        else:
            self.weight_sign = SignType.Unsigned

        eight_bit_formats = ["int8", "uint8", "fp8_e4m3"]
        four_bit_formats = ["uint4", "int4"]

        self.weight_bits = 8 if self.weight_format in eight_bit_formats else 4

        self.quant_compute_mode = quant_config.get("compute_method", "weight_only")
            
        self.group_settings = quant_config.get("group_settings", None)

        # most weight only kernel support asymmetric quantization
        # but activate kernel currently don't support asymmetric quantization.
        self.sym = False if self.quant_compute_mode == "weight_only" else True

        activate_formats = ["int8", "fp8_e4m3"]
        if self.quant_compute_mode == "activate_quant" and self.weight_format not in activate_formats:
            raise ValueError(
                f"activate_quant compute model now only support int8 and fp8_e4m3 weight_format,"
                f"but got weight_format : {self.weight_format}")
        self.activate_type = quant_config.get("activate_format", "int8")

        # add all supported feature into support table.
        # it's only means support status in model convert phase,
        # different device(like GPU,CPU) may have different support status.
        super().set_support_matrix_support("float16", "int8")
        # TODO: should fix this bfloat16 + int8， also bfloat + int8  only support per-channel.
        super().set_support_matrix_support("bfloat16", "int8")

        super().set_support_matrix_support("fp8_e4m3", "fp8_e4m3")

        super().set_support_matrix_support("float16", "uint4")
        super().set_support_matrix_support("bfloat16", "uint4")
        super().set_support_matrix_support("int8", "int8")
        super().set_support_matrix_support("int8", "uint4")


    def get_support_status(self, activation_data_type: str, weight_data_type: str):
        """
        Get support status of quantization, return true if this quantization combo is supported.
        Args:
            activation_data_type:
            weight_data_type:

        Returns:

        """
        return super().get_support_status(activation_data_type, weight_data_type)

    def guess_weight_data_type(self):
        if self.weight_bits == 8:
            return "int8"
        elif self.weight_bits == 4:
            return "uint4"
        else:
            raise NotImplementedError(f"{self.weight_bits} is not supported")

    def get_activate_data_setting(self):
        return self.activate_type

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
        return self.group_settings

    def get_weight_bits(self):
        return self.weight_bits

    def is_quant_symmetric(self) -> bool:
        return self.sym

    def get_subchannel_method(self):
        return SubChannel.ContinuousSplit
