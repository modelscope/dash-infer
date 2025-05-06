'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model_config.py
'''
from transformers import Qwen2Config, Qwen2MoeConfig, LlamaConfig

import torch

from .model.qwen_v1_config import QWenConfig


def torch_dtype_to_as_dtype(in_dtype: torch.dtype):
    if in_dtype == torch.float16:
        return "float16"
    elif in_dtype == torch.bfloat16:
        return "bfloat16"
    elif in_dtype == torch.float32 or in_dtype == torch.float:
        return "float32"
    else:
        raise ValueError(f"not support dtype: {in_dtype}")


class ModelConfigAdapter:
    model_config = {}

    def get_as_model_config(
        self,
        rope_scale_method: "RoPEScaleMethod" = None,
        rope_scale_parameter: dict = None,
    ):
        from .engine import RoPEScaleMethod
        # FIXME: currently only support rope scaling from HF
        # if rope_scale_method != None or rope_scale_method != RoPEScaleMethod.NO_SCALE:
        #     self.model_config["rope_scaling"] = rope_scale_parameter
        return self.model_config

    def get_model_data_type(self) -> str:
        raise ValueError("not impl.")

class QWenConfigAdapter(ModelConfigAdapter):
    def __init__(self, hf_model_config: QWenConfig):
        self.model_config = hf_model_config.__dict__
        self.model_config["model_type"] = "Qwen_v10"
        self.model_config["architectures"] = hf_model_config.architectures
        hidden_size_per_head = int(
            self.model_config["hidden_size"] / self.model_config["num_attention_heads"]
        )
        self.model_config["size_per_head"] = hidden_size_per_head
        self.model_config["eos_token_id"] = hf_model_config.eos_token_id

        # if (hf_model_config.bf16 == True):
        #     self.dtype = "bfloat16"
        # elif hf_model_config.fp16:
        #     self.model_dtype = "float16"
        # elif hf_model_config.fp32:
        #     self.model_dtype = "float32"
        # else:
        #     print("config.json not setup data type correctly, use bfloat16 as model data type")
        #     self.model_dtype = "bfloat16"
        self.model_dtype = hf_model_config.torch_dtype
        if self.model_dtype == None:
            print(
                "config.json not setup data type correctly, use bfloat16 as model data type"
            )
            self.model_dtype = "bfloat16"

    def get_model_data_type(self):
        return self.model_dtype


class QWen2ConfigAdapter(ModelConfigAdapter):
    def __init__(self, hf_model_config: Qwen2Config):
        self.model_config = hf_model_config.__dict__
        self.model_config["model_type"] = "Qwen_v20"
        self.model_config["architectures"] = hf_model_config.architectures

        self.model_config["rotary_emb_base"] = hf_model_config.rope_theta

        hidden_size_per_head = int(
            hf_model_config.hidden_size / hf_model_config.num_attention_heads
        )

        self.model_config["size_per_head"] = hidden_size_per_head

        self.model_dtype = torch_dtype_to_as_dtype(hf_model_config.torch_dtype)
        if self.model_dtype == None:
            print(
                "config.json not setup data type correctly, use bfloat16 as model data type"
            )
            self.model_dtype = "bfloat16"

        print("model config:")
        print(self.model_config)

    def get_model_data_type(self):
        return self.model_dtype

class QWen3ConfigAdapter(ModelConfigAdapter):
    def __init__(self, hf_model_config: Qwen2Config):
        self.model_config = hf_model_config.__dict__
        self.model_config["model_type"] = "Qwen_v30"
        self.model_config["architectures"] = hf_model_config.architectures

        self.model_config["rotary_emb_base"] = hf_model_config.rope_theta

        hidden_size_per_head = int(
            hf_model_config.hidden_size / hf_model_config.num_attention_heads
        )

        self.model_config["size_per_head"] = hidden_size_per_head

        self.model_dtype = torch_dtype_to_as_dtype(hf_model_config.torch_dtype)
        if self.model_dtype == None:
            print(
                "config.json not setup data type correctly, use bfloat16 as model data type"
            )
            self.model_dtype = "bfloat16"

        print("model config:")
        print(self.model_config)

    def get_model_data_type(self):
        return self.model_dtype


class Qwen2MoeConfigAdapter(ModelConfigAdapter):
    def __init__(self, hf_model_config: Qwen2MoeConfig):
        self.model_config = hf_model_config.__dict__
        self.model_config["model_type"] = "Qwen_v20_MOE"

        self.model_config["rotary_emb_base"] = hf_model_config.rope_theta

        hidden_size_per_head = int(
            hf_model_config.hidden_size / hf_model_config.num_attention_heads
        )

        self.model_config["size_per_head"] = hidden_size_per_head

        self.model_dtype = torch_dtype_to_as_dtype(hf_model_config.torch_dtype)
        if self.model_dtype == None:
            print(
                "config.json not setup data type correctly, use bfloat16 as model data type"
            )
            self.model_dtype = "bfloat16"

        print("model config:")
        print(self.model_config)

    def get_model_data_type(self):
        return self.model_dtype


class LlamaConfigAdapter(ModelConfigAdapter):
    def __init__(self, hf_model_config: LlamaConfig):
        self.model_config = hf_model_config.__dict__
        self.model_config["architectures"] = hf_model_config.architectures

        self.model_config["model_type"] = "LLaMA_v2"

        self.model_config["rotary_emb_base"] = hf_model_config.rope_theta

        hidden_size_per_head = int(
            hf_model_config.hidden_size / hf_model_config.num_attention_heads
        )

        self.model_config["size_per_head"] = hidden_size_per_head

        # GQA: num_key_value_heads
        self.model_config["num_key_value_heads"] = hf_model_config.num_key_value_heads
        self.model_dtype = torch_dtype_to_as_dtype(hf_model_config.torch_dtype)

    def get_model_data_type(self):
        return self.model_dtype


class ChatGLMConfigAdapter(ModelConfigAdapter):
    def __init__(self, hf_model_config):
        self.model_config = hf_model_config.__dict__
        self.model_config["architectures"] = hf_model_config.architectures

        if hasattr(hf_model_config, "original_rope"):
            self.model_config["model_type"] = "ChatGLM_v3"
        else:
            self.model_config["model_type"] = "ChatGLM_v2"

        self.model_config["rotary_emb_base"] = 10000

        hidden_size_per_head = int(
            hf_model_config.hidden_size / hf_model_config.num_attention_heads
        )

        self.model_config["size_per_head"] = hidden_size_per_head
        self.model_config["eos_token_id"] = hf_model_config.eos_token_id
        self.model_dtype = torch_dtype_to_as_dtype(self.model_config["torch_dtype"])

        # GQA: num_key_value_heads
        if hf_model_config.multi_query_attention:
            self.model_config["num_key_value_heads"] = (
                hf_model_config.num_attention_heads
                / hf_model_config.multi_query_group_num
            )
        else:
            self.model_config["num_key_value_heads"] = (
                hf_model_config.num_attention_heads
            )

        self.model_config["max_position_embeddings"] = hf_model_config.seq_length
        print("model config:")
        print(self.model_config)

    def get_model_data_type(self):
        return self.model_dtype


class ModelAdapterFactory:
    @staticmethod
    def create_adapter(model_config_instance):
        adapters = {
            "QWenConfig": QWenConfigAdapter,
            "Qwen2Config": QWen2ConfigAdapter,
            "Qwen3Config": QWen3ConfigAdapter,
            "Qwen2MoeConfig": Qwen2MoeConfigAdapter,
            "Qwen2VLForConditionalGeneration": QWen2ConfigAdapter,
            "LlamaConfig": LlamaConfigAdapter,
            "ChatGLMConfig": ChatGLMConfigAdapter,
        }
        class_name = model_config_instance.__class__.__name__
        adapter_class = adapters.get(class_name)
        if adapter_class:
            return adapter_class(model_config_instance)
        else:
            raise ValueError(f"No adapter found for Model: {class_name}")


# as_model_config = ModelAdapterFactory.create_adapter_for_hf_model_config(Config).get_as_model_config()
