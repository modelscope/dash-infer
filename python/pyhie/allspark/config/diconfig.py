'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    diconfig.py
'''
# DIConfig (DashInfer) related config functions, include DIConfig builder

import ruamel.yaml


from .._allspark import AsModelConfig, AsCacheMode, AsEvictionStrategy

from ..runtime_config import AsModelRuntimeConfigBuilder

model_comments = """\
model meta info part, usually imported from huggingface config.json
# huggingface_id:  optional, huggingface id can use to tokenizer etc. 
# modelscope_id:  optional, modelscope id tokenizer id
   """

model_comments_dict = {"huggingface_id": "optional, huggingface id can use to tokenizer etc. ",
                        "modelscope_id": "optional, modelscope id tokenizer id"}

runtime_cfg_comments_dict = {
    "model_name":                   "the model's ID in engine",

    "compute_unit":                 """\
    device_type: device type can be [CUDA, CPU, CPU_NUMA],  
               device_ids: # for CUDA, the len(device_ids) is card number, and compute_thread_in_device can be ignored. for CPU,  the len(device_ids) is ignored, for CPU_NUMA, the device_ids is NUMA id's, and len(device_ids) is NUMA Count
               compute_thread_in_device only works for CPU
     if device_type is CPU:  compute_thread_in_device is how many compute thread when inference,
        suggest value is physical core number(without hyper-thread), or you can pass 0 to let autodetect.
     if device_type is CPU_NUMA:  compute_thread_in_device means compute thread inside NUMA,
        suggest value is physical core number(without hyper-thread), or you can pass 0 to let autodetect.
    """,

    "engine_max_length":                "max length of engine support: for input+output, will allocate resource and warm up by this length.",

    "engine_max_batch":                 "max batch or concurrency supported by this engine, will reject the request if this size meets.",

    "kv_cache_mode":                    """kv-cache mode, choose between : [AsCacheDefault, AsCacheQuantI8, AsCacheQuantU4]
        which means :
   - AsCacheDefault - FP16 or BF16 KV-Cache
   - AsCacheQuantI8 - int8 KV-Cache
   - AsCacheQuantU4 - uint4 KV-Cache""",

    "eviction_strategy":                "how to choose eviction request  when kv-cache is full for GPU choose between(default : MaxLength): [MaxLength, Random]",

    "enable_prefix_cache":              "prefill prefix caching function related settings, if you have lots of common prefix in prompts, this function is strongly suggested, default : TRUE, [TRUE, FALSE] ",
    "prefix_cache_ttl":                 "gpu prefix cache time to live in sec, default : 300 ",

    "cuda_mem":                         """\
    how many gpu memory allocated by this engine is caculated by this formula:
     (TOTAL_DEVICE_GPU_MEM_MB - RESERVE_SIZE_MB) * MEMORY_RATIO
    TOTAL_DEVICE_GPU_MEM_MB - how many gpu memory you device have, if multiple devices have different memory, will choose the least one
    RESERVE_SIZE_MB - defined by reserve_size_mb's value.
    MEMORY_RATIO    - defined by memory_ratio's va 
    for cuda device, this ratio is how many should engine allocate memory,
    this config will override the BFC_MEM_RATIO envvar.
    set this to -1 means use "BFC_MEM_RATIO" env var's setting # comment out following config will use system env, otherwise it will override system device
    memory_ratio: 0.96
    for cuda device, this config is for setup how many memory should reserve in Mega-Bytes(MB) reserve_size_mb: 600
    """,

    "rope_scaling": "Setup RoPE Scaling method config, which is a KV Value Setting, this key-value config will forward to engine's runtime config, reference engine's document for detail settings."
}

generation_comments_dict = {
    "max_length": "max length for this request, input + output, if max length is reached, engine will finish this request. this value should update together with runtime_config.engine_max_length and input length, the config value is just for a default value.",
 }


tokenizers_comments_dict = {
    "source": "support [modelscope, huggingface, local]",
    "local_path" :  " require this path can be called by transformers.AutoTokenizer()"
}

example_di_config_file = """
# this config is describing  model's basic information without depends on transfomers python frameworks etc.
# for easier setup and correct inference result.

# 
# ------------------------
# model meta info part, usually imported from huggingface config.json
# example for qwen1.5 1.8B
model:

# 
#  -------------------
# runtime config part
runtime_config:

# 
# -----------------------
# generation config part
generation_config:

#   
# -----------------------  
# tokenizer part
# it support use model hub's AutoTokenizer to set up tokenizer, also can use path to download
tokenizer:
  source: modelscope  # support [modelscope, huggingface, local]
                      # will report error if huggingface_id, or modelscope_id is not set correctly.
  local_path: USER_DEFINE_TOKENIZER_PATH # require this path can be called by transformers.AutoTokenizer()
"""

class DIConfigBuilder:
    """ Build DIConfig from AllSpark(DashInfer) structure, and export to yaml config and python dict"""
    def __init__(self):
        self.model_dict = {}
        self.runtime_config_dict = {}
        self.generation_config_dict = {}
        self.tokenizer_dict = {}

    def model(self, as_model_config,  hf_id = "", modelscope_id = ""):

        self.model_dict["huggingface_id"] = hf_id
        self.model_dict["modelscope_id"] = modelscope_id
        self.model_dict.update(as_model_config)

        keep_keys = {'huggingface_id', 'modelscope_id', 'architectures', 'hidden_size', 'max_position_embeddings',
                     'num_attention_heads', 'num_hidden_layers', 'num_key_value_heads', 'rms_norm_eps', 'rope_theta',
                     'sliding_window', 'use_sliding_window', 'vocab_size'}

        self.model_dict = self.__fillter_dict(keep_keys, self.model_dict)

        return self


    def runtime_config(self, as_runtime_cfg: AsModelConfig):

        def to_cache_mode_str(mode: AsCacheMode):
            if mode == AsCacheMode.AsCacheDefault:
                return "AsCacheDefault"
            elif mode == AsCacheMode.AsCacheQuantI8:
                return "AsCacheQuantI8"
            elif mode == AsCacheMode.AsCacheQuantU4:
                return "AsCacheQuantU4"
            else:
                raise ValueError("Invalid cache mode")
        def to_eviction_strategy_str(mode : AsEvictionStrategy):
            if mode == AsEvictionStrategy.Random:
                return "Random"
            elif mode == AsEvictionStrategy.MaxLength:
                return "MaxLength"
            else:
                raise ValueError("invalid eviction_strategy")


        self.runtime_config_dict["model_name"] = as_runtime_cfg.model_name
        self.runtime_config_dict['engine_max_length'] = as_runtime_cfg.engine_max_length
        self.runtime_config_dict['engine_max_batch'] = as_runtime_cfg.engine_max_batch
        self.runtime_config_dict['matmul_precision'] = as_runtime_cfg.matmul_precision
        self.runtime_config_dict['lora_names'] = as_runtime_cfg.lora_names

        self.runtime_config_dict['compute_unit'] = {}

        device_type_str, device_ids_list = AsModelRuntimeConfigBuilder.parse_device_type(as_runtime_cfg.compute_unit)
        self.runtime_config_dict['compute_unit']['device_type'] = device_type_str
        self.runtime_config_dict['compute_unit']['device_ids'] = device_ids_list
        self.runtime_config_dict['compute_unit']['compute_thread_in_device'] = as_runtime_cfg.num_threads

        self.runtime_config_dict['kv_cache_mode'] = to_cache_mode_str(as_runtime_cfg.cache_mode)
        self.runtime_config_dict['enable_prefix_cache'] = as_runtime_cfg.enable_prefix_cache
        self.runtime_config_dict['prefix_cache_ttl'] = as_runtime_cfg.prefix_cache_ttl
        self.runtime_config_dict['cuda_mem'] = {}

        self.runtime_config_dict['eviction_strategy'] = to_eviction_strategy_str(as_runtime_cfg.eviction_strategy)

        # todo repo scaling
        return self

    def generation_config(self, hf_generation_config):
        self.generation_config_dict = hf_generation_config
        keep_keys = {'do_sample', 'early_stopping', 'eos_token_id', 'length_penalty', 'max_length',
                     'min_length', 'no_repeat_ngram_size', 'presence_penalty', 'repetition_penalty', 'stop_words_ids',
                     'temperature', 'top_k', 'top_p'}
        self.generation_config_dict = self.__fillter_dict(keep_keys, self.generation_config_dict)

        return self

    def tokenizer(self, source="modelscope", local_path = ""):
        self.tokenizer_dict['source'] = source
        self.tokenizer_dict['local_path'] = local_path
        return self

    def build_py_obj(self):
        """ build diconfig into a python object, if want to config to yaml object, use yaml lib."""
        ret = {}
        ret["model"] = self.model_dict
        ret['runtime_config'] = self.runtime_config_dict
        ret['generation_config'] = self.generation_config_dict
        ret['tokenizer'] = self.tokenizer_dict
        return ret

    def __fillter_dict(self, keep_keys, in_dict):
        filtered_dict = {k: v for k, v in in_dict.items() if k in keep_keys}
        return filtered_dict

    def build_yaml_path(self, output_path):
        """ export  diconfig into a yaml file with path"""
        dict = self.build_py_obj()

        from ruamel.yaml import CommentedMap

        yaml = ruamel.yaml.YAML()
        example_di_config = yaml.load(example_di_config_file)

        example_di_config.yaml_set_comment_before_after_key(key="model", before=model_comments)

        example_di_config.insert(0, "model", CommentedMap(dict["model"]))
        example_di_config.insert(1, "runtime_config", CommentedMap(dict["runtime_config"]))
        example_di_config.insert(2, "generation_config", CommentedMap(dict["generation_config"]))
        example_di_config.insert(3, "tokenizer", CommentedMap(dict["tokenizer"]))

        for key in model_comments_dict:
            example_di_config["model"].yaml_set_comment_before_after_key(key=key, before=model_comments_dict[key])

        for key in runtime_cfg_comments_dict:
            example_di_config['runtime_config'].yaml_set_comment_before_after_key(key=key, before=runtime_cfg_comments_dict[key])

        for key in generation_comments_dict:
            example_di_config['generation_config'].yaml_set_comment_before_after_key(key=key, before=generation_comments_dict[key])

        for key in tokenizers_comments_dict:
            example_di_config['tokenizer'].yaml_set_comment_before_after_key(key=key, before=tokenizers_comments_dict[key])

        with open(output_path, 'w') as file:
            yaml.dump(example_di_config, file)


