'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    engine_utils.py
'''
from .model import *
from os import path, makedirs
import functools, inspect, hashlib
import time
import torch
import sys


class EngineUtils:
    """some common function used in Engine and Engine client."""
    def __init__(self):
        self.model_map = {
            "LLaMA": LLaMA,
            "LLaMA_v2": LLaMA_v2,
            "LLaMA_v3": LLaMA_v3,
            "ChatGLM_v1": ChatGLM_v1,
            "ChatGLM_v2": ChatGLM_v2,
            "ChatGLM_v3": ChatGLM_v3,
            "ChatGLM_v4": ChatGLM_v4,
            "Baichuan_v1": Baichuan_v1,
            "Baichuan_v2": Baichuan_v2,
            "Qwen_v10": Qwen_v10,
            "Qwen_v15": Qwen_v15,
            "Qwen_v20": Qwen_v20,
            "QwenCode_v20": QwenCode_v20,
            "Qwen_v20_MOE":Qwen_v20_MOE,
        }

        self.version = ""

    def dump_torch_args(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ordered_args = inspect.signature(func).bind(*args,
                                                        **kwargs).arguments
            remove_keys = [
                "self", "torch_model", "model_config", "generate_config",
            ]
            dict_args = {
                key: value
                for key, value in ordered_args.items()
                if key not in remove_keys
            }
            # print("func-arg ", dict_args)
            return func(dump_param_to_dict=dict_args, *args, **kwargs)

        return wrapper

    def serialize_model_from_torch(self,
                                   model_name,
                                   model_type,
                                   torch_model,
                                   model_config,
                                   save_dir,
                                   as_model_path,
                                   as_weight_path,
                                   data_type="float32",
                                   derive_type="lmhead",
                                   multigpu_mode=1,
                                   do_binary_add_fused=True,
                                   do_dynamic_quantize_convert=False,
                                   quant_config=None,
                                   enable_i8cache_mha=False,
                                   mha_kv_cache_config=None,
                                   use_dynamic_ntk=False,
                                   use_logn_attn=False,
                                   model_sequence_length=2048,
                                   seqlen_extrapolation=1.0,
                                   lora_cfg=None,
                                   rotary_base=10000.0,
                                   dump_param_to_dict={},
                                   **kwargs):
        print(f"serialize_model_from_torch: quant config:{quant_config}")
        model_path = ""
        weights_path = ""

        if as_model_path is not None and as_weight_path is not None:
            if save_dir is not None:
                print("warnning: serialize_model_from_torch: save_dir is not None, "
                      "and also set as_model_path and as_graph path, will override save_dir settings")
            model_path = as_model_path
            weights_path = as_weight_path
        else:
            if save_dir is None:
                raise ValueError("both as_model path and save dir is None, please specify save_dir name or file path.")
            if not path.isdir(save_dir):
                makedirs(save_dir)
            assert model_name, 'model_name should not be empty!'
            model_path = path.join(save_dir, model_name + ".asgraph")
            weights_path = path.join(save_dir, model_name + ".asparam")

        start_time = time.time()
        only_convert_lora = False
        if type(lora_cfg) == type(dict()):
            only_convert_lora = lora_cfg.get('lora_only', False)
        if not only_convert_lora:
            print("save asgraph to ", model_path)
            print("save asparam to ", weights_path)
            # 清空原本的权重文件
            with open(weights_path, "w") as f:
                f.truncate(0)
        model_proto = self.model_map[model_type](
            torch_model,
            data_type,
            derive_type,  # 4 positional args
            multigpu_mode=multigpu_mode,
            model_config=model_config,
            do_binary_add_fused=do_binary_add_fused,
            do_dynamic_quantize_convert=do_dynamic_quantize_convert,
            quant_config=quant_config,
            enable_i8cache_mha=enable_i8cache_mha,
            mha_kv_cache_config=mha_kv_cache_config,
            weights_path=weights_path,
            use_dynamic_ntk=use_dynamic_ntk,
            use_logn_attn=use_logn_attn,
            model_sequence_length=model_sequence_length,
            seqlen_extrapolation=seqlen_extrapolation,
            lora_cfg=lora_cfg,
            rotary_base=rotary_base,
        )()  #use map weight for build model to decrease time consumption
        dump_param_to_dict['only_convert_lora'] = only_convert_lora
        model_proto = self.dump_build_meta_to_proto(model_proto, weights_path,
                                                    dump_param_to_dict)
        if not only_convert_lora:
            with open(model_path, "wb") as f:
                f.write(model_proto.SerializeToString())
        print("build_model_from_torch: save model = true, time : ",
              time.time() - start_time)

    def dump_build_meta_to_proto(self, model_proto, weights_path,
                                 torch_build_param):
        # version
        bver = model_proto.build_meta.version
        # git_ver_str should be  1.0.0/(GitSha1:85f6ac74) or
        #                        1.0.0/(GitSha1:85f6ac74-dirty)
        git_ver_str = self.version
        ver_list = git_ver_str.split("/")[0].split('-')[0].split(".") # remove rc2 in 1.0.0-rc2
        git_str = git_ver_str.split(":")[1].split(")")[0].split("-")[0]
        print(f"current allspark version full: {git_ver_str} major[", ver_list[0], "] minor[",
              ver_list[1], "] patch[", ver_list[2], "] commit = ", git_str)
        bver.major = int(ver_list[0])
        bver.minor = int(ver_list[1])
        bver.patch = int(ver_list[2])
        bver.git_commit = git_str
        # hash
        hash = model_proto.build_meta.weight_hash
        hash.algorithm = "md5"
        hash_length = 32 * 1024 * 1024  # 32M least
        only_convert_lora = torch_build_param.get('only_convert_lora', False)
        if not only_convert_lora:
            with open(weights_path, 'rb') as weight_file:
                weight_file_content = weight_file.read(hash_length)
                hash_length = len(weight_file_content)
                hash.hash_length.extend([hash_length])
                hash_md5 = hashlib.md5(weight_file_content).hexdigest(
                )  # TODO(zhangyufei): pre-cal md5 instead.
                print("calculate md5 of asgraph = ", hash_md5)
                hash.hash.extend([hash_md5])
        # config dict
        def valid_torch_kvpair(key, value) -> bool:
            if isinstance(value, float) or isinstance(
                    value, list) or isinstance(value, str) or isinstance(
                        value, int) or isinstance(value, bool):
                return True
            return False

        tcfg = model_proto.build_meta.torch_build_config
        for k, v in torch_build_param.items():
            if valid_torch_kvpair(k, v):
                print("torch build meta: \t", k, "\t: ", v)
                tcfg[k] = str(v)
        # return
        return model_proto
