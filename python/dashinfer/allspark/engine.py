#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    engine.py
#
from ._allspark import AsStatus, AsEngine, AsEngineStat
from ._allspark import AsClientEngine
from .model import *
from os import path, makedirs
from dashinfer import allspark
import functools, inspect, hashlib
import time
import torch
import sys


class EngineBase:

    def __init__(self):
        self.model_map = {
            "LLaMA_v2": LLaMA_v2,
            "ChatGLM_v2": ChatGLM_v2,
            "ChatGLM_v3": ChatGLM_v3,
            "Qwen_v10": Qwen_v10,
            "Qwen_v15": Qwen_v15,
        }

        self.version = ""

    def dump_torch_args(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ordered_args = inspect.signature(func).bind(*args,
                                                        **kwargs).arguments
            remove_keys = [
                "self", "torch_model", "model_config", "generate_config"
            ]
            dict_args = {
                key: value
                for key, value in ordered_args.items()
                if key not in remove_keys
            }
            return func(dump_param_to_dict=dict_args, *args, **kwargs)

        return wrapper

    @dump_torch_args
    def serialize_model_from_torch(self,
                                   model_name,
                                   model_type,
                                   torch_model,
                                   model_config,
                                   save_dir,
                                   multinode_mode=1,
                                   data_type="float32",
                                   do_dynamic_quantize_convert=False,
                                   quant_config=None,
                                   use_dynamic_ntk=False,
                                   use_logn_attn=False,
                                   model_sequence_length=2048,
                                   seqlen_extrapolation=1.0,
                                   rotary_base=10000.0,
                                   dump_param_to_dict={},
                                   **kwargs):
        if not path.isdir(save_dir):
            makedirs(save_dir)
        assert model_name, 'model_name should not be empty!'
        model_path = path.join(save_dir, model_name + ".asgraph")
        weights_path = path.join(save_dir, model_name + ".asparam")

        start_time = time.time()
        print("save asgraph to ", model_path)
        print("save asparam to ", weights_path)
        # 清空原本的权重文件
        with open(weights_path, "w") as f:
            f.truncate(0)

        derive_type = "lmhead"
        enable_i8cache_mha = False
        mha_kv_cache_config = None
        do_binary_add_fused = False
        model_proto = self.model_map[model_type](
            torch_model,
            data_type,
            derive_type,  # 4 positional args
            multinode_mode=multinode_mode,
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
            rotary_base=rotary_base,
        )()  #use map weight for build model to decrease time consumption
        model_proto = self.dump_build_meta_to_proto(model_proto, weights_path,
                                                    dump_param_to_dict)
        with open(model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("serialize_model_from_torch: save model = true, time : ",
              time.time() - start_time)

    def dump_build_meta_to_proto(self, model_proto, weights_path,
                                 torch_build_param):
        # version
        bver = model_proto.build_meta.version
        # get_ver_str should be  1.0.0/(GitSha1:85f6ac74) or
        #                        1.0.0/(GitSha1:85f6ac74-dirty)
        get_ver_str = self.version
        ver_list = get_ver_str.split("/")[0].split(".")
        git_str = get_ver_str.split(":")[1].split(")")[0].split("-")[0]
        print("current allspark version major[", ver_list[0], "] minor[",
              ver_list[1], "] patch[", ver_list[2], "] commit = ", git_str)
        bver.major = int(ver_list[0])
        bver.minor = int(ver_list[1])
        bver.patch = int(ver_list[2])
        bver.git_commit = git_str
        # hash
        hash = model_proto.build_meta.weight_hash
        hash.algorithm = "md5"
        hash_length = 32 * 1024 * 1024  # 32M least
        with open(weights_path, 'rb') as weight_file:
            weight_file_content = weight_file.read(hash_length)
            hash_length = len(weight_file_content)
            hash.hash_length.extend([hash_length])
            hash_md5 = hashlib.md5(weight_file_content).hexdigest()
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
        return model_proto


'''
class for allspark engine which is used for inferer.
'''


class Engine(AsEngine):

    def __init__(self):
        super().__init__()
        self.engine = EngineBase()
        self.engine.version = self.get_version_full()

    def get_model_config(self, model_config={}):
        return self.engine.get_model_config(model_config)

    def serialize_model_from_torch(
        self,
        model_name,
        model_type,
        torch_model,
        model_config,
        save_dir,
        multinode_mode=1,
        data_type="float32",
        do_dynamic_quantize_convert=False,
        quant_config=None,
        use_dynamic_ntk=False,
        use_logn_attn=False,
        model_sequence_length=2048,
        seqlen_extrapolation=1.0,
        rotary_base=10000.0,
    ):
        return self.engine.serialize_model_from_torch(
            model_name, model_type, torch_model, model_config, save_dir,
            multinode_mode, data_type, do_dynamic_quantize_convert,
            quant_config, use_dynamic_ntk, use_logn_attn,
            model_sequence_length, seqlen_extrapolation, rotary_base)

    def dump_build_meta_to_proto(self, model_proto, weights_path,
                                 torch_build_param):
        return self.engine.dump_build_meta_to_proto(model_proto, weights_path,
                                                    torch_build_param)

    def build_model_from_config_struct(self, as_model_config_struct):
        self._build_model_from_as_model_config(as_model_config_struct)

    def start_model(
        self,
        model_name: str,
    ) -> AsStatus:
        return self._start_model(model_name)

    def stop_model(
        self,
        model_name: str,
    ) -> AsStatus:
        return self._stop_model(model_name)

    def start_request(
        self,
        model_name: str,
        inputs,
        generate_config={},
    ):
        return self._start_request(model_name, inputs, generate_config)

    def get_no_wait(self, model_name: str, result_queue):
        return self._get_no_wait(model_name, result_queue)

    def get_wait(self, model_name: str, result_queue):
        return self._get_wait(model_name, result_queue)

    def get_output_no_wait(self, model_name: str, result_queue):
        return self._get_output_no_wait(model_name, result_queue)

    def get_output_wait(self, model_name: str, result_queue):
        return self._get_output_wait(model_name, result_queue)

    def get_request_status(self, model_name: str, result_queue):
        return self._get_request_status(model_name, result_queue)

    def stop_request(self, model_name: str, request_handle) -> AsStatus:
        return self._stop_request(model_name, request_handle)

    def release_request(self, model_name: str, request_handle) -> AsStatus:
        return self._release_request(model_name, request_handle)

    def sync_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Block all API calls to this model until the target request is complete.
        If request_handle is None, synchronize all requests.
        """
        return self._sync_request(model_name, request_handle)

    def save_model(self,
                   model_name,
                   saved_dir,
                   with_weight=True,
                   is_text_graph=False):
        return self._save_model(model_name, saved_dir, with_weight,
                                is_text_graph)


'''
class for allspark client engine which lauch mpi daemon services to process multinuma inferer.
In this way, Allspark inferer served as daemon processes based on config numa nums, Client engine
broadcasts request to Allspark daemon services and receive response based on grpc.
Normally, it is used for CPU multi-numa inferer.
'''


class ClientEngine(AsClientEngine):

    def __init__(self):
        super().__init__()
        self.engine = EngineBase()
        self.engine.version = self.get_version_full()

    def get_model_config(self, model_config={}):
        return self.engine.get_model_config(model_config)

    def serialize_model_from_torch(
        self,
        model_name,
        model_type,
        torch_model,
        model_config,
        save_dir,
        multinode_mode=1,
        data_type="float32",
        do_dynamic_quantize_convert=False,
        quant_config=None,
        use_dynamic_ntk=False,
        use_logn_attn=False,
        model_sequence_length=2048,
        seqlen_extrapolation=1.0,
        rotary_base=10000.0,
    ):
        return self.engine.serialize_model_from_torch(
            model_name, model_type, torch_model, model_config, save_dir,
            multinode_mode, data_type, do_dynamic_quantize_convert,
            quant_config, use_dynamic_ntk, use_logn_attn,
            model_sequence_length, seqlen_extrapolation, rotary_base)

    def dump_build_meta_to_proto(self, model_proto, weights_path,
                                 torch_build_param):
        return self.engine.dump_build_meta_to_proto(model_proto, weights_path,
                                                    torch_build_param)

    def build_model_from_config_struct(self, as_model_config_struct):
        self._build_model_from_as_model_config(as_model_config_struct)

    def start_model(
        self,
        model_name: str,
    ) -> AsStatus:
        return self._start_model(model_name)

    def stop_model(
        self,
        model_name: str,
    ) -> AsStatus:
        return self._stop_model(model_name)

    def start_request(
        self,
        model_name: str,
        inputs,
        generate_config={},
    ):
        return self._start_request(model_name, inputs, generate_config)

    def get_no_wait(self, model_name: str, result_queue):
        return self._get_no_wait(model_name, result_queue)

    def get_wait(self, model_name: str, result_queue):
        return self._get_wait(model_name, result_queue)

    def get_output_no_wait(self, model_name: str, result_queue):
        return self._get_output_no_wait(model_name, result_queue)

    def get_output_wait(self, model_name: str, result_queue):
        return self._get_output_wait(model_name, result_queue)

    def get_request_status(self, model_name: str, result_queue):
        return self._get_request_status(model_name, result_queue)

    def stop_request(self, model_name: str, request_handle) -> AsStatus:
        return self._stop_request(model_name, request_handle)

    def release_request(self, model_name: str, request_handle) -> AsStatus:
        return self._release_request(model_name, request_handle)

    def sync_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Block all API calls to this model until the target request is complete.
        If request_handle is None, synchronize all requests.
        """
        return self._sync_request(model_name, request_handle)

    def get_free_frame(
        self,
        model_name: str,
    ) -> int:
        return self._get_free_frame(model_name)

    def get_as_engine_stat(
        self,
        model_name: str,
    ):
        return self._get_as_engine_stat(model_name)

    def reset_stream(self, model_name, **kwargs):
        return self._reset_stream(model_name, kwargs)

    def save_model(self,
                   model_name,
                   saved_dir,
                   with_weight=True,
                   is_text_graph=False):
        return self._save_model(model_name, saved_dir, with_weight,
                                is_text_graph)
