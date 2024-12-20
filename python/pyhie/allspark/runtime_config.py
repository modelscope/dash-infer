'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    runtime_config.py
'''
from ._allspark import AsModelConfig, AsCacheMode

import os
from .engine import TargetDevice


def get_cache_mode_from_str(s):
    if s == "AsCacheDefault":
        return AsCacheMode.AsCacheDefault
    if s == "AsCacheQuantI8":
        return AsCacheMode.AsCacheQuantI8
    if s == "AsCacheQuantU4":
        return AsCacheMode.AsCacheQuantU4
    raise ValueError(f"unknown cache mode found: {s}")


class AsModelRuntimeConfigBuilder:

    def __init__(self):
        self.model_path = "DEFAULT_"
        self.weights_path = "NOT_EXISTS"
        self.thread_num = 0
        self.engine_max_batch = 32
        self.engine_max_length = 2048

        self.new_runtime_cfg = AsModelConfig()
        self.new_runtime_cfg.lora_max_rank = 64
        self.new_runtime_cfg.lora_max_num = 5

    """
    The Runtime config, such as max batch, max length, and runtime feature like kv-cache quantization, etc.

    This class provide more useful pythonic function and more friendly API.
    """

    def model_name(self, model_name: str) -> 'AsModelRuntimeConfigBuilder':
        """Sets the name of the model."""
        self.new_runtime_cfg.model_name = model_name
        return self

    def model_dir(self, model_dir, file_name_prefix) -> 'AsModelRuntimeConfigBuilder':
        self.model_path = os.path.join(model_dir, file_name_prefix + ".asgraph")
        self.weights_path = os.path.join(model_dir, file_name_prefix + ".asparam")

        self.new_runtime_cfg.model_path = self.model_path
        self.new_runtime_cfg.weights_path = self.weights_path

        return self

    def model_file_path(self, model_file_name, weight_file_path):
        self.model_path = model_file_name
        self.weights_path = weight_file_path
        self.new_runtime_cfg.model_path = self.model_path
        self.new_runtime_cfg.weights_path = self.weights_path
        return self

    def compute_unit(self, target_device: TargetDevice, device_id_array=None,
                     compute_thread_in_device: int = 0) -> 'AsModelRuntimeConfigBuilder':
        """
        Set up runtime compute unit in a single function, device enum are listed in TargetDevice enum,
        can be {CUDA, CPU, CPU_NUMA}.

        for CUDA, the device_count is card number, and compute_thread_in_device can be ignored.
        for CPU,  the device_count is always 1, and compute_thread_in_device is how many compute thread when inference,
                  suggest value is physical core number(without hyper-thread), or you can pass 0 to let autodetect.
        for CPU_NUMA, the device_count means NUMA count, and compute_thread_in_device means compute thread inside NUMA

        Args:
            target_device: device enum are listed in TargetDevice enum, can be {CUDA, CPU, CPU_NUMA}.
            device_id_array: when target_device is CUDA, device_count are card id, like card 0, card1,  pass [0,1],
                             when target_device is CPU it's NUMA node number, default is [0]
            compute_thread_in_device: for cpu compute_thread_in_device means compute thread inside one NUMA

        Returns:  void

        """
        if device_id_array is None:
            device_id_array = [0]
        if not isinstance(target_device, TargetDevice):
            # do convert from string to Target Device.
            if isinstance(target_device, str):
                lower_str = str.lower(target_device)
                target_device_str = target_device
                mapping = {"cuda": TargetDevice.CUDA, "cpu": TargetDevice.CPU, "cpu_numa": TargetDevice.CPU_NUMA}
                if lower_str in mapping:
                    target_device = mapping[lower_str]
                else:
                    raise ValueError(f"target device not supported: {target_device_str}")

        def helper_func(prefix: str, id_list: list):
            for id in id_list:
                prefix += str(id)
                prefix += ','
            prefix = prefix[:-1]  # remove the last ,
            return prefix

        compute_unit_str: str

        self.thread_num = compute_thread_in_device
        if target_device == TargetDevice.CUDA:
            if len(device_id_array) == 0:
                compute_unit_str = TargetDevice.CUDA.name + ":0"
            else:
                compute_unit_str = TargetDevice.CUDA.name + ":"
                compute_unit_str = helper_func(compute_unit_str, device_id_array)

        elif target_device == TargetDevice.CPU:
            compute_unit_str = TargetDevice.CPU.name + ":0"
        elif target_device == TargetDevice.CPU_NUMA:
            compute_unit_str = TargetDevice.CPU.name + ":"
            compute_unit_str = helper_func(compute_unit_str, device_id_array)
        else:
            raise ValueError(f"not support target device {target_device.name} ")

        self.new_runtime_cfg.compute_unit = compute_unit_str
        self.new_runtime_cfg.num_threads = self.thread_num
        return self


    def max_length(self, length: int) -> 'AsModelRuntimeConfigBuilder':
        """Sets the maximum sequence length for the engine."""
        if isinstance(length, int):
            self.new_runtime_cfg.engine_max_length = length
        elif isinstance(length, str):
            self.new_runtime_cfg.engine_max_length = int(length)
        return self

    def max_batch(self, batch: int) -> 'AsModelRuntimeConfigBuilder':
        """Sets the maximum batch size for the engine."""
        if isinstance(batch, int):
            self.new_runtime_cfg.engine_max_batch = batch
        elif isinstance(batch, str):
            self.new_runtime_cfg.engine_max_batch = int(batch)
        return self

    def lora_max_rank(self, rank: int) -> 'AsModelRuntimeConfigBuilder':
        """Sets the maximum sequence length for the engine."""
        if isinstance(rank, int):
            self.new_runtime_cfg.lora_max_rank = rank
        elif isinstance(rank, str):
            self.new_runtime_cfg.lora_max_rank = int(rank)
        return self

    def lora_max_num(self, num: int) -> 'AsModelRuntimeConfigBuilder':
        """Sets the maximum sequence length for the engine."""
        if isinstance(num, int):
            self.new_runtime_cfg.lora_max_num = num
        elif isinstance(num, str):
            self.new_runtime_cfg.lora_max_num = int(num)
        return self

    
    def max_prefill_length(self, length: int) -> 'AsModelRuntimeConfigBuilder':
        self.new_runtime_cfg.engine_max_prefill_length = length
        return self
    
    def prefill_cache(self, enable=True) -> 'AsModelRuntimeConfigBuilder':
        self.new_runtime_cfg.enable_prefix_cache = enable
        return self

    def prefix_cache_ttl(self, ttl=300) -> 'AsModelRuntimeConfigBuilder':
        self.new_runtime_cfg.prefix_cache_ttl = ttl
        return self

    def enable_sparsity_matmul(self, enable=False) -> 'AsModelRuntimeConfigBuilder':
        if enable == True:
            device_type, device_ids = self.new_runtime_cfg.compute_unit.strip().split(":")
            assert device_type == "CUDA", "Only Support Sparse Matmul with CUDA"

            import torch
            if not (8 <= torch.cuda.get_device_capability()[0] < 9):
                enable = False
        self.new_runtime_cfg.enable_sparsity_matmul = enable
        return self

    def kv_cache_mode(self, cache_mode: AsCacheMode):
        self.new_runtime_cfg.cache_mode = cache_mode
        return self
    
    def kv_cache_span_size(self, span_size: int):
        """
        Valid span_size is 16, 32, 64, and 128. Default is 32.
        """
        self.new_runtime_cfg.cache_span_size = span_size
        return self

    @staticmethod
    def parse_device_type(compute_unit):
        def get_device_type_from_string(type_str):
            if type_str == "CPU" or type_str == "CPU_NUMA" or type_str == "CUDA":
                return type_str
            raise ValueError(f"invalid device type: {type_str} not supported.")

        pos = compute_unit.find(':')
        if pos == -1:
            raise ValueError(f"Not Support ComputeUnit: {compute_unit}")

        device_type = get_device_type_from_string(compute_unit[:pos])
        device_ids_str = compute_unit[pos + 1:]
        device_ids = [int(item) for item in device_ids_str.split(',') if item.strip()]

        return device_type, device_ids

    def set_compute_unit_from_str(self, compute_unit):
        """ setup compute unit like CUDA:0, CUDA:0,1 like string."""
        self.new_runtime_cfg.compute_unit = compute_unit
        return self
    def set_span_init_size(self, span_init_size):
        """ initial span size, 0 for auto compute """
        self.new_runtime_cfg.cache_span_num_init = span_init_size
        return self

    def from_dict(self, rfield):
        """param dict, the format should align with DIConfig 's define. """

        (self.model_name(rfield['model_name'])
         .compute_unit(rfield['compute_unit']['device_type'], rfield['compute_unit']['device_ids'], rfield['compute_unit']['compute_thread_in_device'])
         .max_length(rfield['engine_max_length']).max_batch(rfield['engine_max_batch']))


        self.update_from_dict(rfield)
        return self

    def update_from_dict(self, rfield):
        """ Update dict, """
        if "enable_prefix_cache" in rfield:
            self.prefill_cache(bool(rfield['enable_prefix_cache']))
        if "prefix_cache_ttl" in rfield:
            self.prefix_cache_ttl(int(rfield['prefix_cache_ttl']))
        if "kv_cache_mode" in rfield:
            self.kv_cache_mode(get_cache_mode_from_str(rfield['kv_cache_mode']))
        if "kv_cache_span_size" in rfield:
            self.kv_cache_span_size(int(rfield['kv_cache_span_size']))
        if "enable_sparsity_matmul" in rfield:
            self.enable_sparsity_matmul(bool(rfield['enable_sparsity_matmul']))
        if "cache_span_num_init" in rfield:
            self.set_span_init_size(int(rfield['cache_span_num_init']))
        if "engine_max_prefill_length" in rfield:
            self.max_prefill_length(int(rfield("engine_max_prefill_length")))
        if "engine_max_length" in rfield:
            self.max_length(int(rfield["engine_max_length"]))
        if "engine_max_batch" in rfield:
            self.max_batch(int(rfield["engine_max_batch"]))

        return self



    def build(self) -> AsModelConfig:
        # make sure next build call will create a new config.
        ret = self.new_runtime_cfg
        self.new_runtime_cfg = AsModelConfig()
        return ret
