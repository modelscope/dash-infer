#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    helper.py
#
import os
import sys
from enum import Enum
import numpy as np
import time
import json
import pandas as pd
from tabulate import tabulate
from dataclasses import dataclass, field
from typing import Any, Optional, List

import torch
import torch.utils.dlpack
from transformers import AutoTokenizer, AutoModelForCausalLM

from dashinfer import allspark
from dashinfer.allspark.quantization import QuantizeConfig


class EngineHelper():

    def __init__(self, config):
        self.model_name = config["model_name"]
        self.model_type = config["model_type"]

        self.model_path = os.path.expanduser(config["model_path"])
        self.data_type = config["data_type"]
        self.device_type = config["device_type"]
        self.device_ids = [config["device_ids"][0]]

        self.multinode_mode = config["multinode_mode"]
        if len(config["device_ids"]) > 1:
            self.multinode_mode = True

        self.model_name += "_" + self.device_type.lower()
        self.model_name += "_multi" if self.multinode_mode else "_single"
        self.model_name += "_" + self.data_type

        self.convert_config = config["convert_config"]
        if self.convert_config["do_dynamic_quantize_convert"] == True:
            if config["quantization_config"]["weight_type"] in [
                    "int8", "uint8"
            ]:
                self.model_name += "_a16w8"
            elif config["quantization_config"]["weight_type"] in ["uint4"]:
                self.model_name += "_a16w4"

            self.quant_config = QuantizeConfig(
                activation_type=config["quantization_config"]
                ["activation_type"],
                weight_type=config["quantization_config"]["weight_type"],
                extra_option={
                    "SubChannel": config["quantization_config"]["SubChannel"],
                    "GroupSize": config["quantization_config"]["GroupSize"],
                })
        else:
            self.quant_config = None

        print(f"### convert_config: {self.convert_config}")

        self.default_gen_cfg = config["generation_config"]

        self.engine_config = config["engine_config"]
        print(f"### engine_config: {self.engine_config}")

        self.profiling_data = self.ProfilingData(
            self.engine_config["engine_max_batch"])

        self.tokenizer = None
        self.torch_model = None
        self.torch_model_module = None
        self.model_config = None
        self.verbose = False

        if self.multinode_mode:
            self.engine = allspark.ClientEngine()
        else:
            self.engine = allspark.Engine()

    @dataclass
    class Request:
        id: int = -1
        model_info: Optional[str] = None
        torch_input: Optional[dict] = None
        in_text: Optional[str] = None
        in_tokens: Optional[List[int]] = None
        in_tokens_len: Optional[int] = None
        out_text: Optional[str] = None
        out_tokens: Optional[List[int]] = None
        out_tokens_len: Optional[int] = None
        status: Optional[int] = None
        gen_cfg: Optional[dict] = None
        start_timestamp: Optional[str] = None
        end_timestamp: Optional[str] = None
        context_time: float = 0
        generate_time: float = 0
        handle: Any = field(default=None)
        queue: Any = field(default=None)

    class ProfilingData():

        def __init__(self, max_batch):
            self.reset(max_batch)
            self.df = pd.DataFrame(columns=[
                "Batch_size", "Request_num", "Avg_in_tokens", "Avg_out_tokens",
                "Avg_context_time(s)", "Avg_generate_time(s)",
                "Avg_throughput(token/s)", "Throughput(token/s)", "QPS"
            ])

        def reset(self, max_batch):
            self.max_batch = max_batch
            self.total_input_len = 0.0
            self.total_output_len = 0.0
            self.total_context_time = 0.0
            self.total_generate_time = 0.0
            self.total_throughput = 0.0
            self.total_time = 0.0
            self.avg_input_len = 0.0
            self.avg_output_len = 0.0
            self.avg_context_time = 0.0
            self.avg_generate_time = 0.0
            self.avg_throughput = 0.0
            self.request_num = 0
            self.qps = 0.0

        def update(self, in_len, out_len, ctx_time, gen_time):
            self.request_num += 1
            self.batch_size = self.request_num if self.request_num < self.max_batch else self.max_batch
            self.total_input_len += in_len
            self.total_output_len += out_len
            self.total_context_time += ctx_time
            self.total_generate_time += gen_time

            self.avg_input_len = self.total_input_len / self.request_num
            self.avg_output_len = self.total_output_len / self.request_num
            self.avg_context_time = self.total_context_time / self.request_num
            self.avg_generate_time = self.total_generate_time / self.request_num
            self.avg_throughput = self.avg_output_len / self.avg_generate_time
            self.total_throughput = self.avg_throughput * self.batch_size

        def update_df(self, total_time):
            self.total_time = total_time
            if self.total_time > 0.0:
                self.qps = self.request_num / self.total_time
            else:
                self.qps = float("nan")
                print(
                    f"[Warning] total timecost is not set, cannot calculate QPS."
                )

            new_frame = pd.DataFrame({
                "Batch_size": [self.batch_size],
                "Request_num": [self.request_num],
                "Avg_in_tokens": [self.avg_input_len],
                "Avg_out_tokens": [self.avg_output_len],
                "Avg_context_time(s)": [self.avg_context_time],
                "Avg_generate_time(s)": [self.avg_generate_time],
                "Avg_throughput(token/s)": [self.avg_throughput],
                "Throughput(token/s)": [self.total_throughput],
                "QPS": [self.qps]
            })

            self.df = (new_frame.copy() if self.df.empty else pd.concat(
                [self.df, new_frame], ignore_index=True))

        def show(self):
            print(
                tabulate(self.df,
                         showindex=False,
                         headers="keys",
                         tablefmt="psql",
                         numalign="left",
                         floatfmt=".3f"))

    def init_tokenizer(self, hf_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path,
                                                       trust_remote_code=True,
                                                       padding_side="left")
        if self.tokenizer.eos_token_id == None:
            self.tokenizer.eos_token_id = self.default_gen_cfg["eos_token_id"]
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.default_gen_cfg["eos_token_id"] == -1:
            self.default_gen_cfg["eos_token_id"] = self.tokenizer.eos_token_id

    def init_torch_model(self, hf_model_path):
        self.torch_model = AutoModelForCausalLM.from_pretrained(
            hf_model_path, device_map="cpu", trust_remote_code=True).eval()

        if (self.data_type == 'float32'):
            self.torch_model = self.torch_model.float()
        elif (self.data_type == 'float16'):
            self.torch_model = self.torch_model.half()
        elif (self.data_type == 'bfloat16'):
            self.torch_model = self.torch_model.bfloat16()
        else:
            self.torch_model = None
            raise ValueError("unsupported data type: {}".format(
                self.data_type))

        self.model_config = self.torch_model.config.__dict__
        if "use_dynamic_ntk" not in self.model_config:
            self.model_config["use_dynamic_ntk"] = False
        if "use_logn_attn" not in self.model_config:
            self.model_config["use_logn_attn"] = False
        if "rotary_emb_base" not in self.model_config:
            self.model_config["rotary_emb_base"] = 10000
        if "rope_theta" in self.model_config:
            self.model_config["rotary_emb_base"] = max(
                self.model_config["rotary_emb_base"],
                self.model_config["rope_theta"])

        hidden_size_per_head = int(self.model_config["hidden_size"] /
                                   self.model_config["num_attention_heads"])
        self.model_config["size_per_head"] = hidden_size_per_head

        self.torch_model_module = self.torch_model.state_dict()

    def check_model_exist(self):
        model_path = os.path.join(self.model_path,
                                  self.model_name + ".asgraph")
        if not os.path.exists(model_path):
            print(f"\nNo such file or directory: {model_path}\n")
            return False
        model_path = os.path.join(self.model_path,
                                  self.model_name + ".asparam")
        if not os.path.exists(model_path):
            print(f"\nNo such file or directory: {model_path}\n")
            return False
        return True

    def init_engine(self):

        def get_physical_cores_per_numa_node():
            import re
            import subprocess
            try:
                # execute lscpu command and get its output
                lscpu_output = subprocess.run("LANG=C LC_ALL=C lscpu",
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               shell=True,
                                               text=True).stdout
            except Exception as e:
                print(f"[Warning] Error executing lscpu: {e}, use default value 0")
                return 0

            try:
                core_range = re.findall(r'NUMA node0 CPU\(s\):\s+(\d+)-(\d+)',
                                        lscpu_output)
                core_num = int(core_range[0][1]) - int(core_range[0][0]) + 1

                threads_per_core = re.findall(r'Thread\(s\) per core:\s+(\d+)',
                                              lscpu_output)
                threads_per_core = int(threads_per_core[0])
                return int(core_num / threads_per_core)
            except Exception as e:
                print(f"[Warning] Error parse lscpu ouput: {e}, use default value 0")
                return 0

        begin = time.time()

        if self.check_model_exist() == False:
            exit(-1)

        as_model_config = allspark.AsModelConfig(
            model_name=self.model_name,
            model_path=os.path.join(self.model_path,
                                    self.model_name + ".asgraph"),
            weights_path=os.path.join(self.model_path,
                                      self.model_name + ".asparam"),
            engine_max_length=self.engine_config["engine_max_length"],
            engine_max_batch=self.engine_config["engine_max_batch"],
        )

        compute_unit = self.device_type + ":"
        for id in self.device_ids:
            compute_unit += str(id) + ","
        compute_unit = compute_unit[:-1]
        as_model_config.compute_unit = compute_unit

        as_model_config.cache_mode = allspark.AsCacheMode.AsCacheDefault
        as_model_config.prefill_mode = allspark.AsMHAPrefill.AsPrefillDefault

        if self.engine_config["num_threads"] > 0:
            as_model_config.num_threads = self.engine_config["num_threads"]
        else:
            as_model_config.num_threads = get_physical_cores_per_numa_node()

        as_model_config.matmul_precision = self.engine_config[
            "matmul_precision"]

        self.engine.build_model_from_config_struct(as_model_config)

        end = time.time()
        sec = end - begin
        print("build model over, build time is {}".format(sec))

        self.engine.start_model(self.model_name)

    def uninit_engine(self):
        if self.engine.get_rank_id() == 0:
            if self.engine_config["do_profiling"]:
                profile_info = self.engine.get_op_profiling_info(
                    self.model_name)
                print()
                print(f"***************")
                print(f"* profile_info")
                print(f"***************")
                print(f"{profile_info}")

        self.engine.stop_model(self.model_name)

    def create_request(self, prompt: list, gen_cfg=None):
        if prompt == None or len(prompt) == 0:
            raise ValueError("No valid input")

        is_encoded = False
        if isinstance(prompt[0], list):
            is_encoded = True
        elif isinstance(prompt[0], str):
            is_encoded = False
        else:
            raise ValueError("Invalid inputs")

        if gen_cfg != None and len(gen_cfg) != len(prompt):
            raise ValueError("len(gen_cfg) does not match len(prompt)")

        request_list = []
        for i in range(len(prompt)):
            if is_encoded == False:
                input_text = prompt[i]
                p_tokens = self.tokenizer(input_text, return_tensors='pt')
                input_ids = p_tokens["input_ids"].tolist()
            else:
                input_text = self.tokenizer.decode(prompt[i], skip_special_tokens=False)
                input_ids = [prompt[i]]

            torch_input = {
                "input_ids": torch.Tensor(input_ids).to(torch.int64),
            }
            request = self.Request()
            request.id = i
            request.model_info = self.model_name
            request.in_text = input_text
            request.in_tokens = input_ids[0]
            request.in_tokens_len = len(input_ids[0])
            request.torch_input = torch_input
            if gen_cfg != None:
                request.gen_cfg = gen_cfg[i]
            else:
                request.gen_cfg = self.default_gen_cfg
            request_list.append(request)
        '''
        if self.verbose:
            for i in range(len(request_list)):
                request = request_list[i]
                print(f"*****************")
                print(f"* Request {i}")
                print(f"*****************")
                print(f"** text input **\n{request.in_text}\n")
                print(f"** encoded input, len: {request.in_tokens_len} **\n{request.in_tokens}\n")
                print(f"** torch input **")
                for key, value in request.torch_input.items():
                    print(f"{key}, shape: {value.shape}\n{value}")
                print()
        '''

        return request_list

    def convert_model(self, hf_model_path):
        if self.engine.get_rank_id() != 0:
            # only convert model on rank 0
            return

        if (self.torch_model_module != None):
            print("trans model from huggingface model:", hf_model_path)
            print("Dashinfer model will save to ", self.model_path)

            begin = time.time()

            print(f"### model_config: {self.model_config}")
            self.engine.serialize_model_from_torch(
                model_name=self.model_name,
                model_type=self.model_type,
                torch_model=self.torch_model_module,
                multinode_mode=self.multinode_mode,
                model_config=self.model_config,
                data_type=self.data_type,
                do_dynamic_quantize_convert=self.
                convert_config["do_dynamic_quantize_convert"],
                quant_config=self.quant_config,
                use_dynamic_ntk=self.model_config["use_dynamic_ntk"],
                use_logn_attn=self.model_config["use_logn_attn"],
                rotary_base=self.model_config["rotary_emb_base"],
                save_dir=self.model_path,
            )

            end = time.time()
            sec = end - begin
            print("convert model from HF finished, build time is {} seconds".format(sec))
        else:
            raise ValueError("torch model is not initialized")

    def convert_request_to_jsonstr(self, request):

        def skip_unserializable(value):
            """处理不能序列化的键值对并跳过"""
            return None

        # 将dataclass对象转换为字典
        request_dict = request.__dict__

        # 将字典转换为json格式的字符串
        json_str = json.dumps(request_dict,
                              ensure_ascii=False,
                              default=skip_unserializable)
        return json_str

    def print_inference_result(self, request):
        if self.engine.get_rank_id() != 0:
            return

        if self.verbose:
            print(f"***********************************")
            print(f"* Answer (dashinfer) for Request {request.id}")
            print(f"***********************************")
            print(f"** context_time: {request.context_time} s, generate_time: {request.generate_time} s\n")
            print(f"** encoded input, len: {request.in_tokens_len} **\n{request.in_tokens}\n")
            print(f"** encoded output, len: {request.out_tokens_len} **\n{request.out_tokens}\n")
            print(f"** text input **\n{request.in_text}\n")
            print(f"** text output **\n{request.out_text}\n")

    def print_inference_result_all(self, request_list):
        for request in request_list:
            if self.verbose:
                self.print_inference_result(request)

    def print_profiling_data(self, request_list, total_time=0):
        if self.engine.get_rank_id() != 0:
            return

        self.profiling_data.reset(self.engine_config["engine_max_batch"])
        for request in request_list:
            self.profiling_data.update(request.in_tokens_len,
                                       request.out_tokens_len,
                                       request.context_time,
                                       request.generate_time)

        self.profiling_data.update_df(total_time)
        self.profiling_data.show()

    def process_one_request_impl(self, request, stream_mode=False):
        torch_input = request.torch_input
        print(f"### generation_config: {request.gen_cfg}")

        output_ids = []
        request.start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                                time.localtime())
        time_start = time.time()
        status, request_handle, result_queue = self.engine.start_request(
            self.model_name, {
                "input_ids":
                torch.utils.dlpack.to_dlpack(torch_input["input_ids"]),
            },
            generate_config=request.gen_cfg)
        request.handle = request_handle
        request.queue = result_queue
        request.status = int(status)

        while True:
            status = request.queue.GenerateStatus()
            request.status = int(status)

            if status == allspark.GenerateRequestStatus.Init:
                pass
            elif status == allspark.GenerateRequestStatus.Generating:
                generated_elem = request.queue.Get()
                new_ids = generated_elem.ids_from_generate
                if len(output_ids) == 0 and len(new_ids) > 0:
                    request.context_time = time.time() - time_start
                    time_after_ctx = time.time()
                    request.out_text = ""

                if (len(new_ids) > 0):
                    output_ids.append(new_ids[0])

                request.out_tokens = output_ids
                request.out_tokens_len = len(output_ids)
                request.out_text = self.tokenizer.decode(
                    request.out_tokens, skip_special_tokens=True)
                if stream_mode:
                    yield request.out_text
            elif status == allspark.GenerateRequestStatus.GenerateFinished:
                request.generate_time = time.time() - time_after_ctx
                request.end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                                      time.localtime())
                break
            elif status == allspark.GenerateRequestStatus.GenerateInterrupted:
                print("[Error] Request interrupted!")
                break
            else:
                print(f"[Error] Unexpected status: {status}")
                break

        # for key, value in request.items():
        #     print(f"Key: {key}, Value's DataType: {type(value)}")
        self.engine.release_request(self.model_name,
                                    request_handle=request.handle)

    def process_one_request(self, request):
        for _ in self.process_one_request_impl(request, stream_mode=False):
            pass

    def process_one_request_stream(self, request):
        yield from self.process_one_request_impl(request, stream_mode=True)

    def save_config_as_json(config, file_path):
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    def get_config_from_json(file_path):
        config = None
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} doesn't exist.")

        with open(file_path, 'r') as f:
            config = json.load(f)
        return config
