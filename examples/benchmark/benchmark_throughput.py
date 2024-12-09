'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    benchmark_throughput.py
'''
import os
import sys
import time
import datetime
import argparse
import json
import torch
import queue
import random
import pandas as pd
import re
from tabulate import tabulate

import signal
import sys

from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, List

from dashinfer import allspark
from PromptLoader import *
from InspectPrompt import inspect_prompt_list

request_cnt = 0
running = True
long_time_test_duration_seconds = 0


def sample_request(list_of_req, n):
    if len(list_of_req) <= n:
        return list_of_req  # 如果n比列表长度大或相等，直接返回整个列表
    else:
        return random.sample(list_of_req, n)  # 否则，从列表中均匀随机抽取n个元素

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
    ref_answer: Optional[str] = None
    ref_answer_tokens: Optional[List[int]] = None
    ref_answer_tokens_len: Optional[int] = None
    prefix_len: int = 0
    prefix_len_gpu: int = 0
    prefix_len_cpu: int = 0
    status: Optional[int] = None
    gen_cfg: Optional[dict] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    context_time: float = 0
    generate_time: float = 0
    valid: bool = False
    handle: Any = field(default=None)
    queue: Any = field(default=None)


class ProfilingData():
    def __init__(self, model_name, config):
        self.device_type = config.device_type
        self.device_num_col_name = "NUMA_num" if self.device_type == "CPU" else "Device_num"
        self.num_dev = len(config.device_ids)
        self.model_name = model_name.replace(" ", "_")
        self.df = pd.DataFrame(columns=["Model_name", "Device_type", "Thread_num", self.device_num_col_name,
                                        "Batch_size", "Request_num", "Avg_in_tokens", "Avg_out_tokens",
                                        "Avg_context_time(s)", "Avg_generate_time(s)",
                                        "Avg_throughput(token/s)", "Throughput(token/s)", "QPS"])

        self.detail_df = pd.DataFrame(columns=["Request_id", "Model_name", "Device_type", "Thread_num", self.device_num_col_name,
                                               "In_Tokens", "Out_Tokens", "Prefix_Len(GPU)", "Prefix_Len(CPU)", "Ctx_Secs", "Gen_Secs"])
        self.print_df = self.df
        self.reset(config.engine_max_batch)

    def reset(self, max_batch):
        self.max_batch = max_batch
        self.total_input_len = 0.0
        self.total_output_len = 0.0
        self.total_context_time = 0.0
        self.total_generate_time = 0.0
        self.total_prefix_cache_gpu = 0.0
        self.total_prefix_cache_cpu = 0.0
        self.total_throughput = 0.0
        self.total_time = 0.0
        self.avg_input_len = 0.0
        self.avg_output_len = 0.0
        self.avg_context_time = 0.0
        self.avg_generate_time = 0.0
        self.avg_throughput = 0.0
        self.request_num = 0
        self.qps = 0.0
        self.df = self.df.iloc[0:0]
        self.detail_df = self.detail_df.iloc[0:0]


    def update(self, request_id, in_len, out_len, prefix_len_gpu, prefix_len_cpu, ctx_time, gen_time):
        self.request_num += 1
        self.batch_size = self.request_num if self.request_num < self.max_batch else self.max_batch
        self.total_input_len += in_len
        self.total_output_len += out_len
        self.total_context_time += ctx_time
        self.total_generate_time += gen_time
        self.total_prefix_cache_gpu += prefix_len_gpu
        self.total_prefix_cache_cpu += prefix_len_cpu

        self.avg_input_len = self.total_input_len / self.request_num
        self.avg_output_len = self.total_output_len / self.request_num
        self.avg_context_time = self.total_context_time / self.request_num
        self.avg_generate_time = self.total_generate_time / self.request_num
        self.avg_throughput = self.avg_output_len / self.avg_generate_time
        self.total_throughput = self.avg_throughput * self.batch_size

        self.detail_df = pd.concat([self.detail_df, pd.DataFrame({"Request_id": [request_id],
                                                                  "Model_name": [self.model_name],
                                                                  "Device_type": [self.device_type],
                                                                  "Thread_num": [1],
                                                                  self.device_num_col_name: [self.num_dev],
                                                                  "In_Tokens": [in_len],
                                                                  "Out_Tokens": [out_len],
                                                                  "Prefix_Len(GPU)": [prefix_len_gpu],
                                                                  "Prefix_Len(CPU)": [prefix_len_cpu],
                                                                  "Ctx_Secs": [ctx_time],
                                                                  "Gen_Secs": [gen_time],
                                                                  })])

    def update_df(self, total_time):
        self.total_time = total_time
        if self.total_time > 0.0:
            self.qps = self.request_num / self.total_time
        else:
            self.qps = float("nan")
            print(f"[Warning] total timecost is not set, cannot calculate QPS.")

        self.df = pd.concat([self.df,
                             pd.DataFrame({
                                 "Model_name": [self.model_name],
                                 "Device_type": [self.device_type],
                                 "Thread_num": [1],
                                 self.device_num_col_name: [self.num_dev],
                                 "Batch_size": [self.batch_size],
                                 "Request_num": [self.request_num],
                                 "Avg_in_tokens": [self.avg_input_len],
                                 "Avg_out_tokens": [self.avg_output_len],
                                 "Prefix_Cache(GPU)": [f"{self.total_prefix_cache_gpu * 100 / self.total_input_len}%"],
                                 "Prefix_Cache(CPU)": [f"{self.total_prefix_cache_cpu * 100 / self.total_input_len}%"],
                                 "Avg_context_time(s)": [self.avg_context_time],
                                 "Avg_generate_time(s)": [self.avg_generate_time],
                                 "Avg_Req_Tput(token/s)": [self.avg_throughput],
                                 "Total_Tput(token/s)": [self.total_throughput],
                                 "QPS": [self.qps]
                             })
                             ], ignore_index=True)

        self.print_df = pd.concat([self.print_df, self.df])

    def show(self):
        col_to_print = ["Batch_size", "Request_num", "Avg_in_tokens", "Avg_out_tokens", "Avg_context_time(s)",
                        "Avg_generate_time(s)", "Prefix_Cache(GPU)", "Prefix_Cache(CPU)",
                        "Avg_Req_Tput(token/s)", "Total_Tput(token/s)", "QPS"]
        df_to_print = self.print_df.loc[:, col_to_print]
        print(tabulate(df_to_print, showindex=False, headers="keys", tablefmt="outline", numalign="left", floatfmt=".3f"))
        # print(tabulate(self.df, showindex=False, headers="keys", tablefmt="psql", numalign="left", floatfmt=".3f"))

    def save_report(self):
        current_datetime = datetime.datetime.now()
        # date_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        file_name_report = "report_" + self.model_name + ".csv"
        file_name_detail = "detail_" + self.model_name + ".csv"
        # use append to accumulate different thread data together.
        # not create duplicate header
        detail_file_exists = os.path.isfile(file_name_detail)
        report_file_exists = os.path.isfile(file_name_report)

        self.detail_df.to_csv(
            file_name_detail, index=False, mode='a', header=not detail_file_exists, float_format="%.3f")

        self.df.to_csv(
            file_name_report, index=False, mode='a', header=not report_file_exists, float_format="%.3f")

        print(f"Report and Detail have been written to files: \n"
              f"Report: {file_name_report}\n"
              f"Detail: {file_name_detail}\n")



class ProfileTool():
    @staticmethod
    def print_profiling_data(engine, profile_data, request_list, max_batch, total_time=0):
        if engine.get_rank_id() != 0:
            return

        profile_data.reset(max_batch)
        for request in request_list:
            if not request.valid:
                continue
            profile_data.update(request.id, request.in_tokens_len, request.out_tokens_len,
                                request.prefix_len_gpu, request.prefix_len_cpu,
                                request.context_time, request.generate_time)

        profile_data.update_df(total_time)
        profile_data.show()
        profile_data.save_report()

    @staticmethod
    def print_prefix_cache_stat(engine, request_list):
        if engine.get_rank_id() != 0:
            return

        total_tokens = 0
        gpu_cached_tokens = 0
        cpu_cached_tokens = 0
        total_cached_tokens = 0
        for request in request_list:
            if not request.valid:
                continue

            total_tokens += request.in_tokens_len
            gpu_cached_tokens += request.prefix_len_gpu
            cpu_cached_tokens += request.prefix_len_cpu

        total_cached_tokens = gpu_cached_tokens + cpu_cached_tokens
        if total_tokens > 0:
            gpu_hit_rate = gpu_cached_tokens / total_tokens * 100
            cpu_hit_rate = cpu_cached_tokens / total_tokens * 100
            total_hit_rate = total_cached_tokens / total_tokens * 100
        else:
            gpu_hit_rate = 0
            cpu_hit_rate = 0
            total_hit_rate = 0

#        msg = "PrefixCache Info: "
#        msg += f"total_tokens: {total_tokens}\n"
#        msg += f"total_cached_tokens: {total_cached_tokens}, hit_rate: {total_hit_rate:.2f}%\n"
#        msg += f"gpu_cached_tokens: {gpu_cached_tokens}, hit_rate: {gpu_hit_rate:.2f}%\n"
#        msg += f"cpu_cached_tokens: {cpu_cached_tokens}, hit_rate: {cpu_hit_rate:.2f}%\n"
#        print(msg)



def process_prompt(tokenizer, model_name, prompt : list, default_gen_config = None, ref_answer=None, gen_cfg=None, need_format=False, is_random=False):
    def add_prefix_suffix(model_type, prompt):
        if model_type in ["Baichuan_v1", "Baichuan_v2"]:
            prompt = "<reserved_106>" + prompt + "<reserved_107>"

        if model_type in ["Qwen_v10", "Qwen_v15", "Qwen_v20"]:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

        if model_type in ["ChatGLM_v2"]:
            prompt = "[Round 1]\n\n问：" + prompt + "\n\n答："

        if model_type in ["ChatGLM_v3"]:
            prompt = "<|user|>\n" + prompt + "<|assistant|>"

        if model_type in ["ChatGLM_v4"]:
            prompt = "[gMASK] <sop> " + "<|user|>\n" + prompt + "<|assistant|>\n"

        if model_type in ["LLaMA_v3"]:
            prompt = "<|begin_of_text|>" + \
                     "<|start_header_id|>system<|end_header_id|>\n\n" + \
                     "You are a helpful assistant." + "<|eot_id|>" + \
                     "<|start_header_id|>user<|end_header_id|>\n\n" + \
                     prompt + "<|eot_id|>" + \
                     "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    if prompt == None or len(prompt) == 0:
        raise ValueError("No valid input")

    is_encoded = False
    if isinstance(prompt[0], list):
        is_encoded = True
    elif isinstance(prompt[0], str):
        is_encoded = False
    else:
        raise ValueError("Invalid inputs")

    request_list = []
    for i in range(len(prompt)):
        input_text = ""
        if is_encoded == False:
            if need_format:
                input_text = add_prefix_suffix("Qwen_v15", prompt[i] )
            else:
                input_text = prompt[i]
            p_tokens = tokenizer(input_text, return_tensors='pt')
            input_ids = p_tokens["input_ids"].tolist()
        else:
            if not is_random:
                input_text = tokenizer.decode(prompt[i], skip_special_tokens=False)
            input_ids = [prompt[i]]

        torch_input = {
            "input_ids": torch.Tensor(input_ids).to(torch.int64),
        }
        request = Request()
        global request_cnt
        request.id = request_cnt
        request_cnt += 1
        # request.id = i
        request.model_info = model_name
        request.in_text = input_text
        request.in_tokens = input_ids[0]
        request.in_tokens_len = len(input_ids[0])
        request.torch_input = torch_input
        request.ref_answer = None if ref_answer == None else ref_answer[i]
        request.ref_answer_tokens = None if ref_answer == None else tokenizer(request.ref_answer)["input_ids"]
        request.ref_answer_tokens_len = 0 if ref_answer == None else len(request.ref_answer_tokens)
        if gen_cfg == None:
            request.gen_cfg = default_gen_config
        else:
            request.gen_cfg = copy.deepcopy(gen_cfg[i])
        # request max len is input len + output len
        origin_len = request.gen_cfg.build()['max_length']
        request.gen_cfg.update({'max_length' :   request.in_tokens_len + config.test_max_output})

        request_list.append(request)

    '''
    if self.verbose:
        for i in range(len(request_list)):
            request = request_list[i]
            msg = "*****************\n"
            msg += f"* Request {i}\n"
            msg += "*****************\n"
            msg += f"** text input **\n{repr(request.in_text)}\n\n"
            msg += f"** encoded input, len: {request.in_tokens_len} **\n{request.in_tokens}\n\n"
            msg += f"** torch input **\n"
            for key, value in request.torch_input.items():
                msg += f"{key}, shape: {value.shape}\n{value}\n"
            print(msg)
    '''

    return request_list



class BenchmarkConfig():
    def __init__(self, args):
        self.model_path = args.model_path
        self.was_from_modelscope = args.modelscope
        self.was_weight_quant = args.weight_quant
        self.weight_only_quant = args.weight_only_quant
        self.cache_mode = args.cache_mode
        self.device_type = args.device_type
        self.device_ids = args.device_ids

        self.engine_max_length = args.engine_max_length
        self.engine_max_batch = args.engine_max_batch
        self.engine_enable_prefix_cache = args.engine_enable_prefix_cache

        self.test_qps = args.test_qps
        self.test_sample_size = args.test_sample_size
        self.test_max_output = args.test_max_output
        self.test_dataset_path = args.test_dataset_path
        self.test_random_input = args.test_random_input
        self.test_dataset_id = args.test_dataset_id
        self.prefix_cache_list = args.prefix_cache_rate_list

        self.guided_decode = args.guided_decode

        if self.engine_max_length - self.test_max_output <= 0:
            raise ValueError("engine max length too small, should at least largeer than test max output")

        if self.test_dataset_path and self.test_dataset_id == None:
            raise ValueError("dataset id is missing.")

        if self.test_dataset_path and self.test_random_input == True:
            raise ValueError("dataset and random input cannot be enabled in same time.")

        if len(self.prefix_cache_list) != 0:
            print(f"prefix cache test list:{self.prefix_cache_list}")
            if self.engine_enable_prefix_cache == False:
                print("got prefix cache list, enable prefix cache enbale config.")
                self.engine_enable_prefix_cache = True

        for rate in self.prefix_cache_list:
            if rate < 0 or rate > 1.0:
                raise ValueError("prefix cache rate must between (0.0, 1.0]")

def one_request(request, engine, model_loader, stream_mode):
    torch_input = request.torch_input
    print("one request\n")
    print(f"### generation_config: {request.gen_cfg}")

    output_ids = []
    request.start_timestamp =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_start = time.time()


    status, request_handle, result_queue = engine.start_request_ids("model",
                                                                    model_loader,
                                                                    request.in_tokens,
                                                                    request.gen_cfg)
    if status != allspark.AsStatus.ALLSPARK_SUCCESS:
        request.valid = False
        print("[Error] Request failed to start!")
        sys.exit(1)


    request.valid = True
    request.handle = request_handle
    request.queue = result_queue
    request.status = int(status)

    while True:
    # status = self.engine.get_request_status(self.model_name, request.queue)
        status = request.queue.GenerateStatus()
        request.status = int(status)

        if status == allspark.GenerateRequestStatus.Init:
            pass
        elif status == allspark.GenerateRequestStatus.Generating or status == allspark.GenerateRequestStatus.GenerateFinished:
        # new_ids = self.engine.get_wait(self.model_name, request.queue)
            generated_elem = request.queue.Get()
            if generated_elem is not None:
                new_ids = generated_elem.ids_from_generate
            # prefix_cache_len = generated_elem.prefix_cache_len
                if len(output_ids) == 0 and len(new_ids) > 0:
                    request.context_time = time.time() - time_start
                    time_after_ctx = time.time()
                    request.out_text = ""
                    request.prefix_len = generated_elem.prefix_cache_len
                    request.prefix_len_gpu = generated_elem.prefix_len_gpu
                    request.prefix_len_cpu = generated_elem.prefix_len_cpu

                if (len(new_ids) > 0):
                    output_ids.extend(new_ids)

                request.out_tokens = output_ids
                request.out_tokens_len = len(output_ids)
                request.out_text = model_loader.get_tokenizer().decode(request.out_tokens, skip_special_tokens=True)
                #if stream_mode:
                    #yield request.out_text

            if status == allspark.GenerateRequestStatus.GenerateFinished:
                request.generate_time = time.time() - time_after_ctx
                request.end_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                break
        elif status == allspark.GenerateRequestStatus.GenerateInterrupted:
            request.valid = False
            print("[Error] Request interrupted!")
            break
        else:
            request.valid = False
            print(f"[Error] Unexpected status: {status}")
            break

    engine.stop_request("model", request_handle)
    engine.release_request("model", request_handle=request.handle)
    return request

def request_generator(freq, task_queue, request_list):
    time_interleave = 1.0 / freq
    for request in request_list:
        task_queue.put(request)
        time.sleep(time_interleave)

def request_processor(engine, config, model_loader, task_queue, future_list, progress_bar, task_generator_thread):
    def done_callback(f):
        request = f.argument
        f.result()
        progress_bar.update(1)
        # engine_helper.print_inference_result(request)

    # 创建线程池
    executor = ThreadPoolExecutor(max_workers=config.engine_max_batch)
    print(f"current level : {config.engine_max_batch}")

    # 使用线程池处理任务
    while True:
        if not task_queue.empty():
            # 从队列获取任务，如果队列为空，会阻塞直到队列中有新的任务
            # print("task not empty")
            request = task_queue.get()
        elif task_generator_thread.is_alive():
            # 如果队列为空，判断是否还在生成任务，如果还在生成任务就继续
            # print("task queue alive.")
            continue
        else:
            # 如果队列为空，并且已经不在生成任务了，就跳出
            # print("task empty")
            break

        def test_fun(one):
            print("test fun", one)
            return 1
        # 将任务提交到线程池
        kwargs = {'request': request, 'engine': engine, 'model_loader': model_loader, 'stream_mode': False}

        f = executor.submit(one_request, **kwargs)
        f.argument = request
        f.add_done_callback(done_callback)
        future_list.append(f)

def test_model_stress(config: BenchmarkConfig):

    from tqdm import tqdm
   ###

    freq = config.test_qps
    request_num = config.test_sample_size

    prompt_list = []
    original_insepct = []
    filtered_insepct = []

    need_format = False
    if config.test_random_input:
        prompt_list = random_prompt(config.test_sample_size, config.engine_max_length - config.test_max_output)
        print(f"random generated prompt")
        original_insepct = prompt_list
        filtered_insepct = original_insepct
    else:

        stress_file = config.test_dataset_path
        dataset_name = os.path.basename(stress_file)
        prompt_list, need_format = test_prompt_from_jsonl(
            stress_file,
            config.test_dataset_id)

        original_insepct = inspect_prompt_list(prompt_list, f"{dataset_name}_original", plot=True)

        max_len=config.engine_max_length - config.test_max_output
        print(f"[INFO] Max prompt length: {max_len}")
        print(f"[INFO] Number of prompts before filter: {len(prompt_list)}")
        prompt_list = list(filter(lambda prompt: len(prompt) < max_len, prompt_list))
        print(f"[INFO] Number of prompts after filter: {len(prompt_list)}")

        if (len(prompt_list) == 0):
            raise ValueError("filter prompt cannot be 0")
        filtered_insepct = inspect_prompt_list(prompt_list, f"{dataset_name}_filtered", plot=True)



    # random select from dataset, make distribution towards uniform
    prompt_list = sample_request(prompt_list, request_num)

    ## start engine.

    in_memory = False
    # 示例函数，用来运行模型

    safe_model_name = str(config.model_path).replace("/", "_")
    model_real_path = config.model_path
    # 实际上你需要在这里实现你的模型逻辑
    if config.was_from_modelscope:
        import modelscope
        model_real_path = modelscope.snapshot_download(config.model_path)
        # replace mmodel with path.

    model_loader = allspark.HuggingFaceModel(model_real_path, safe_model_name, in_memory_serialize=in_memory,
                                             trust_remote_code=True)


    engine = allspark.Engine()
    (model_loader.load_model()
     .read_model_config()
     .serialize(engine, model_output_dir=".", enable_quant=config.was_weight_quant, weight_only_quant=config.weight_only_quant)
     .free_model())

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder("model", config.device_type,
                                                                           config.device_ids, max_batch=config.engine_max_batch)
    runtime_cfg_builder.max_length(config.engine_max_length)

    runtime_cfg_builder.prefill_cache(config.engine_enable_prefix_cache)

    runtime_cfg = runtime_cfg_builder.build()

    engine.install_model(runtime_cfg)

    model_loader.free_memory_serialize_file()

    engine.start_model('model')

    # like change to engine max length
    tokenizer = model_loader.read_model_config().init_tokenizer().get_tokenizer()


    gen_cfg_builder = model_loader.create_reference_generation_config_builder(runtime_cfg)

    if config.test_random_input:
        gen_cfg_builder.early_stopping(False)

    if config.guided_decode:
        gen_cfg_updates = {}
        gen_cfg_updates["response_format"] = {"type": "json_object"}
        gen_cfg_builder.update(gen_cfg_updates)

    no_cache_request_list = process_prompt(tokenizer, safe_model_name, default_gen_config=gen_cfg_builder,
                                       prompt=prompt_list[:request_num], need_format=need_format,
                                       is_random = config.test_random_input)


    request_rounds = []
    request_rounds.append(no_cache_request_list)

    if len(config.prefix_cache_list) > 0:
        # make sure prefix cache rate in decending order.
        config.prefix_cache_list = sorted(config.prefix_cache_list, key=abs, reverse=True)

        # generate request.
        for cache_rate in config.prefix_cache_list:
            tmp_prompts = copy.deepcopy(prompt_list)
            for idx in range(len(tmp_prompts)):
               change_id_position =  (int)(len(tmp_prompts[idx]) * cache_rate)
               tmp_prompts[idx][change_id_position] += 1  # change one token make prefix cache invalid in that position.
            request_list = process_prompt(tokenizer, safe_model_name, default_gen_config=gen_cfg_builder,
                                          prompt=tmp_prompts, need_format=need_format,
                                          is_random = config.test_random_input)
            print(f"process prompt for {cache_rate}")

            request_rounds.append(request_list)


    progress_bar = tqdm(total=request_num)
    profile_data = ProfilingData(safe_model_name, config)

    print(f"request rounds list len: {len(request_rounds)}")
    print("Finish Request generate, start benchmark...")

    running_seconds = 2;
    if long_time_test_duration_seconds > 0:
        running_seconds = long_time_test_duration_seconds


    end_time = time.time() + running_seconds

    while running and time.time() < end_time:
        for request_list in request_rounds:
            if not running:
                break;
            task_queue = queue.Queue()
            future_list = []

            global_start = time.time()

            # 开启一个线程产生任务
            task_generator_thread = Thread(target=request_generator, args=(freq, task_queue, request_list))
            task_generator_thread.start()

            request_processor(engine, config, model_loader, task_queue, future_list, progress_bar, task_generator_thread)

            # 等待任务生成线程结束
            task_generator_thread.join()

            # 阻塞，等待所有任务完成
            print("wait for all future.")
            for future in future_list:
                future.result()

            global_end = time.time()

            progress_bar.reset()

            total_timecost = global_end - global_start
        #    engine_helper.print_inference_result_all(request_list)
            ProfileTool.print_profiling_data(engine, profile_data,  request_list, config.engine_max_batch, total_timecost)
            ProfileTool.print_prefix_cache_stat(engine, request_list)
            print(f"total timecost: {total_timecost} s")
            if not config.test_random_input:
                print(f"Dataset {dataset_name} inspection (original):")
                print(original_insepct)
                print(f"Dataset {dataset_name} inspection (filtered with max prompt length {max_len}):")
                print(filtered_insepct)

    engine.stop_model('model')
    engine.release_model('model')

    progress_bar.close()
 #   engine_helper.uninit_allspark_engine()


def parse_duration(duration_str):
    """
    convert time string to seconds.
    - 1h  1 hour
    - 30m 30 minutes
    - 10s 10 seconds
    - 1h30m10s combo string
    """
    pattern = re.compile(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?')
    match = pattern.fullmatch(duration_str)
    if not match:
        raise ValueError(f"无法解析的持续时间格式: {duration_str}")

    hours, minutes, seconds = match.groups()
    total_seconds = 0
    if hours:
        total_seconds += int(hours) * 3600
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += int(seconds)

    return total_seconds

def parse_float_list(value):
    try:
        return [float(item) for item in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("float number required, seperated by ','")

def signal_handler(sig, frame):
    global running
    logging.info("signal received, exiting")
    running = False

if __name__ == '__main__':

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    parser = argparse.ArgumentParser(description='Benchmark model with random data or provided data list.')
    parser.add_argument('--model_path', type=str, required=False, help='The name of the model to run', default="qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--modelscope", type=bool, required=False, default=True, help="use modelscope download model")
    parser.add_argument("--weight_quant", type=bool, required=False, default=False, help="use weight quant")
    parser.add_argument("--weight_only_quant", type=bool, required=False, default=False, help="do weight only quant")
    parser.add_argument("--cache_mode", type=str, required=False, default="default", help="kv cache mode : [defualt,8bit,4bit]")
    parser.add_argument("--device_type", type=str, required=False, default="CUDA", help="device tyep [CUDA,CPU]")
    parser.add_argument("--device_ids", type=list_of_ints, required=False, default="0", help="device ids like 0,1")
    parser.add_argument("--verbose", type=bool, required=False, default=False, help="verbose logging")


    parser.add_argument("--engine_max_length", type=int, required=False, default=8192, help="engine max length, dataset will be filtered by this length.")
    parser.add_argument("--engine_max_batch", type=int, required=False, default=32, help="engine max batch, this value same as test concurrency.")
    parser.add_argument("--engine_enable_prefix_cache", type=bool, required=False, default=False, help="enable prefix cache.")

    parser.add_argument("--test_qps", type=float, required=True, help="send test request by seconds.")
    parser.add_argument("--test_sample_size", type=int, required=False, default=100, help="how many sample data should be tested.")
    parser.add_argument("--test_max_output", type=int, required=False, default=200, help="max output size ")
    parser.add_argument("--test_dataset_path", type=str, required=False, default=None, help="data set used in benchmark.")
    parser.add_argument("--test_dataset_id", type=int, required=False, default=None, help="dataset id choose between [0,1,2,3]")
    parser.add_argument("--test_random_input", action="store_true", default=False, help="use random data to benchmark")
    parser.add_argument("--guided_decode", type=bool, required=False, default=False, help="enable guided decode for json object.")
    parser.add_argument("--prefix_cache_rate_list", type=parse_float_list, required=False, default=[], help="add one cache running list, for benchmark different cache hit rate result, must be in decending order, like  0.99, 0.9, 0.6, 0.3, benchmark result will in multiple line, first line was total cache miss")

    parser.add_argument(
        '--duration',
        type=str,
        default='0s',
        help='long time testing duration setting，eg: 1h, 30m, 45s, 1h30m10s'
    )

    args = parser.parse_args()
    print(f"test start with {args}")

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        long_time_test_duration_seconds = parse_duration(args.duration)
    except ValueError as e:
        print(f"time setting error: {e}, time str : {args.duration}")
        sys.exit(1)

    print(f"long time test: {long_time_test_duration_seconds}")

    if args.verbose:
        os.putenv("ALLSPARK_TIME_LOG", "1")
        os.putenv("HIE_LOG_STATUS_INTERVAL", "1")

    config = BenchmarkConfig(args)

    test_model_stress(config)
