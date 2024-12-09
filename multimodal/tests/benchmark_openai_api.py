'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    benchmark_openai_api.py
'''
import argparse
from dataclasses import dataclass, field
import time
from typing import List, Optional
from datasets import load_dataset
import queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import copy
import os
import random
import numpy as np

np.random.seed(123)
random.seed(0xCADE)

bench_request = []


@dataclass
class BenchRequest:
    conv_len: int = 0
    prompts: list[str] = field(default_factory=lambda: list())
    image_urls: list[str] = field(default_factory=lambda: list())
    max_tokens: int = 0
    context_time: Optional[float] = 0
    context_len: Optional[int] = 0
    vit_len: Optional[int] = 0
    decode_len: Optional[int] = 0
    decode_time: Optional[float] = 0
    vit_time: Optional[float] = 0
    session_stats: Optional[int] = 0
    vl_request: Optional[str] = None
    start_time: Optional[float] = 0
    request_id: Optional[int] = 0
    vit_preprocess_time: Optional[float] = 0
    vit_forward_time: Optional[float] = 0
    vit_len: Optional[int] = 0


class OpenAIAPIBenchmark:
    def __init__(self) -> None:
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:8000/v1"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        try:
            self.model = self.client.models.list().data[0].id
        except Exception:
            self.model = "model"

    def bench(self, requests):
        responses = []
        for request in requests:
            MESSAGES = []
            for i in range(request.conv_len * 2 - 1):
                if i % 2 == 0:
                    MESSAGES.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": request.prompts[i]}],
                        }
                    )
                    while len(request.image_urls) > 0:
                        MESSAGES[-1]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": request.image_urls.pop(),
                                },
                            }
                        )
                else:
                    MESSAGES.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": responses[(i - 1) // 2],
                                }
                            ],
                        }
                    )
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=MESSAGES,
                max_tokens=request.max_tokens,
                top_p=0.01,
                temperature=0.1,
                frequency_penalty=1.05,
            )
            responses.append(response.choices[0].message.content)
            update_request(
                request.request_id,
                conv_len=request.conv_len,
                status=None,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                duration=(time.time() - start_time),
            )


def update_request(
    id, conv_len=0, status=None, prompt_tokens=0, completion_tokens=0, duration=0.0
):
    global bench_request
    bench_request[id][conv_len - 1].context_len = prompt_tokens
    bench_request[id][conv_len - 1].decode_len = completion_tokens
    bench_request[id][conv_len - 1].decode_time = duration


def get_conv(conv):
    qa = []
    for c in conv:
        for i in range(len(c) // 2):
            qa.append(
                (c[2 * i]["value"].replace("<image>\n", ""), c[2 * i + 1]["value"])
            )
    return qa


def process_bench_request(
    images: List,
    conversations: List,
    request_num: int,
    multi_turns: int,
    response_lens: List,
    image_nums: List,
):
    global bench_request

    req_idx = 0
    img_idx = 0
    i = 0
    while i < request_num:
        img_urls = []
        for k in range(image_nums[i]):
            img_urls.append("file://" + images[img_idx + k])
        img_idx += image_nums[i]

        prompts = []
        requests = []
        for _ in range(multi_turns + 1):
            msg = conversations.pop()

            prompts.append(msg[0])
            requests.append(
                BenchRequest(
                    request_id=i,
                    conv_len=(len(prompts) // 2 + 1),
                    prompts=copy.deepcopy(prompts),
                    image_urls=copy.deepcopy(img_urls),
                    max_tokens=response_lens[i],
                )
            )
            req_idx += 1
            prompts.append(msg[1])
        bench_request.append(requests)
        i += 1

    return bench_request


def request_generator(freq, task_queue, request_list):
    time_interleave = 1.0 / freq
    for request in request_list:
        task_queue.put(request)
        time.sleep(time_interleave)


def async_request_processor(task_queue, future_list, bs):
    def done_callback(future):
        request = future.argument
        future.result()

    global model
    executor = ThreadPoolExecutor(max_workers=bs)

    start_time = time.time()
    start_benchmark_time = start_time
    while True:
        if not task_queue.empty():
            request = task_queue.get()
        elif task_generator_thread.is_alive():
            continue
        else:
            break
        future = executor.submit(model.bench, request)
        future.argument = request
        future.add_done_callback(done_callback)
        future_list.append(future)
    return start_benchmark_time


def gen_random_lens(distribution: str, len_mean, len_range, num_requests):
    if distribution == "uniform":
        if len_range == 0:
            return [len_mean for _ in range(num_requests)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        num_to_generate = list(
            map(lambda _: random.randint(low, high), range(num_requests))
        )
        return num_to_generate
    elif distribution == "exponential":
        np.random.seed(random.randint(0, 1e6))
        return [
            min(round(s), len_range)
            for s in np.random.exponential(scale=len_mean, size=num_requests)
        ]
    elif distribution == "capped_exponential":
        np.random.seed(random.randint(0, 1e6))
        response_lens = []
        while len(response_lens) < num_requests:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range:
                response_lens.append(sample)
        return response_lens
    else:
        raise ValueError(f"unknown distribution {distribution=}")


def print_profiling_data(total_timecost):
    global bench_request
    total_decode_tokens = 0
    total_decode_time = 0
    total_context_len = 0
    total_output_len = 0
    total_request = len(bench_request) * len(bench_request[0])
    for req in bench_requests:
        for r in req:
            total_decode_tokens += r.decode_len
            total_decode_time += r.decode_time

            total_context_len += r.context_len
            total_output_len += r.decode_len
    print(
        f"input token lens: {total_context_len / total_request :.2f} (average) / {total_context_len} (total) --- "
    )
    print(
        f"output token lens: {total_output_len / total_request :.2f} (average) / {total_output_len} (total) --- "
    )
    print(
        f"QPS: {total_request / total_timecost : .2f} requests/sec, TPS: {total_decode_tokens / total_timecost :.2f} tokens/sec"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--req-nums", type=int, default=100)
    parser.add_argument("--multi-turn", type=int, default=0)
    parser.add_argument("--response-mean", type=int, default=120)
    parser.add_argument("--response-len-range", type=int, default=64)
    parser.add_argument("--image-nums-mean", type=int, default=3)
    parser.add_argument("--image-nums-range", type=int, default=1)
    parser.add_argument("--frequency", type=float, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.prompt_file, split="train")
    qa = get_conv(ds["conversations"])

    response_lens = gen_random_lens(
        "uniform", args.response_mean, args.response_len_range, args.req_nums
    )
    image_nums = gen_random_lens(
        "uniform", args.image_nums_mean, args.image_nums_range, args.req_nums
    )
    image_list = []
    from PIL import Image
    for root, dirs, files in os.walk(args.image_folder):
        for filename in files:
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, filename)

                im = Image.open(file_path)
                width, height = im.size
                if width * height > 28 * 28 * 1024:
                    continue

                image_list.append(file_path)

    bench_requests = process_bench_request(
        image_list, qa, args.req_nums, args.multi_turn, response_lens, image_nums
    )

    model = OpenAIAPIBenchmark()

    global_start = time.time()

    task_queue = queue.Queue()
    future_list = []
    task_generator_thread = Thread(
        target=request_generator, args=(args.frequency, task_queue, bench_requests)
    )

    global_start_time = time.time()
    task_generator_thread.start()
    start_benchmark_time = async_request_processor(
        task_queue, future_list, args.batch_size
    )
    task_generator_thread.join()
    for future in future_list:
        future.result()

    global_end = time.time()
    print(f"Total time: {global_end - global_start_time :.2f} sec")
    print_profiling_data(global_end - global_start_time)
