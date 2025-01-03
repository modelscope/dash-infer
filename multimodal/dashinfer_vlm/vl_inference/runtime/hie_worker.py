'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    hie_worker.py
'''
from ..utils.hie_allspark import *
from ..utils.hie import *
from ..utils.qwen_vl_status import VLStatusCode
from .vit import Vit, VitStatus
from ..utils.config import VitConfig

import threading
import queue

from ..utils.trt.vit_process import VisualTRT_V2
import torch
import numpy as np
import time
import logging
from ..utils.env import getenv
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class VitRequest:
    request_id: Optional[int] = -1
    # vit_id == < 0 means exit
    vit_id: Optional[int] = -1
    # image or audio; can be url/torch/np array
    vit_input: Optional[Any] = None
    # image grid thw, used in qwen2-vl
    vit_grid_thw: Optional[List[int]] = None
    # audio attn mask, used in qwen2-al
    audio_attn_mask: Optional[torch.Tensor] = None
    start_time: Optional[float] = field(default_factory=time.time)


class HieWokerImpl(threading.Thread):
    def __init__(
        self,
        request_queue,
        result_queue,
        device,
        model_path,
        exit_signal,
        precision="fp16",
        profile=False,
        backend="tensorrt",
        trt_vit_config=None,
    ):
        super().__init__()
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.precision = precision
        self.device = device
        self.profile = profile
        self.model_path = model_path
        self.exit_signal = exit_signal
        self.model_type = getenv("QWEN_MODEL_TYPE", "QWEN2-VL").upper()
        self.backend = backend
        self.trt_vit_config = trt_vit_config

    # get audio seq len to truncate embeddings
    def get_audio_seq_len(self, audio_attention_mask):
        non_zero_indices = torch.nonzero(audio_attention_mask.squeeze()).squeeze()
        audio_seq_len = (
            non_zero_indices[0]
            if non_zero_indices.numel() > 0
            else audio_attention_mask.shape[1]
        )
        return (audio_seq_len - 2) // 2 + 1

    def run(self):
        # init hie model
        if self.model_type == "QWEN2-VL":
            # warm up
            image = torch.randn(
                2436,
                1176,
                dtype=torch.float16 if self.precision == "fp16" else torch.float32,
            )
            grid_thw = torch.tensor([[1, 58, 42]], dtype=torch.int64)
            first_grid = grid_thw[0, 0].item()
            batch_tensor = torch.zeros(first_grid)
            dict(
                {
                    "image": np.array(image.contiguous().cpu().numpy()),
                    "grid_thw": np.array(grid_thw.contiguous().cpu().numpy()),
                    "batch": np.array(batch_tensor.contiguous().cpu().numpy()),
                }
            )
            if self.backend == "tensorrt":
                self.model = VisualTRT_V2(
                    vit_engine_path=self.model_path, trt_vit_config=self.trt_vit_config
                )
            elif self.backend == "transformers":
                self.model = self.model_path.to(self.device)
                with torch.no_grad():
                    self.model(image.to(self.device), grid_thw=grid_thw.to(self.device))
            elif self.backend == "hie":
                raise NotImplementedError
        else:
            raise NotImplementedError

        # init finish
        init_vit = Vit(status=VitStatus.INIT_DONE)
        self.result_queue.put(init_vit)
        while True:
            task = self.request_queue.get()
            if task.vit_id < 0:
                break
            if time.time() - task.start_time > 40:
                logging.error(
                    f"request_queue get timeout and skip processing request_id:{task.request_id}"
                )
                continue
            # logging.info(f"ready to get run vit res request id:{task.request_id} id:{task.vit_id}")
            try:
                vit_result = self.process_request(task)
            except Exception as e:
                logging.error(
                    f"run vit res request id:{task.request_id} id:{task.vit_id} error:{e}"
                )
                task.status = VLStatusCode.VL_OTHER_ERROR
            self.result_queue.put(vit_result)
            # logging.info(f"put run vit res request id:{task.request_id} id:{task.vit_id}")

    def get_vit_result(self, image, input_info):
        output = None
        if self.model_type == "QWEN1-VL":
            output = self.model(image, use_flashattn=True)
        elif self.model_type == "QWEN2-VL":
            # grid_thw = torch.tensor(
            #                 [input_info["vit_grid_t"], input_info["vit_grid_h"], input_info["vit_grid_w"]], dtype=torch.int32
            #             ).unsqueeze(0)
            grid_thw = np.array(
                [
                    [
                        input_info["vit_grid_t"],
                        input_info["vit_grid_h"],
                        input_info["vit_grid_w"],
                    ]
                ]
            )
            grid_thw = torch.from_numpy(grid_thw)
            first_grid = grid_thw[0, 0].item()
            grid_thw = grid_thw.to(dtype=torch.int32, device=self.device)
            batch_tensor = torch.zeros(first_grid).to(
                dtype=torch.int32, device=self.device
            )
            # output = self.model(image, grid_thw, batch_tensor)
            if self.backend == "tensorrt":
                output = self.model(image, grid_thw, batch_tensor)
            elif self.backend == "transformers":
                with torch.no_grad():
                    output = self.model(image, grid_thw=grid_thw)
            # output = torch.load("/root/workspace/dash-infer/multimodal/image_embeds_hf.pt", weights_only=True).to(device="cuda", dtype=torch.float32)
            # print("vit output shape: ", output.shape)
        else:
            output = self.model(image.contiguous().to(self.device), input_info)
        
        return output

    def process_request(self, task: VitRequest) -> None:
        # preprocess
        preprocess_start = time.time() if self.profile else None
        status = VLStatusCode.VL_SUCCESS
        image = task.vit_input
        image_info = None
        audio_output_lengths = None
        vit_res = Vit(
            request_id=task.request_id, vit_id=task.vit_id, status=VitStatus.VL_SUCCESS
        )
        dtype = torch.half if self.precision == "fp16" else torch.float
        if isinstance(image, str):
            logging.error(
                "image/audio do not support url input, image should be preprocess before sending to vit"
            )
            status = VLStatusCode.VL_IMAGE_FORMAT_ERROR
            vit_res.status = status
            return vit_res
        elif isinstance(image, torch.Tensor):
            if self.model_type == "QWEN2-VL" and (
                task.vit_grid_thw is None or len(task.vit_grid_thw) != 3
            ):
                logging.error(
                    "images_infos should be set list[t,h,w] when using vit version 2"
                )
                status = VLStatusCode.VL_IMAGE_FORMAT_ERROR
                vit_res.status = status
                return vit_res
            if self.model_type == "QWEN2-AL" and (
                task.audio_attn_mask is None
                or not isinstance(task.audio_attn_mask, torch.Tensor)
            ):
                logging.error(
                    "audios_info['audio_attention_mask'] should be set to torch tensor when using qwen2-al"
                )
                status = VLStatusCode.VL_IMAGE_FORMAT_ERROR
                vit_res.status = status
                return vit_res
            if task.audio_attn_mask is not None:
                if image.dim() == 2:
                    image = image.unsqueeze(0)
                if task.audio_attn_mask.dim() == 1:
                    task.audio_attn_mask = task.audio_attn_mask.unsqueeze(0)
                audio_output_lengths = self.get_audio_seq_len(task.audio_attn_mask)
                image_info = task.audio_attn_mask.to(dtype=dtype, device=self.device)
            if task.vit_grid_thw and len(task.vit_grid_thw) == 3:
                image_info = {
                    "vit_grid_t": task.vit_grid_thw[0],
                    "vit_grid_h": task.vit_grid_thw[1],
                    "vit_grid_w": task.vit_grid_thw[2],
                }
                vit_res.vit_grid_thw = [
                    image_info["vit_grid_t"],
                    image_info["vit_grid_h"],
                    image_info["vit_grid_w"],
                ]
            image = image.to(dtype=dtype, device=self.device)
        elif isinstance(image, np.ndarray):
            if self.model_type == "QWEN2-VL" and (
                task.vit_grid_thw is None or len(task.vit_grid_thw) != 3
            ):
                logging.error(
                    "images_infos should be set list[t,h,w] when using vit version 2"
                )
                status = VLStatusCode.VL_IMAGE_FORMAT_ERROR
                vit_res.status = status
                return vit_res
            if self.model_type == "QWEN2-AL" and (
                task.audio_attn_mask is None
                or not isinstance(task.audio_attn_mask, np.ndarray)
            ):
                logging.error(
                    "audios_info['attention_mask'] should be set to torch tensor when using qwen2-al"
                )
                status = VLStatusCode.VL_IMAGE_FORMAT_ERROR
                vit_res.status = status
                return vit_res
            if task.audio_attn_mask is not None:
                if image.ndim == 2:
                    image = np.expand_dims(image, axis=0)
                if task.audio_attn_mask.ndim == 1:
                    task.audio_attn_mask = np.expand_dims(task.audio_attn_mask, axis=0)
                audio_output_lengths = self.get_audio_seq_len(task.audio_attn_mask)
                image_info = task.audio_attn_mask.to(dtype=dtype, device=self.device)
            if task.vit_grid_thw and len(task.vit_grid_thw) == 3:
                image_info = {
                    "vit_grid_t": task.vit_grid_thw[0],
                    "vit_grid_h": task.vit_grid_thw[1],
                    "vit_grid_w": task.vit_grid_thw[2],
                }
                vit_res.vit_grid_thw = [
                    image_info["vit_grid_t"],
                    image_info["vit_grid_h"],
                    image_info["vit_grid_w"],
                ]
            image = torch.from_numpy(image).to(dtype=dtype, device=self.device)
        else:
            logging.error(
                f"image type must be np.ndarray/torch.Tensor, but got {type(image)}"
            )
            status = VLStatusCode.VL_IMAGE_FORMAT_ERROR
            vit_res.status = status
            return vit_res
        preprocess_end = time.time() if self.profile else None
        # forward
        output = self.get_vit_result(image, image_info)
        forward_end = time.time() if self.profile else None
        if audio_output_lengths is not None:
            vit_res.set_embs(output.detach().squeeze(0).cpu()[:audio_output_lengths])
        else:
            vit_res.set_embs(output.detach().squeeze(0).cpu())
        vit_res.status = status
        if self.profile:
            vit_res.vit_preprocess_time = preprocess_end - preprocess_start
            vit_res.vit_forward_time = forward_end - preprocess_end
            print(
                f"request_id: {vit_res.request_id} vit preprocess time: {vit_res.vit_preprocess_time}, vit forward time: {vit_res.vit_forward_time}"
            )
        return vit_res


class HieWorker:
    def __init__(self, vit_config: VitConfig, trt_vit_config=None) -> None:
        # mp.set_start_method('spawn')
        # manager = multiprocessing.Manager()
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processes = []
        self.exit_signal = threading.Event()
        logging.info("init hie process, workers: {}".format(vit_config.workers))
        for i in range(vit_config.workers):
            device = torch.device(f"cuda:{i}")
            p = HieWokerImpl(
                self.request_queue,
                self.result_queue,
                model_path=vit_config.model_path,
                exit_signal=self.exit_signal,
                device=device,
                precision=vit_config.precision,
                profile=vit_config.profile,
                backend=vit_config.backend,
                trt_vit_config=trt_vit_config,
            )
            p.start()
            self.processes.append(p)
        self.wait_init_done(vit_config.workers)
        self.thread = threading.Thread(target=self.loop_get_result)
        # client_request_queues for clinet to wait results
        self.client_request_queues = {}
        self.thread.start()

    def loop_get_result(self):
        while True:
            # get result from result queue put from multi-process
            vit_res = self.result_queue.get()
            # logging.info(f"loop get vit res request id:{vit_res.request_id} id:{vit_res.vit_id} time: {time.time()}")
            if vit_res.is_terminated():
                break
            # put result to client queue
            if vit_res.request_id in self.client_request_queues:
                self.client_request_queues[vit_res.request_id].put(vit_res)
            else:
                logging.error(
                    f"request_id:{vit_res.request_id} not in client_request_queues"
                )

    def add_request(self, request, **kwargs):
        images_or_audios = kwargs.get("images_or_audios", [])
        grid_thw = kwargs.get("grid_thw", [])
        audio_attn_mask = kwargs.get("audio_attn_mask", [])
        for i, input in enumerate(images_or_audios):
            # multi-process to forward
            if input is None:
                continue
            request_vit = VitRequest(
                request_id=request.request_id,
                vit_id=i,
                vit_input=input,
                vit_grid_thw=(
                    grid_thw[i]
                    if (
                        grid_thw is not None
                        and len(grid_thw) > i
                        and grid_thw[i] is not None
                    )
                    else None
                ),
                audio_attn_mask=(
                    audio_attn_mask[i]
                    if (
                        audio_attn_mask is not None
                        and len(audio_attn_mask) > i
                        and audio_attn_mask[i] is not None
                    )
                    else None
                ),
            )
            # request_vit = Vit(request_id=request.request_id, vit_id=i, image=image, embs=torch.randn(1, 792, 4096).to(torch.float16))
            # self.client_request_queues[request_vit.request_id].put(request_vit)
            self.request_queue.put(request_vit)
            # print(f"len request_queue:{self.request_queue.qsize()}")

    def get_result(self, num, request_id, hie_results):
        for i in range(num):
            vit_res = None
            try:
                vit_res = self.client_request_queues[request_id].get(timeout=40)
            except queue.Empty:
                self.client_request_queues.pop(request_id)
                logging.error(
                    f"Queue is empty, timeout occurred request_id:{request_id}"
                )
                raise ValueError(f"get vit result timeout request_id:{request_id}")

            # logging.info(f"worker get vit res request id:{vit_res.request_id} id:{vit_res.vit_id} time: {time.time()}")
            hie_results.add_vit(vit_res)
            if vit_res.request_id != request_id:
                raise ValueError(
                    f"request_id not match {request_id} != {vit_res.request_id} "
                )

    def wait_init_done(self, num):
        for i in range(num):
            self.result_queue.get()
        pass

    def eval(self, request, hie_results, **kwargs) -> int:
        # add request to queue(thread) to get result from multi-process
        self.client_request_queues[request.request_id] = queue.Queue()
        images_or_audios = kwargs.get("images_or_audios", [])
        self.add_request(request, **kwargs)
        valid_images_or_audios = sum(
            1 for image in images_or_audios if image is not None
        )
        self.get_result(valid_images_or_audios, request.request_id, hie_results)
        self.client_request_queues.pop(request.request_id)
        return valid_images_or_audios

    def terminate(self):
        term_vit = Vit(status=VitStatus.TEMINATE_WAITING)
        self.result_queue.put(term_vit)
        self.thread.join()
        self.exit_signal.set()
        for _ in self.processes:
            term_vit = VitRequest(vit_id=-1)
            self.request_queue.put(term_vit)
        for p in self.processes:
            p.join()
