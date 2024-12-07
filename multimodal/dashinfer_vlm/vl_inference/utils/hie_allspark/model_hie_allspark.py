'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model_hie_allspark.py
'''
import logging
import torch
from ..vl_logger import (
    VlSlsStep,
    logger_info,
    logger_error,
)

from dashinfer.allspark import AsStatus
from dashinfer import allspark
from .model import Model
from typing import Optional, List
from dataclasses import dataclass
from ..qwen_vl_status import VLStatusCode, Interval
import time

# import shortuuid


@dataclass
class AllSparkRequest:
    request_id: Optional[int] = -1
    torch_input: Optional[dict] = None
    input_lists: Optional[List[List[int]]] = None
    gen_cfg: Optional[dict] = None
    vit_embs: Optional[List[torch.Tensor]] = None
    vit_positions: Optional[torch.Tensor] = None
    vit_keys: Optional[List[torch.Tensor]] = None
    vit_target_token: Optional[int] = None
    old_context_len: Optional[int] = 0
    # new total prompt token num
    new_context_len: Optional[int] = 0
    # new image/audio token num
    new_image_audio_token_len: Optional[int] = 0
    # new image/audio token num
    new_video_token_len: Optional[int] = 0
    # max total tokens including prompt and expected output length
    max_total_tokens: Optional[int] = 0


class AllSparkM6Model(Model):
    def __init__(self, as_model_config: allspark.AsModelConfig):
        self.model_name = as_model_config.model_name
        logging.info("ModelName: {}".format(as_model_config.model_name))
        logging.info("model_path: {}".format(as_model_config.model_path))
        logging.info("weights_path: {}".format(as_model_config.weights_path))
        if (
            as_model_config.model_name is None
            or as_model_config.model_path is None
            or as_model_config.weights_path is None
        ):
            raise Exception(
                "Please specify correct model_name, model_path, weights_path"
            )
        self.engine = allspark.Engine()
        self.engine.build_model_from_config_struct(as_model_config)
        # start as model
        status = self.engine.start_model(self.model_name)
        assert status == AsStatus.ALLSPARK_SUCCESS

    def __del__(self):
        if self.engine is not None and self.model_name is not None:
            self.engine.stop_model(self.model_name)

    def forward(self, request: AllSparkRequest):
        prompt_token_num_list = [
            request.new_context_len,
            request.new_image_audio_token_len,
            request.new_video_token_len,
        ]
        request_id = request.request_id
        status, request_handle, result_queue = self.engine.start_request(
            self.model_name,
            {
                "input_ids": torch.utils.dlpack.to_dlpack(
                    request.torch_input["input_ids"]
                ),
                "attention_mask": torch.utils.dlpack.to_dlpack(
                    request.torch_input["attention_mask"]
                ),
            },
            generate_config=request.gen_cfg,
        )
        if status != AsStatus.ALLSPARK_SUCCESS:
            logger_error(
                step=VlSlsStep.vl_as_error,
                request_id=request_id,
                context={"status": int(status), "msg": "start_request error"},
            )
            yield VLStatusCode.VL_OTHER_ERROR, request_id, status, [], prompt_token_num_list
            return
        # request.handle = request_handle
        # request.queue = result_queue
        # request.status = int(status)
        status = allspark.GenerateRequestStatus.Init
        as_start_time = time.time()
        output_ids = []

        # print(self.engine.get_op_profiling_info(self.model_name))
        def cleanup(request):
            # print(self.engine.get_op_profiling_info(self.model_name))
            self.engine.release_request(self.model_name, request_handle=request_handle)
            logging.info(f"cleanup request id:{request.request_id}")
            # sls log
            logger_info(
                step=VlSlsStep.vl_as_stop,
                request_id=request.request_id,
                context={
                    "status": int(status),
                    "total_tokens": len(output_ids),
                },
                interval=Interval(
                    type="as_time", cost=int((time.time() - as_start_time) * 1000)
                ),
            )
            # should release embedding info
            # request.gen_cfg["extra_embedding"] = None
            # request.gen_cfg = {}
            # request.torch_input = None
            # request.vit_embs = None
            # request = None

        while True:
            new_ids = []
            generated_elem = result_queue.Get()
            if generated_elem is not None:
                new_ids = generated_elem.ids_from_generate
            status = result_queue.GenerateStatus()
            if status == allspark.GenerateRequestStatus.Generating:
                if len(output_ids) == 0:
                    logger_info(
                        step=VlSlsStep.vl_as_first_token,
                        request_id=request_id,
                        context={
                            "input_len": request.torch_input["input_ids"].shape[1]
                        },
                        interval=Interval(
                            type="first_token_time",
                            cost=int((time.time() - as_start_time) * 1000),
                        ),
                    )
                output_ids.extend(new_ids)
                yield VLStatusCode.VL_SUCCESS, request_id, status, new_ids, prompt_token_num_list
            elif status == allspark.GenerateRequestStatus.GenerateFinished:
                output_ids.extend(new_ids)
                cleanup(request)
                yield VLStatusCode.VL_SUCCESS, request_id, status, new_ids, prompt_token_num_list
                break
            elif status == allspark.GenerateRequestStatus.GenerateInterrupted:
                logging.warn("[Warn]Request interrupted!")
                cleanup(request)
                yield VLStatusCode.VL_SUCCESS, request_id, status, new_ids, prompt_token_num_list
                break
            else:
                logging.error("[Error] Unexpected status: {}".format(status))
                cleanup(request)
                yield VLStatusCode.VL_SUCCESS, request_id, status, new_ids, prompt_token_num_list
                break

    def terminate(self):
        pass
