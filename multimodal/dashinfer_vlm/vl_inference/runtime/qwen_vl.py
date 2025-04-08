'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    qwen_vl.py
'''
from .hie_worker import HieWorker
from ..utils.config import QWEN_MODEL_TYPES
from ..utils.config import SPECIAL_TOKENS_DICT
from .hie_allspark_worker import HieAllsparkWorker
from ..utils.hie_allspark.model_hie_allspark import (
    AllSparkRequest,
)
from ..utils.qwen_vl_status import VLStatusCode, Interval
from ..utils.cache.cache_manager import CacheManager
from ..utils.vl_logger import (
    VlSlsStep,
    logger_info,
    logger_error,
)
from ..utils.config import VitConfig, CacheConfig
from ..utils.env import getenv
from ..utils.hie.vit_preprocess import Preprocessor
from ..utils.qwen_vl_truncate import qwen_vl_truncate
from ..utils import error as error
from .vit import Vit, VitGroup
import threading
from dashinfer import allspark

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
import torch
import numpy as np
import logging
import time
import copy

ImageInfoKeys = Literal["images", "urls", "grid_thw"]
AudioInfoKeys = Literal["audios", "urls", "audio_attention_mask"]


@dataclass
class VLPreprocessResp:
    request_id: Optional[int] = -1
    # llm length for images/videos/audios {'url':len}
    data: Optional[Dict] = None
    # List to store vision/audio lengths in request order
    vision_lens: Optional[List[int]] = None


@dataclass
class VLRequest:
    request_id: Optional[int] = -1
    tokenizer = None
    chat_format: str = "CHATML"
    model_type: str = "QWEN2-VL"
    preprocessor: Preprocessor = None
    min_pixels: int = 0
    max_pixels: int = 0
    fps: Optional[Any] = None
    context_length: int = 0
    context_tokens: List[int] = None
    output_ids: List[int] = field(default_factory=list)

    input_tokens: Optional[List[List[int]]] = None
    truncate_lengths: Optional[List[int]] = None
    # max input length
    max_input_length: Optional[int] = -1
    # # List to store images which supports str(url/path)/np.ndarray/torch.Tensor
    # images: Optional[List[Any]] = None
    # # List to store corresponding URLs or paths
    # images_url: Optional[List[Any]] = None
    # # List to store preprocessed image info
    # images_info: Optional[dict] = None
    # dict to store image infos
    # {"images": Optional[List[Any]] = None, "urls": Optional[List[Any]], "grid_thw": Optional[List[List[int]]] = None}
    # "images": List to store images which supports str(url/path)/np.ndarray/torch.Tensor
    # "urls": List to store corresponding URLs or paths
    # "grid_thw": List to store corresponding grid_thw
    # "config": List to store corresponding config
    # "config": [{'type':"image/video": 'url':"url/filepath", "min_pixels": 0, "max_pixels": 0, "fps": 1}],
    images_info: Optional[Dict[ImageInfoKeys, List[Optional[Any]]]] = None

    # dict to store audio infos
    # {"audios": Optional[List[Any]] = None, "urls": Optional[List[Any]], "audio_attention_mask": Optional[List[float16]] = None}
    # "audios": List to store audio which supports str(url/path)/np.ndarray/torch.Tensor
    # "urls": List to store corresponding URLs or paths
    # "audio_attention_mask": List to store corresponding audio_attention_mask
    audios_info: Optional[Dict[AudioInfoKeys, List[Optional[Any]]]] = None
    # [{'type': "image/video/audio", 'url' = 'xxxxx', 'min_pixels': 0, 'max_pixels': 0, 'fps': 1}]
    preprocess_req: Optional[List[dict]] = None
    # List to store md5 hash of images
    keys: Optional[List[Any]] = None
    # List to store the multimodal type
    # 0: image, 1: video, 2: audio
    mm_types: Optional[List[int]] = None
    gen_cfg: Optional[dict] = None
    # for benchmark output
    vit_preprocess_time: Optional[float] = 0
    vit_forward_time: Optional[float] = 0
    vit_cache_time: Optional[float] = 0
    vit_time: Optional[float] = 0
    vit_len: Optional[int] = 0
    as_context_time: Optional[float] = 0
    as_context_len: Optional[int] = 0

    # def __init__(self, id, tokenizer, chat_format, model_type, preprocessor=None, min_pixels=None, max_pixels=None, fps=None):
    #     self.request_id = id
    #     self.tokenizer = tokenizer
    #     self.chat_format = chat_format
    #     self.model_type = model_type

    #     self.preprocessor = preprocessor
    #     self.min_pixels = min_pixels
    #     self.max_pixels = max_pixels
    #     self.fps = fps

    def set_input_tokens_bak(self, text, vision_num=1):
        from ..utils.chat_format_utils import (
            encode_string_to_tokens,
        )

        raw_text, self.context_tokens, raw_text_len, self.context_length = (
            encode_string_to_tokens(text, self.tokenizer, self.chat_format)
        )
        from ..utils.config import SPECIAL_TOKENS_DICT

        self.context_tokens[14 : 2 * vision_num] = [
            SPECIAL_TOKENS_DICT.get("IMAGE_BOS_V2"),
            SPECIAL_TOKENS_DICT.get("IMAGE_EOS_V2"),
        ] * vision_num
        # # add special token to the prompt if url is provided
        # if self.model_type == "QWEN2-AL":
        #     self.context_tokens[14:2] = [151647, 151648]
        # elif self.model_type == "QWEN2-VL":
        #     from ..utils.config import SPECIAL_TOKENS_DICT
        #     self.context_tokens[14:2 * vision_num] = [SPECIAL_TOKENS_DICT.get('IMAGE_BOS_V2'), SPECIAL_TOKENS_DICT.get('IMAGE_EOS_V2')] * vision_num
        # else:
        #     self.context_tokens[14:2] = [151857, 151857]
        self.input_tokens = [self.context_tokens]

    def set_input_tokens(self, prompt, tokenizer):
        input_ids = tokenizer(prompt).input_ids
        # print(input_ids)
        self.input_tokens = [input_ids]

    def get_images(
        self, vision_format, images, preprocessor, min_pixels, max_pixels, fps=None
    ):
        select_image = [x for x in images]
        select_image_path = []
        extra_info = []
        if vision_format == "image":
            for i, image in enumerate(images):
                status, pre_image, vit_len = preprocessor.get_vision_info(
                    url=image,
                    type=vision_format,
                    fps=fps,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                # select_image.append(image)
                extra_info.append(pre_image["grid_thw"])
                pre_image = pre_image["image"]
                # if self.model_type == 'QWEN1-VL':
                #     status, pre_image, vit_len = preprocessor.get_patch_image(image)
                #     pre_image = pre_image['image']
                # elif self.model_type == 'QWEN2-VL':
                #     vision_format = 'image' #TODO: video?
                #     status, pre_image, vit_len = preprocessor.get_vision_info(url=image, type=vision_format, fps=fps, min_pixels=min_pixels, max_pixels=max_pixels)
                #     extra_info.append(pre_image["grid_thw"])
                #     pre_image = pre_image['image']
                # else:
                #     status, pre_image, vit_len = preprocessor.get_audio_info(image)
                #     extra_info.append(pre_image['mask'])
                #     pre_image = pre_image['feature']
                #     print(f"pre_image shape:{pre_image.shape}, atten_mask shape:{extra_info[i].shape}")
                if status == VLStatusCode.VL_SUCCESS:
                    select_image_path.append(image)
                    select_image[i] = pre_image
        elif vision_format == "video":
            status, pre_image, vit_len = preprocessor.get_vision_info(
                url=tuple(images),
                type=vision_format,
                fps=fps,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            extra_info.append(pre_image["grid_thw"])
            pre_image = pre_image["image"]
            if status == VLStatusCode.VL_SUCCESS:
                # select_image_path = [tuple(images)]
                select_image_path = ["".join(images)]
                select_image = [pre_image]
        return select_image, select_image_path, extra_info

    def set_image_info(self, vision_format, image_urls):
        images, images_url, extra_info = self.get_images(
            vision_format,
            image_urls,
            self.preprocessor,
            self.min_pixels,
            self.max_pixels,
            self.fps,
        )
        self.images_info = {
            "images": images,
            "urls": images_url,
            "grid_thw": extra_info,
        }


class QwenVl:
    def __init__(
        self,
        as_config: allspark.AsModelConfig,
        vit_config: Optional[VitConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        trt_vit_config=None,
    ) -> None:
        import os

        # 打印所有环境变量
        for key, value in os.environ.items():
            print(f"{key}: {value}")
        if not vit_config or vit_config.check_valid_config() is False:
            # only run llm
            self.hie_worker = None
            pass
        else:
            self.hie_worker = HieWorker(vit_config, trt_vit_config)
        # init allspark worker
        self.as_worker = HieAllsparkWorker(as_config)
        self.profile = (
            True if vit_config is not None and vit_config.profile is True else False
        )
        # init cache manager
        self.cache_manager = CacheManager(cache_config)
        # init time list
        self.hie_vit_times = []
        self.cache_vit_times = []
        self.hie_model_type = getenv("QWEN_MODEL_TYPE", "QWEN2-VL").upper()
        if self.hie_model_type not in QWEN_MODEL_TYPES:
            raise ValueError(f"QWEN_MODEL_TYPE shoule be set one of {QWEN_MODEL_TYPES}")
        # init preprocess
        self.preprocess = Preprocessor(
            workers=vit_config.workers,
            model_type=self.hie_model_type,
            dtype=torch.float16,
        )
        self.retry_start_time = 0
        self.cache_valid = True
        self.lock = threading.Lock()
        self.req_model_map = {
            "QWEN1-VL": "images_info",
            "QWEN2-VL": "images_info",
            "QWEN2-AL": "audios_info",
            "GUMMY-AL": "audios_info",
        }
        self.req_info_map = {
            "QWEN1-VL": "images",
            "QWEN2-VL": "images",
            "QWEN2-AL": "audios",
            "GUMMY-AL": "audios",
        }
        self.req_extra_map = {
            "QWEN1-VL": None,
            "QWEN2-VL": "grid_thw",
            "QWEN2-AL": "audio_attention_mask",
            "GUMMY-AL": "audio_attention_mask",
        }
        self.max_input_len = int(getenv("DS_LLM_MAX_IN_TOKENS", "20000"))
        self.max_total_len = int(getenv("DS_LLM_MAX_TOKENS", "32000"))

    """
    支持input_tokens格式:
    [1,2,3,4,5,正常id..., 151857,151858, 1,2,3,4,5,正常id...], 151857,151858 表示151859插入的位置
    [1,2,3,4,5,正常id..., 151857,151859, 151859, 151859, 多个重复151859..., 151858, 1,2,3,4,5,正常id...]
    注意: 如果输入tokens是多个连续的151859这种格式的,151859长度要和vit_seq_len一致, 这个要输入保证, 这里不再判断
    """

    def get_as_request(self, vl_request: VLRequest, hie_results) -> AllSparkRequest:
        """根据vit结果构建llm请求的input_ids和其他相关参数

        Args:
            vl_request (VLRequest): vl request
            hie_results (_type_): hie vit 结果

        Raises:
            ValueError: _description_
            ValueError: _description_
            AssertionError: _description_
            AssertionError: _description_
            AssertionError: _description_

        Returns:
            AllSparkRequest: _description_
        """
        as_request = AllSparkRequest()
        as_request.request_id = vl_request.request_id
        as_request.gen_cfg = copy.deepcopy(vl_request.gen_cfg)
        as_request.vit_embs = []
        target_id = 151859
        replace_bos_id = 151857
        replace_eos_id = 151858
        if self.hie_model_type == "QWEN1-VL":
            token_bos = SPECIAL_TOKENS_DICT.get("IMAGE_BOS")
            token_eos = SPECIAL_TOKENS_DICT.get("IMAGE_EOS")
            target_id = SPECIAL_TOKENS_DICT.get("IMAGE_TARGET")
        elif self.hie_model_type == "QWEN2-VL":
            token_bos = SPECIAL_TOKENS_DICT.get("IMAGE_BOS_V2")
            token_eos = SPECIAL_TOKENS_DICT.get("IMAGE_EOS_V2")
            target_id = SPECIAL_TOKENS_DICT.get("IMAGE_TARGET_V2")
        else:
            token_bos = SPECIAL_TOKENS_DICT.get("AUDIO_BOS")
            token_eos = SPECIAL_TOKENS_DICT.get("AUDIO_EOS")
            target_id = SPECIAL_TOKENS_DICT.get("AUDIO_TARGET")
        if vl_request.input_tokens is None or len(vl_request.input_tokens) == 0:
            raise error.InputNullTokensError(
                "input null token... may be upload service for vit cache"
            )
        if (
            isinstance(vl_request.input_tokens, list)
            and len(vl_request.input_tokens) == 1
        ):
            pass
        else:
            raise ValueError("input_tokens should be list[list[int]]")
        input_tokens = copy.deepcopy(vl_request.input_tokens[0])
        # covert to numpy
        if isinstance(vl_request.input_tokens[0], torch.Tensor):
            input_tokens = vl_request.input_tokens[0].numpy()
        elif isinstance(vl_request.input_tokens[0], list):
            input_tokens = np.array(vl_request.input_tokens[0])
        elif isinstance(vl_request.input_tokens[0], np.ndarray):
            pass
        else:
            raise ValueError(
                "Unsupported input type: {}".format(type(vl_request.input_tokens[0]))
            )
        as_request.old_context_len = input_tokens.shape[0]
        if self.hie_model_type == "QWEN2-VL":
            input_tokens[input_tokens == replace_bos_id] = token_bos
            input_tokens[input_tokens == replace_eos_id] = token_eos
        if not np.isin(target_id, input_tokens):
            count = np.sum(
                (input_tokens[:-1] == token_bos) & (input_tokens[1:] == token_eos)
            )
            positions = np.where(
                (input_tokens[:-1] == token_bos) & (input_tokens[1:] == token_eos)
            )[0]
            if count != len(hie_results):
                raise AssertionError(
                    f"vit result nums:{len(hie_results)} should be equal to request count: {count}"
                )
            shift = 0  # 插入的位移
            for i, pos in enumerate(positions):
                vit_len = hie_results[
                    i
                ].get_vit_len()  # 假设这里的 pos 是依据 hie_results 索引
                input_tokens = np.insert(
                    input_tokens, pos + 1 + shift, [target_id] * vit_len
                )
                as_request.vit_embs.append(
                    hie_results[i].get_embs()
                )  # 将 emb 插入 as_request 的 vit_embs
                shift += vit_len  # 更新位移
        else:
            # 找出连续的151859出现的次数
            def count_continuous_occurrences(input_array, target):
                counts = []
                current_count = 0

                for num in input_array:
                    if num == target:
                        current_count += 1
                    else:
                        if current_count > 0:
                            counts.append(current_count)
                            current_count = 0

                # Check if the last series of 151859 was counted
                if current_count > 0:
                    counts.append(current_count)

                return counts

            target_list = count_continuous_occurrences(input_tokens, target_id)
            if len(target_list) != len(hie_results):
                raise AssertionError(
                    f"vit result nums:{len(hie_results)} should be equal to request count: {len(target_list)}"
                )
            for i in range(len(hie_results)):
                if hie_results[i].get_vit_len() != target_list[i]:
                    raise AssertionError(
                        f"vit embedding len:{hie_results[i].get_vit_len()} should be equal to input target length: {target_list[i]}"
                    )
                as_request.vit_embs.append(hie_results[i].get_embs())
        if self.hie_model_type == "GUMMY-AL":
            # remove token_bos/token_eos, GUMMY-AL need to remove token_bos/token_eos
            values_to_remove = [token_bos, token_eos]
            input_tokens = input_tokens[~np.isin(input_tokens, values_to_remove)]
        inputs = input_tokens.reshape(1, -1)
        vl_request.as_context_len = inputs.shape[1]
        attention_mask = np.ones(inputs.shape, dtype=np.int64)
        as_request.torch_input = {
            "input_ids": torch.Tensor(inputs).to(torch.int64),
            "attention_mask": torch.Tensor(attention_mask).to(torch.int64),
        }
        # torch.save(torch.Tensor(inputs).to(torch.int64), 'tensor_input.pt')
        as_request.vit_keys = [key for key in (vl_request.keys or [])]
        if len(as_request.vit_keys) != len(hie_results):
            raise ValueError(
                f"vit_keys length should be equal to vit_results length: {len(as_request.vit_keys)} != {len(hie_results)}"
            )
        for i in range(len(as_request.vit_keys)):
            vit_len_hex = format(hie_results[i].get_vit_len(), "x")
            if len(vit_len_hex) % 2 != 0:
                vit_len_hex = "0" + vit_len_hex
            byte_array = bytearray.fromhex(as_request.vit_keys[i] + vit_len_hex)
            byte_tensor = torch.tensor(byte_array, dtype=torch.uint8)
            as_request.vit_keys[i] = byte_tensor
        #     print(f"byte_array:{byte_array} byte_tensor:f{byte_tensor} shape:{byte_tensor.shape}")
        #     torch.save(byte_tensor, f'tensor_input_key_{i}.pt')

        # for qwen-vl2 get position
        as_request.input_lists = [input_tokens.tolist()]
        # qwen-vl2 get position
        as_request.vit_positions = [
            [result.get_vit_grid_thw() for result in hie_results]
        ]
        as_request.vit_target_token = target_id
        as_request.new_context_len = vl_request.as_context_len
        as_request.max_total_tokens = self.max_total_len
        if isinstance(vl_request.mm_types, list):
            if len(vl_request.mm_types) != len(hie_results):
                raise ValueError(
                    f"mm_types length should be equal to vit_results length: {len(vl_request.mm_types)} != {len(hie_results)}"
                )
            as_request.new_image_audio_token_len = 0
            as_request.new_video_token_len = 0
            for i in range(len(vl_request.mm_types)):
                if vl_request.mm_types[i] == 0 or vl_request.mm_types[i] == 2:
                    as_request.new_image_audio_token_len += hie_results[i].get_vit_len()
                else:
                    as_request.new_video_token_len += hie_results[i].get_vit_len()
        else:
            as_request.new_image_audio_token_len = sum(
                result.get_vit_len() for result in hie_results
            )
        # logging.error(f"new_context_len:{as_request.new_context_len} max_total_tokens:{as_request.max_total_tokens}, new_image_audio_token_len:{as_request.new_image_audio_token_len}, new_video_token_len:{as_request.new_video_token_len}")
        return as_request

    def get_cache_vit_result_from_request(self, vl_request, hie_results) -> None:
        """从chat预处理图像获取缓存结果, 如果有cache命中还需要更新 images or audios, 置为none, 表示不用做后面的vit了

        Args:
            vl_request (_type_): vl_request
            hie_results (_type_): 存储缓存结果
        """
        cache_start = time.time()
        vl_request_attr_dict = vl_request.__dict__
        info_dict = vl_request_attr_dict.get(self.req_model_map[self.hie_model_type])
        urls = info_dict.get("urls")
        extra_key = self.req_extra_map[self.hie_model_type]
        extras = info_dict.get(extra_key)
        if self.hie_model_type == "QWEN2-VL" or self.hie_model_type == "QWEN1-VL":
            extras_grid = extras
        else:
            extras_grid = None
        if extras_grid is None:
            extras_grid = [None] * len(urls)
        images_or_audios = info_dict.get(self.req_info_map[self.hie_model_type])
        cache_vits = self.cache_manager.get_vits_from_cache(
            urls=urls, grid_thws=extras_grid
        )
        # cache sls log
        time_cost = int((time.time() - cache_start) * 1000)
        # update keys for insert
        vl_request.keys = CacheManager.get_md5_keys(urls)
        if self.hie_model_type == "QWEN2-VL" or self.hie_model_type == "QWEN1-VL":
            vl_request.mm_types = [0 for _ in urls]
        else:
            vl_request.mm_types = [2 for _ in urls]
        logger_info(
            step=VlSlsStep.vl_cache_end,
            request_id=vl_request.request_id,
            context={
                "keys": [result["key"] for result in cache_vits],
                "urls": [urls[i] for i, _ in enumerate(cache_vits)],
                "time": [result["cache_time"] for result in cache_vits],
                "cache_num": sum(
                    1 for result in cache_vits if result["result"] is not None
                ),
                "status": [result["status"] for result in cache_vits],
                "total_num": len(cache_vits),
                "source": [result["source"] for result in cache_vits],
                "local": [result["local"] for result in cache_vits],
            },
            interval=Interval(type="cache_time", cost=time_cost),
        )
        # add cache to hie_result
        # import torch
        # cache_tensor = torch.load('data.pt')
        # cache_tensor = torch.squeeze(cache_tensor).float().cpu()
        # print(f"cache_tensor:{cache_tensor}, shape:{cache_tensor.shape}")
        # cache_vits.append({"key":"md5_hash", "result": cache_tensor, "cache_time": 0})
        for i, result in enumerate(cache_vits):
            if result["result"] is None:
                continue
            vit_res = Vit(
                vit_id=i,
                request_id=vl_request.request_id,
                embs=result["result"],
                vit_grid_thw=result["grid_thw"],
                vit_cache_time=result["cache_time"],
                status=VLStatusCode.VL_SUCCESS,
                from_cache=True,
            )
            hie_results.add_vit(vit_res)
            images_or_audios[i] = None
        # update images or audios
        info_dict[self.req_info_map[self.hie_model_type]] = images_or_audios

    def insert_vit_cache(self, vl_request: VLRequest, hie_results: VitGroup):
        vl_request_attr_dict = vl_request.__dict__
        info_dict = vl_request_attr_dict.get(self.req_model_map[self.hie_model_type])
        urls = info_dict.get("urls")
        for i, result in enumerate(hie_results):
            if (
                hie_results[i].is_from_cache() is False
                and hie_results[i].status == VLStatusCode.VL_SUCCESS
            ):
                if isinstance(urls[i], str):
                    hash_key = CacheManager.get_md5_key(urls[i])
                    res = self.cache_manager.insert(
                        hash_key,
                        hie_results[i].get_embs(),
                        grid_thw=hie_results[i].get_vit_grid_thw(),
                        sync=False,
                    )
                    if res is False:
                        logger_info(
                            step=VlSlsStep.vl_cache_error,
                            request_id=vl_request.request_id,
                            context={"url": urls[i], "key": hash_key},
                        )

    def check_valid_config_input(self, vl_request: VLRequest):
        """
        需要一体化做解码情况下,检查输入配置

        Args:
            vl_request (VLRequest):

        Raises:
            ValueError: 各种参数校验失败
        """
        if vl_request.preprocess_req is None:
            raise ValueError(
                f"preprocess_req is None, request_id:{vl_request.request_id}"
            )

        if vl_request.max_input_length is None or vl_request.max_input_length == -1:
            vl_request.max_input_length = self.max_input_len
            logging.info(
                f"max_input_length is None, use default config from 'DS_LLM_MAX_IN_TOKENS' self.max_input_len:{self.max_input_len} request_id:{vl_request.request_id}"
            )
        if vl_request.max_input_length >= self.max_total_len:
            raise error.InputTokensError(
                f"max_input_length:{vl_request.max_input_length} should be less than max_total_len:{self.max_total_len}, request_id:{vl_request.request_id}"
            )
        for config in vl_request.preprocess_req:
            type = config.get("type")
            url = config.get("url")
            min_pixels = config.get("min_pixels")
            max_pixels = config.get("max_pixels")
            fps = config.get("fps")
            if type is None or type not in ["image", "video", "audio"]:
                raise ValueError(
                    f"preprocess_req type should be image/video/audio, type:{type}, request_id:{vl_request.request_id}"
                )
            if url is None:
                raise ValueError(
                    f"preprocess_req url should be set, request_id:{vl_request.request_id}"
                )
            if type == "image":
                if self.hie_model_type == "QWEN2-VL" and (
                    min_pixels is None or max_pixels is None
                ):
                    raise ValueError(
                        f"mix_pixels or max_pixels should be set in QWEN2-VL, request_id:{vl_request.request_id}"
                    )
            elif type == "video":
                if self.hie_model_type == "QWEN2-VL":
                    if min_pixels is None or max_pixels is None:
                        raise ValueError(
                            f"mix_pixels or max_pixels should be set in QWEN2-VL, request_id:{vl_request.request_id}"
                        )
                    if isinstance(url, str) and fps is None:
                        raise ValueError(
                            f"video is str, fps should be set, request_id:{vl_request.request_id}"
                        )
                else:
                    raise ValueError(
                        f"video only support in QWEN2-VL, please set env variable export QWEN_MODEL_TYPE=QWEN2-VL, request_id:{vl_request.request_id}"
                    )

    def check_valid_streaming_input(self, vl_request: VLRequest):
        """
        流式输入验证, 如果输入没带max_input_length,使用self.max_input_len
        输入'type'必须是'video', 'url'必须是流式图片list,且必须被self.preprocess.temporal_patch_size整除
        Args:
            vl_request (VLRequest): 请求参数
        Raises:
            ValueError: 输入检验失败
        """
        if vl_request.preprocess_req is None or len(vl_request.preprocess_req) == 0:
            raise ValueError(
                f"preprocess_req is None, request_id:{vl_request.request_id}"
            )
        if vl_request.max_input_length is None or vl_request.max_input_length == -1:
            vl_request.max_input_length = self.max_input_len
            logging.info(
                f"max_input_length is None, use default config from 'DS_LLM_MAX_IN_TOKENS' self.max_input_len:{self.max_input_len} request_id:{vl_request.request_id}"
            )
        for config in vl_request.preprocess_req:
            type = config.get("type")
            url = config.get("url")
            min_pixels = config.get("min_pixels")
            max_pixels = config.get("max_pixels")
            if type is None or type not in ["video"]:
                raise ValueError(
                    f"preprocess_req type should be only video, type:{type}, request_id:{vl_request.request_id}"
                )
            if url is None or not isinstance(url, list):
                raise ValueError(
                    f"preprocess_req url should be set to list, request_id:{vl_request.request_id}"
                )
            if len(url) % self.preprocess.temporal_patch_size != 0:
                raise ValueError(
                    f"preprocess_req url len:{len(url)} should be multiple of temporal_patch_size. request_id:{vl_request.request_id}"
                )
            if self.hie_model_type == "QWEN2-VL" and (
                min_pixels is None or max_pixels is None
            ):
                raise ValueError(
                    f"mix_pixels or max_pixels should be set in QWEN2-VL video, request_id:{vl_request.request_id}"
                )

    def check_valid_images_input(self, vl_request: VLRequest):
        """
        多模态图片/视频输入验证(解码预处理后), 参数校验
        输入images_info里的key必须有'images'表示预处理的图像或视频信息, 'urls'用于结果缓存的key信息
        Args:
            vl_request (VLRequest): 请求参数
        Raises:
            ValueError: 输入检验失败
        """
        if vl_request.images_info is None:
            raise ValueError(f"images_info is None, request_id:{vl_request.request_id}")
        input_keys = ["images", "urls"]
        # 检查每个必须的键是否都在字典中且值为长度大于1的列表
        for key in input_keys:
            if key not in vl_request.images_info:
                raise ValueError(f"Key '{key}' is missing from images_info.")
            if not isinstance(vl_request.images_info[key], list):
                raise ValueError(f"The value for key '{key}' is not a list.")
            if len(vl_request.images_info[key]) < 1:
                raise ValueError(
                    f"The list for key '{key}' does not have more than 1 element."
                )
        # 检查列表的长度是否相等
        length = len(vl_request.images_info[input_keys[0]])
        for key in input_keys[1:]:
            if len(vl_request.images_info[key]) != length:
                raise ValueError(
                    f"The list for key '{key}' does not have the same length as others."
                )
        # 如果version == 2 输入
        if self.hie_model_type == "QWEN2-VL":
            if not isinstance(vl_request.images_info["images"][0], str):
                # grid_thw is not None
                if (
                    "grid_thw" not in vl_request.images_info
                    or len(vl_request.images_info["grid_thw"]) != length
                ):
                    raise ValueError(
                        f"grid_thw is None, request_id:{vl_request.request_id}"
                    )

    def check_valid_audios_input(self, vl_request: VLRequest):
        """
        多模态audio输入验证(解码预处理后), 参数校验
        输入images_info里的key必须有'audios'表示预处理后的audio信息, 'urls'用于结果缓存的key信息
        Args:
            vl_request (VLRequest): 请求参数
        Raises:
            ValueError: 输入检验失败
        """
        if vl_request.audios_info is None:
            raise ValueError(f"audios_info is None, request_id:{vl_request.request_id}")
        input_keys = ["audios", "urls"]
        # check each required key is in dict and value is list and length > 1
        for key in input_keys:
            if key not in vl_request.audios_info:
                raise ValueError(f"Key '{key}' is missing from audios_info.")
            if not isinstance(vl_request.audios_info[key], list):
                raise ValueError(f"The value for key '{key}' is not a list.")
            if len(vl_request.audios_info[key]) < 1:
                raise ValueError(
                    f"The list for key '{key}' does not have more than 1 element."
                )
        # check list length
        length = len(vl_request.audios_info[input_keys[0]])
        for key in input_keys[1:]:
            if len(vl_request.audios_info[key]) != length:
                raise ValueError(
                    f"The list for key '{key}' does not have the same length as others."
                )
        # check audio_attention_mask
        if not isinstance(vl_request.audios_info["audios"][0], str):
            if (
                "audio_attention_mask" not in vl_request.audios_info
                or len(vl_request.audios_info["audio_attention_mask"]) != length
            ):
                raise ValueError(
                    f"audio_attention_mask is None, request_id:{vl_request.request_id}"
                )

    def hie_vit_forward(self, vl_request: VLRequest, hie_results: VitGroup):
        """
        使用hie vit入口函数, 一个请求可能包括多图/多视频/多语音, 会把请求发往多个vit线程进行推理,最后再合并到VitGroup

        Args:
            vl_request (VLRequest): 请求参数
            hie_results (VitGroup): vit结果
        """
        # all images are cached
        vl_request_attr_dict = vl_request.__dict__
        # cache sls log
        info_dict = vl_request_attr_dict.get(self.req_model_map[self.hie_model_type])
        if info_dict is None:
            return
        images_or_audios = info_dict.get(self.req_info_map[self.hie_model_type])
        extra_key = self.req_extra_map[self.hie_model_type]
        extras = info_dict.get(extra_key)
        if self.hie_model_type == "QWEN2-VL" or self.hie_model_type == "QWEN1-VL":
            extras_attn_mak = None
            extras_grid = extras
        else:
            extras_attn_mak = extras
            extras_grid = None
        if all(image is None for image in images_or_audios):
            return
        # vit forward
        vit_start = time.time()
        self.hie_worker.eval(
            vl_request,
            hie_results,
            images_or_audios=images_or_audios,
            audio_attn_mask=extras_attn_mak,
            grid_thw=extras_grid,
        )
        int((time.time() - vit_start) * 1000)
        # async insert vit cache
        self.insert_vit_cache(vl_request, hie_results)

    def build_new_request_from_cache(
        self, vl_pre_result: List[Dict], vl_request: VLRequest
    ) -> List[Dict]:
        """
        从前处理结果中构建新的请求
        函数主要实现了(1)多轮对话截断处理 (2)根据截断信息和本地缓存信息构建vl_request.images_info/vl_request.audios_info (3) 缓存得到的vit先插入到local memory
        Args:
            vl_pre_result (`List[Dict]`): 前处理结果
            vl_request (VLRequest): 请求参数

        Raises:
            error.InputTokensError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        # tokens truncate add here
        # [{'type': "image/video/audio", 'url' = 'xxxxx', 'min_pixels': 0, 'max_pixels': 0, 'fps': 1}]
        # input: vl_pre_resp, vl_request, modify VLRequest.input_tokens if any and return List of Dict like VLRequest.preprocess_req
        token_bos = SPECIAL_TOKENS_DICT.get("IMAGE_BOS")
        truncate_vision_lens = len(vl_pre_result)
        input_vision_lens = []
        cache_start = time.time()
        for i in range(len(vl_pre_result)):
            vit_len = 0
            if isinstance(vl_pre_result[i], tuple):
                _, _, vit_len = vl_pre_result[i]
            elif isinstance(vl_pre_result[i], list):
                for result in vl_pre_result[i]:
                    vit_len += result[2]
            input_vision_lens.append(vit_len)
        if self.hie_model_type == "QWEN1-VL":
            token_bos = SPECIAL_TOKENS_DICT.get("IMAGE_BOS")
        elif self.hie_model_type == "QWEN2-VL":
            token_bos = SPECIAL_TOKENS_DICT.get("IMAGE_BOS_V2")
        else:
            token_bos = SPECIAL_TOKENS_DICT.get("AUDIO_BOS")
        try:
            if vl_request.truncate_lengths is not None:
                logger_info(
                    step=VlSlsStep.vl_streaming_truncate_start,
                    request_id=vl_request.request_id,
                    context={
                        "truncate_lengths": vl_request.truncate_lengths,
                        "vision_lens": input_vision_lens,
                        "max_input_length": vl_request.max_input_length,
                    },
                )
                if vl_request.input_tokens is None or len(vl_request.input_tokens) == 0:
                    pass
                else:
                    if isinstance(vl_request.input_tokens[0], torch.Tensor):
                        input_tokens = vl_request.input_tokens[0].tolist()
                    elif isinstance(vl_request.input_tokens[0], list):
                        input_tokens = vl_request.input_tokens[0]
                    elif isinstance(vl_request.input_tokens[0], np.ndarray):
                        input_tokens = vl_request.input_tokens[0].tolist()
                    vl_request.input_tokens[0], truncate_vision_lens = qwen_vl_truncate(
                        input_tokens,
                        vl_request.truncate_lengths,
                        input_vision_lens,
                        max_input_tokens=vl_request.max_input_length,
                        bos_id=token_bos,
                    )
        except Exception as e:
            raise error.InputTokensError(str(e))
        if truncate_vision_lens > len(vl_pre_result):
            raise ValueError(
                f"build_new_request_from_cache error, request_id:{vl_request.request_id} truncate_vision_lens:{truncate_vision_lens} length should be less than request_len:{len(vl_pre_result)}"
            )
        # print(f"pre_data_list:{pre_data_list}")
        # logging.info(f"vl_pre_result:{vl_pre_result} truncate_vision_lens:{truncate_vision_lens}")
        images_pre = []
        images_url = []
        grid_thw = []
        audio_pre = []
        audio_url = []
        audio_attn_mask = []
        total_nums = 0
        cache_nums = 0

        def add_request_info(data):
            if data["type"] == "image" or data["type"] == "video":
                if data.get("image") is not None:
                    images_pre.append(data["image"])
                if data.get("grid_thw") is not None:
                    grid_thw.append(data["grid_thw"])
                if data.get("url") is not None:
                    images_url.append(data["url"])
            elif data["type"] == "audio":
                # {"audios": Optional[List[Any]] = None, "urls": Optional[List[Any]], "audio_attention_mask": Optional[List[float16]] = None}
                if data.get("feature") is not None:
                    audio_pre.append(data["feature"])
                if data.get("mask") is not None:
                    audio_attn_mask.append(data["mask"])
                if data.get("url") is not None:
                    audio_url.append(data["url"])

        if truncate_vision_lens > 0:
            vl_pre_result = vl_pre_result[-truncate_vision_lens:]
            for data in vl_pre_result:
                if isinstance(data, tuple):
                    total_nums += 1
                    vit_caches = self.cache_manager.get_vit_from_cache(
                        url=data[1].get("url", None),
                        grid_thw=data[1].get("grid_thw", None),
                        vit_len=data[2],
                    )
                    if vit_caches.get("result") is not None:
                        cache_nums += 1
                        if vit_caches.get("source", "local") == "global":
                            self.cache_manager.insert(
                                vit_caches.get("key"),
                                vit_caches.get("result"),
                                grid_thw=vit_caches.get("grid_thw", None),
                                use_redis=False,
                            )
                    else:
                        add_request_info(data[1])
                elif isinstance(data, list):
                    for result in data:
                        total_nums += 1
                        vit_caches = self.cache_manager.get_vit_from_cache(
                            url=result[1].get("url", None),
                            grid_thw=result[1].get("grid_thw", None),
                            vit_len=result[2],
                        )
                        if vit_caches.get("result") is not None:
                            cache_nums += 1
                            if vit_caches.get("source", "local") == "global":
                                self.cache_manager.insert(
                                    vit_caches.get("key"),
                                    vit_caches.get("result"),
                                    grid_thw=vit_caches.get("grid_thw", None),
                                    use_redis=False,
                                )
                        else:
                            add_request_info(result[1])
        if len(images_pre) > 0:
            vl_request.images_info = {
                "images": images_pre,
                "urls": images_url,
                "grid_thw": grid_thw,
            }
        if len(audio_pre) > 0:
            vl_request.audios_info = {
                "audios": audio_pre,
                "urls": audio_url,
                "audio_attention_mask": audio_attn_mask,
            }
        # update keys for prefix cache
        preprocess_req = vl_request.preprocess_req[-truncate_vision_lens:]
        vl_request.keys = []
        vl_request.mm_types = []
        for config in preprocess_req:
            if isinstance(config.get("url"), list):
                # merge url
                url = " ".join(config.get("url"))
                vl_request.keys.append(CacheManager.get_md5_key(url))
            else:
                vl_request.keys.append(CacheManager.get_md5_key(config.get("url")))
            type = config.get("type", "image")
            if type == "image":
                vl_request.mm_types.append(0)
            elif type == "video":
                vl_request.mm_types.append(1)
            elif type == "audio":
                vl_request.mm_types.append(2)
        time_cost = int((time.time() - cache_start) * 1000)
        logger_info(
            step=VlSlsStep.vl_cache_end,
            request_id=vl_request.request_id,
            context={
                "total_nums": total_nums,
                "cache_nums": cache_nums,
                "truncate_vision_lens": truncate_vision_lens,
            },
            interval=Interval(type="cache_time", cost=time_cost),
        )
        return vl_pre_result

    def get_image_audio_from_request(self, vl_request: VLRequest) -> List[Dict]:
        """
        根据用户输入的请求参数, 进行解码和缓存, 函数完成后可以执行build_new_request_from_cache构建请求参数进行vit

        Args:
            vl_request (VLRequest): 请求参数,主要是url和预处理参数

        Raises:
            error.VisionDecodeError: 解码异常

        Returns:
            VLPreprocessResp: url及对应的embedding seq length
        """
        # get preprocess from cache
        vl_pre_resp = VLPreprocessResp()
        vl_pre_resp.request_id = vl_request.request_id
        vl_pre_resp.data = {}
        # vl_pre_resp.data = self.cache_manager.get_preprocess_from_request(request_id=str(vl_request.request_id))[1]
        results = []
        request_images = []
        request_audios = []
        bad_request = False
        error_info = None
        # set request based on cache result
        # logging.info(f"get_image_audio_from_request request_id:{vl_request.request_id} vl_request.preprocess_req:{vl_request.preprocess_req}")
        for config in vl_request.preprocess_req:
            if config.get("type") == "image" or config.get("type") == "video":
                request_images.append(config)
            elif config.get("type") == "audio":
                request_audios.append(config)
        start_time = time.time()
        if len(request_images) > 0:
            # vl_request.images_info['images'].clear()
            results += self.preprocess.batch_get_vision_info(request_images)
            for i in range(len(results)):
                if isinstance(results[i], tuple):
                    status, info, vit_len = results[i]
                    if status != VLStatusCode.VL_SUCCESS:
                        bad_request = True
                        error_info = info["error"]
                    if vl_pre_resp.data is None:
                        vl_pre_resp.data = {}
                    vl_pre_resp.data.update(
                        {info.get("url"): {"vit_len": vit_len, "status": status}}
                    )
                elif isinstance(results[i], list):
                    status, info, vit_len = results[i][0]
                    for status, info, vit_len in results[i]:
                        if status != VLStatusCode.VL_SUCCESS:
                            bad_request = True
                            error_info = info["error"]
                        if vl_pre_resp.data is None:
                            vl_pre_resp.data = {}
                        vl_pre_resp.data.update(
                            {info.get("url"): {"vit_len": vit_len, "status": status}}
                        )

                # vl_request.images_info['images'].append(results[i][1].get('image') if results[i][1] is not None else None)
                # if self.hie_model_type == "QWEN2-VL":
                #     vl_request.images_info['grid_thw'].append(results[i][1].get('grid_thw') if results[i][1] is not None else None)
        if len(request_audios) > 0:
            # vl_request.images_info['audios'].clear()
            # vl_request.images_info['audio_attention_mask'].clear()
            results += self.preprocess.batch_get_audio_info(request_audios)
            for i in range(len(results)):
                status, info, vit_len = results[i]
                if status != VLStatusCode.VL_SUCCESS:
                    bad_request = True
                if vl_pre_resp.data is None:
                    vl_pre_resp.data = {}
                vl_pre_resp.data.update(
                    {info.get("url"): {"vit_len": vit_len, "status": status}}
                )
                # vl_request.audios_info['audios'].append(results[i][1].get('feature') if results[i][1] is not None else None)
                # vl_request.audios_info['audio_attention_mask'].append(results[i][1].get('mask') if results[i][1] is not None else None)
        # preprocess end log
        # logging.info(f"vl_pre_resp:{vl_pre_resp}")
        logger_info(
            step=VlSlsStep.vl_preprocess_end,
            request_id=vl_request.request_id,
            context={
                # "status": [result[0] for result in results],
                "urls": [url for url in vl_pre_resp.data.keys()],
                "data": [len for len in vl_pre_resp.data.values()],
            },
            interval=Interval(
                type="preprocess_time", cost=int((time.time() - start_time) * 1000)
            ),
        )
        if bad_request:
            raise error.VisionDecodeError(
                f"preprocess error:{error_info}, request_id:{vl_request.request_id}"
            )
        return results

    def merge_vit_results(
        self, vl_request: VLRequest, hie_results: VitGroup, vl_pre_result: List[Dict]
    ):
        if vl_pre_result is None:
            # 没有前处理结果直接返回vit结果
            return hie_results
        new_hie_results = VitGroup(request_id=vl_request.request_id)
        for i, data in enumerate(vl_pre_result):
            if isinstance(data, tuple):
                vit_caches = self.cache_manager.get_vit_from_cache(
                    url=data[1].get("url", None),
                    grid_thw=data[1].get("grid_thw", None),
                    vit_len=data[2],
                    use_redis=False,
                )
                if vit_caches.get("result") is not None:
                    hie_vit = Vit(
                        vit_id=i,
                        request_id=vl_request.request_id,
                        embs=vit_caches.get("result"),
                        vit_grid_thw=vit_caches.get("grid_thw", None),
                        status=VLStatusCode.VL_SUCCESS,
                    )
                    new_hie_results.add_vit(hie_vit)
            elif isinstance(data, list):
                emb_list = []
                grid_thw_list = []
                for result in data:
                    vit_caches = self.cache_manager.get_vit_from_cache(
                        url=result[1].get("url", None),
                        grid_thw=result[1].get("grid_thw", None),
                        vit_len=result[2],
                        use_redis=False,
                    )
                    if (
                        vit_caches.get("result") is not None
                        and vit_caches.get("grid_thw") is not None
                    ):
                        emb_list.append(vit_caches.get("result"))
                        grid_thw_list.append(vit_caches.get("grid_thw"))
                if len(emb_list) > 0:
                    merged_tensor = torch.cat(emb_list, dim=0)
                    merged_grid_thw = [
                        len(grid_thw_list),
                        grid_thw_list[0][1],
                        grid_thw_list[0][2],
                    ]
                    hie_vit = Vit(
                        vit_id=i,
                        request_id=vl_request.request_id,
                        embs=merged_tensor,
                        vit_grid_thw=merged_grid_thw,
                        status=VLStatusCode.VL_SUCCESS,
                    )
                    new_hie_results.add_vit(hie_vit)
        return new_hie_results

    def forward(self, vl_request: VLRequest):
        """
        一体化forward函数,dashllm调用推理入口
        内部执行步骤: 前处理 --> 获取vit缓存 --> vit推理 --> llm推理

        Args:
            vl_request (VLRequest): 输入参数

        Yields:
            _type_: 返回迭代器, dashllm需要for loop获取每次llm输出,直到结束或返回错误
        """
        vl_request.vit_time = time.time()
        # init vit results
        hie_results = VitGroup(request_id=vl_request.request_id)
        images = None
        vl_pre_resp = None
        if (
            vl_request.images_info is None
            and vl_request.audios_info is None
            and vl_request.preprocess_req is None
        ):
            # only run llm if no images
            pass
        else:
            if vl_request.preprocess_req is not None:
                try:
                    # check valid input preprocess
                    self.check_valid_config_input(vl_request)
                    vl_pre_resp = self.get_image_audio_from_request(vl_request)
                    vl_pre_resp = self.build_new_request_from_cache(
                        vl_pre_resp, vl_request
                    )
                except ValueError as e:
                    logger_error(
                        step=VlSlsStep.vl_input_error,
                        request_id=vl_request.request_id,
                        context={"msg": f"error:{e}"},
                    )
                    yield VLStatusCode.VL_REQUEST_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                    return
                except error.VisionDecodeError as e:
                    logger_error(
                        step=VlSlsStep.vl_preprocess_error,
                        request_id=vl_request.request_id,
                        context={"msg": f"error:{e}"},
                    )
                    yield VLStatusCode.VL_VISION_DECODE_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                    return
                except error.InputTokensError as e:
                    logger_error(
                        step=VlSlsStep.vl_preprocess_error,
                        request_id=vl_request.request_id,
                        context={"msg": f"error:{e}"},
                    )
                    yield VLStatusCode.VL_INPUT_TOKENS_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                    return
            else:
                try:
                    self.get_cache_vit_result_from_request(vl_request, hie_results)
                except Exception as e:
                    logger_error(
                        step=VlSlsStep.vl_cache_error,
                        request_id=vl_request.request_id,
                        context={"msg": f"error:{e}"},
                    )
                    yield VLStatusCode.VL_REQUEST_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                    return
            images = None

            if vl_request.images_info and isinstance(vl_request.images_info, dict):
                images = vl_request.images_info.get("images")

            if (
                images is None
                and vl_request.audios_info
                and isinstance(vl_request.audios_info, dict)
            ):
                images = vl_request.audios_info.get("audios")
            # data check
            try:
                if vl_request.images_info:
                    self.check_valid_images_input(vl_request)
                if vl_request.audios_info:
                    self.check_valid_audios_input(vl_request)
            except ValueError as e:
                logger_error(
                    step=VlSlsStep.vl_input_error,
                    request_id=vl_request.request_id,
                    context={"msg": f"error:{e}"},
                )
                yield VLStatusCode.VL_REQUEST_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                return
            # hie vit forward
            try:
                self.hie_vit_forward(vl_request, hie_results)
            except ValueError as e:
                logger_error(
                    step=VlSlsStep.vl_vit_error,
                    request_id=vl_request.request_id,
                    context={"msg": f"error:{e}"},
                )
                yield VLStatusCode.VL_OTHER_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                return
            # vit end log
            logger_info(
                step=VlSlsStep.vl_vit_end,
                request_id=vl_request.request_id,
                context={
                    "image_num": len(images) if images else 0,
                    "vit_num": len(hie_results),
                    "vit_len": {
                        result.get_vit_id(): result.get_vit_len()
                        for result in hie_results
                    },
                },
                interval=Interval(
                    type="vit_time",
                    cost=int((time.time() - vl_request.vit_time) * 1000),
                ),
            )
        # check hie result
        for result in hie_results:
            if result.status != VLStatusCode.VL_SUCCESS:
                logger_error(
                    step=VlSlsStep.vl_vit_error,
                    request_id=vl_request.request_id,
                    context={
                        "msg": f"vit error result.status:{result.status} result.vit_id:{result.vit_id}",
                    },
                )
                yield result.status, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
                return
        if images is not None and len(images) != len(hie_results):
            logger_error(
                step=VlSlsStep.vl_vit_error,
                request_id=vl_request.request_id,
                context={
                    "msg": f"request id:{vl_request.request_id}, image num:{len(images)}, hie result num:{len(hie_results)}"
                },
            )
            yield VLStatusCode.VL_VIT_NUMS_MISMATCH_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
            return

        hie_results = self.merge_vit_results(vl_request, hie_results, vl_pre_resp)
        # import os
        # m1 = torch.load(os.path.join("./", "emb.pt")).cpu().float()
        # hie_results = [{"id": 0, "result": m1, "preprocess_time": 1.1, "forward_time": 1 }]
        # format as request
        if self.profile and len(hie_results) > 0:
            # only profile first image
            vl_request.vit_time = time.time() - vl_request.vit_time
            # logging.info(f"vit time: {vl_request.vit_time}, request id:{vl_request.request_id}")
            vl_request.vit_cache_time = sum(
                result.vit_cache_time for result in hie_results
            )
            vl_request.vit_preprocess_time = sum(
                result.vit_preprocess_time for result in hie_results
            )
            vl_request.vit_forward_time = sum(
                result.vit_forward_time for result in hie_results
            )
            vl_request.vit_len = sum(result.get_vit_len() for result in hie_results)
        try:
            # allspark forward
            as_request = self.get_as_request(vl_request, hie_results)
            if self.hie_model_type == "GUMMY-AL":
                # output hie vit results
                yield VLStatusCode.VL_SUCCESS, vl_request.request_id, allspark.GenerateRequestStatus.Init, [
                    vit.to(torch.float32) for vit in as_request.vit_embs
                ], []
            yield from self.as_worker.eval(as_request)
        except error.InputNullTokensError:
            yield VLStatusCode.VL_SUCCESS, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
        except ValueError as e:
            logger_error(
                step=VlSlsStep.vl_as_error,
                request_id=vl_request.request_id,
                context={"msg": f"error:{e}"},
            )
            yield VLStatusCode.VL_INPUT_FORMAT_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []
        except AssertionError as e:
            logger_error(
                step=VlSlsStep.vl_as_error,
                request_id=vl_request.request_id,
                context={"msg": f"error:{e}"},
            )
            yield VLStatusCode.VL_VIT_NUMS_MISMATCH_ERROR, vl_request.request_id, allspark.GenerateRequestStatus.Init, [], []

    def terminate(self):
        if self.hie_worker is not None:
            self.hie_worker.terminate()
        if self.as_worker is not None:
            self.as_worker.terminate()
        pass
