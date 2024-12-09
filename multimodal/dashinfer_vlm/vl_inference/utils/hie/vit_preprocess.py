'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    vit_preprocess.py
'''
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math
import torch
import requests
from ..qwen_vl_status import VLStatusCode
from ..cache.local_cache import LocalCache
from functools import lru_cache
import random
import base64
import concurrent.futures
from io import BytesIO
from typing import Union, List
import numpy as np
import subprocess
import logging


@lru_cache(maxsize=1)
def video_sniper_codec_is_available():
    try:
        return True
    except Exception as e:
        logging.warning("_video_sniper_codec_is_available e:", e)
        return False


@lru_cache(maxsize=1)
def video_torchvision_is_available():
    try:
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def video_torchvision_set_video_backend():
    import torchvision

    try:
        torchvision.set_video_backend("cuda")
        print("torchvision.set_video_backend cuda")
        return
    except Exception:
        pass

    try:
        torchvision.set_video_backend("video_reader")
        print("torchvision.set_video_backend video_reader")
        return
    except Exception:
        pass

    print("torchvision.set_video_backend pyav")


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    if height < 10 or width < 10:
        raise ValueError(f"height:{height} or width:{width} must be larger than 10")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def get_video_frame_count(url) -> int:
    import cv2

    video_capture = cv2.VideoCapture(url)

    if not video_capture.isOpened():
        print("Error: Unable to open video stream.")
        return 0
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()

    return frame_count


def fetch_image(image: str) -> torch.Tensor:
    image_obj = None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0",
            "Accept-Encoding": None,
        }
        if image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            image_obj = Image.open(response.raw)
        elif image.startswith("data:image"):
            data = image.split(";", 1)[1]
            if data.startswith("base64,"):
                data = base64.b64decode(data[7:])
                image_obj = Image.open(BytesIO(data))
            else:
                raise ValueError(
                    "Unrecognized image input, support local path, http url, base64 and "
                    "PIL.Image, got {image}"
                )
        else:
            image_obj = Image.open(image)
    except Exception as e:
        return (
            VLStatusCode.VL_VISION_DECODE_ERROR,
            None,
            f"fetch_image get failed, image:{image}, error:{e}",
        )
    image_tensor = transforms.functional.pil_to_tensor(image_obj.convert("RGB"))
    return VLStatusCode.VL_SUCCESS, image_tensor, None


def fetch_video_by_sniper_codec(
    url: str,
    fps: str,
    nframe_factor=2,
    use_cuda=True,
    workers=1,
    timeout=10,
    max_content_length=150 * 1024 * 1024,
):
    if not use_cuda:
        config = {
            "url": url,
            "output_fps": fps,
            "soft_decode": "true",
            "output_format": "rgb",
        }
    else:
        config = {
            "url": url,
            "output_fps": fps,
            "gpu_id": str(random.randint(0, workers - 1)),
            "soft_decode": "false",
            "output_format": "rgb",
            "output_device": "cpu",
        }
    # print(f"config:{config}")
    import sniper_codec

    client = sniper_codec.VideoDecoderClient(config)

    status = client.Init()
    if status != sniper_codec.CodecStatus.CODEC_SUCCESS:
        logging.error(
            f"fetch_video_by_sniper_codec init failed, error:{status}, url:{url}"
        )
        raise ValueError(
            f"fetch_video_by_sniper_codec init failed, error:{status}, url:{url}"
        )
    status = client.Start()
    if status != sniper_codec.CodecStatus.CODEC_SUCCESS:
        logging.error(
            f"fetch_video_by_sniper_codec start failed, error:{status}, url:{url}"
        )
        raise ValueError(
            f"fetch_video_by_sniper_codec start failed, error:{status}, url:{url}"
        )

    video_frames = []
    while True:
        frame = client.GetVideoFrame()
        if frame is None or len(video_frames) == 768:
            break
        # # frame meta
        # print("width: ", frame.width(),
        #       ", height: ", frame.height(),
        #       ", pitch: ", frame.pitch(),
        #       ", format: ", frame.format(),
        #       ", ts: ", frame.ts(),
        #       ", idx: ", frame.index())
        video_frames.append(
            torch.utils.dlpack.from_dlpack(frame.GetData().ToDLPack()).permute(2, 0, 1)
        )

    client.Stop()

    while len(video_frames) % nframe_factor != 0:
        video_frames.pop()

    return VLStatusCode.VL_SUCCESS, torch.stack(video_frames), None


def fetch_video_by_torchvision(url: str, fps: float = 2, nframe_factor=2):
    import torchvision

    video_torchvision_set_video_backend()
    video, _, info = torchvision.io.read_video(
        url,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    )
    nframes = video.size(0) / info["video_fps"] * fps
    nframes = min(round_by_factor(nframes, nframe_factor), 768)
    idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
    video = video[idx]
    return VLStatusCode.VL_SUCCESS, video, None


def fetch_video(
    url: str,
    fps: float = 2,
    use_cuda=True,
    workers=1,
    max_content_length=150 * 1024 * 1024,
):
    # use_sniper_codec = video_sniper_codec_is_available()
    use_sniper_codec = False
    use_torchvision = video_torchvision_is_available()
    status = VLStatusCode.VL_VISION_DECODE_ERROR
    error = None
    video_frames = None
    if use_sniper_codec:
        try:
            status, video_frames, error = fetch_video_by_sniper_codec(
                url, str(fps), use_cuda=use_cuda, workers=workers
            )
            if video_frames is not None:
                return status, video_frames, None
            elif use_cuda:
                # use cpu to do again if cuda decode error
                status, video_frames, error = fetch_video_by_sniper_codec(
                    url, str(fps), use_cuda=False, workers=workers
                )
                if video_frames is not None:
                    return status, video_frames, None
        except Exception as e:
            logging.error(
                f"fetch_video fetch_video_by_sniper_codec failed, error:{e}, url:{url}"
            )
            return (
                status,
                None,
                f"fetch_video fetch_video_by_sniper_codec failed, error:{e}, url:{url}",
            )
    if use_torchvision and video_frames is None:
        try:
            status, video_frames, error = fetch_video_by_torchvision(url, fps)
            if video_frames is not None:
                return status, video_frames, None
        except Exception as e:
            logging.error(
                f"fetch_video fetch_video_by_torchvision failed, error:{e}, url:{url}"
            )
            return (
                status,
                None,
                f"fetch_video fetch_video_by_torchvision failed, error:{e}, url:{url}",
            )

    logging.error(
        f"fetch_video failed failed, use_sniper_codec:{use_sniper_codec}, url:{url}"
    )
    return status, None, error


# ffmpeg to read audio from url
def ffmpeg_read_audio(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        ) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError(
            "ffmpeg was not found but is required to load audio files from filename"
        ) from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio


class Preprocessor:
    def __init__(self, **kwargs):
        from transformers import WhisperFeatureExtractor
        from ..env import getenv

        if getenv("VL_SNIPER_CUDA_DECODE", "0") == "1":
            # workers for cuda preprocess
            self.workers = kwargs.get("workers", torch.cuda.device_count())
            if self.workers <= 0:
                self.use_cuda = False
                logging.warning("No GPU found, using CPU instead.")
            else:
                self.use_cuda = True
                logging.warning("Using %d GPU(s) for preprocess." % self.workers)
        else:
            self.workers = 0
            self.use_cuda = False
            logging.warning("force using CPU instead.")
        self.vl_version = 2 if kwargs.get("model_type", "QWEN2-VL") == "QWEN2-VL" else 1
        self.dtype = kwargs.get("dtype", torch.float16)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.patch_size = 14
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.feature_extractor = WhisperFeatureExtractor(feature_size=128)
        self.min_pixels = 28 * 28 * 4
        self.max_pixels = 28 * 28 * 5120
        self.fps = 1
        # only cache for video type image list
        self.local_cache = LocalCache(max_cache_size=128)
        # audio
        self.model_type = getenv("QWEN_MODEL_TYPE", "QWEN2-VL").upper()
        self.padding = "max_length" if self.model_type == "QWEN2-AL" else "longest"

    def get_vit_seq_len(self, resized_h, resized_w, temporal=2):
        grid_t = temporal // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        return grid_t * grid_h * grid_w // (self.merge_size * self.merge_size)

    def get_patch_image(self, url, device=torch.device("cpu")):
        status, image, error = fetch_image(url)
        if status != VLStatusCode.VL_SUCCESS:
            return status, {"url": url, "error": error}, 0
        _, h, w = image.shape
        resized_height, resized_width = smart_resize(h, w)

        # 将图像转换为Tensor并移动到CUDA上
        # image_tensor = self.tensor_transform(image).to(device)
        image = (
            transforms.functional.resize(
                image,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
            )
            .float()
            .div(255)
        )
        patch_image = self.image_transform(image)
        return (
            VLStatusCode.VL_SUCCESS,
            {
                "type": "image",
                "image": patch_image.unsqueeze(0).to(self.dtype),
                "url": url,
            },
            self.get_vit_seq_len(resized_height, resized_width),
        )  # batch

    @lru_cache(maxsize=10)
    def get_vision_info(
        self,
        url: Union[str, tuple],
        type: str,
        fps: float,
        min_pixels: int,
        max_pixels: int,
    ):
        # print(f"get_vision_info: url:{url}, type:{type}, fps:{fps}, min_pixels:{min_pixels}, max_pixels:{max_pixels}")
        if type == "image":
            if self.vl_version == 1:
                # qwen1 vl image preprocess
                return self.get_patch_image(url)
            # qwen2 vl image preprocess
            status, images, error = fetch_image(url)
            if status != VLStatusCode.VL_SUCCESS:
                return status, {"url": url, "error": error}, 0
            images = images.unsqueeze(0).expand(self.temporal_patch_size, -1, -1, -1)
        elif type == "video":
            if isinstance(url, tuple):
                images_list = []
                first_url = url[0]
                status, image, error = fetch_image(first_url)
                if status != VLStatusCode.VL_SUCCESS:
                    return status, {"url": first_url, "error": error}, 0
                first_h, first_w = image.shape[1], image.shape[2]
                images_list.append(image)
                for one_url in url[1:]:
                    status, image, error = fetch_image(one_url)
                    if status != VLStatusCode.VL_SUCCESS:
                        return status, {"url": one_url, "error": error}, 0
                    if image.shape[1] != first_h or image.shape[2] != first_w:
                        image = transforms.functional.resize(
                            image, size=(first_h, first_w)
                        )
                    images_list.append(image)
                # add last frame
                while len(images_list) % self.temporal_patch_size != 0:
                    images_list.append(images_list[-1])
                images = torch.stack(images_list)
            else:
                # sniper hardware accel to decode video
                status, images, error = fetch_video(
                    url, fps, self.use_cuda, self.workers
                )
                if status != VLStatusCode.VL_SUCCESS:
                    return status, {"url": url, "error": error}, 0
            video_frames = images.shape[0]
            if video_frames < self.temporal_patch_size:
                return (
                    VLStatusCode.VL_VISION_DECODE_ERROR,
                    {
                        "url": url,
                        "error": f"video frame num:{video_frames} less than {self.temporal_patch_size}",
                    },
                    0,
                )
            max_pixels = ((self.patch_size * self.merge_size) ** 2) * min(
                768, 30720 / video_frames * 2
            )
            if min_pixels > max_pixels:
                min_pixels = max_pixels
        else:
            return (
                VLStatusCode.VL_REQUEST_ERROR,
                {"url": url, "error": f"type error type:{type}"},
                0,
            )
        # [T, 3, H, W]
        temporal, n_channel, h, w = images.shape

        # h, w, self.patch_size, self.merge_size 1365 2048 14 2
        resized_height, resized_width = smart_resize(
            h,
            w,
            factor=self.patch_size * self.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        images = (
            transforms.functional.resize(
                images,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
            )
            .float()
            .div(255)
        )

        # [T, 3, H, W]
        patch_image = self.image_transform(images)
        grid_t = patch_image.size(0) // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

        # resized_height, self.patch_size, resized_width 812 14 1204
        # grid_t, grid_h, grid_w 1 58 86

        llm_h, llm_w = grid_h // self.merge_size, grid_w // self.merge_size
        vit_seq_len = grid_t * grid_h * grid_w
        # [T/2, 2, 3, H/28, 2, 14, W/28, 2, 14]
        flatten_image = patch_image.reshape(
            grid_t,
            self.temporal_patch_size,
            n_channel,
            llm_h,
            self.merge_size,
            self.patch_size,
            llm_w,
            self.merge_size,
            self.patch_size,
        )
        # [T/2, H/28, W/28, 2, 2, 3, 2, 14, 14]
        flatten_image = flatten_image.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # [vit_seq_len, 3*2*14*14]
        vit_flatten_image = flatten_image.reshape(
            vit_seq_len,
            n_channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return (
            VLStatusCode.VL_SUCCESS,
            {
                "type": type,
                "image": vit_flatten_image.to(self.dtype),
                "grid_thw": [grid_t, grid_h, grid_w],
                "url": url,
            },
            self.get_vit_seq_len(resized_height, resized_width, temporal),
        )

    def show_cache_info(self):
        # 获取缓存信息
        cache_stats = self.get_video_image_list.cache_info()
        print(f"Cache hits: {cache_stats.hits}")
        print(f"Cache misses: {cache_stats.misses}")
        print(f"Cache maxsize: {cache_stats.maxsize}")
        print(f"Current cache size: {cache_stats.currsize}")

    @lru_cache(maxsize=128)
    def get_video_image_list(
        self,
        urls: tuple,
        type: str,
        fps: float,
        min_pixels: int,
        max_pixels: int,
        height: int,
        width: int,
    ):
        url_value = ""
        first_h = height
        first_w = width
        # if type == 'video' and self.vl_version == 2:
        if self.vl_version == 2:
            images_list = []
            url_status = {}  # 一个字典来保持每个 URL 对应的状态

            def fetch_and_store(url, index):
                status, image, error = fetch_image(url)
                url_status[index] = (status, image, error)  # 存储状态
                return index  # 返回索引

            # 使用 ThreadPoolExecutor 来执行函数
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(fetch_and_store, url, index): index
                    for index, url in enumerate(urls)
                }
                for future in concurrent.futures.as_completed(futures):
                    pass
                for index, url in enumerate(urls):
                    status, image, error = url_status[index]
                    if status != VLStatusCode.VL_SUCCESS:
                        return status, {"url": urls[index], "error": error}, 0
                    # logging.info(f"image shape: {image.shape}")
                    if first_h == 0 or first_w == 0:
                        first_h, first_w = image.shape[1], image.shape[2]
                    elif image.shape[1] != first_h or image.shape[2] != first_w:
                        # resize image to the same size
                        image = transforms.functional.resize(
                            image, size=(first_h, first_w)
                        )
                    # logging.info(f"image shape2: {image.shape} first_h:{first_h} first_w:{first_w}")
                    images_list.append(image)  # 按照索引顺序添加图像
                    url_value += f"{url}"
                    # logging.info(f"index: {index}, status: {status}, error: {error} url_value: {url_value}")
            images = torch.stack(images_list)
            if images.size(0) != self.temporal_patch_size:
                logging.error(
                    f"qwen2 vl image preprocess: images.size(0) != self.temporal_patch_size, {images.size(0)} != {self.temporal_patch_size}"
                )
                return (
                    VLStatusCode.VL_REQUEST_ERROR,
                    {
                        "url": url_value,
                        "error": f"qwen2 vl image preprocess: images.size(0) != self.temporal_patch_size, {images.size(0)} != {self.temporal_patch_size}",
                    },
                    0,
                )
        else:
            return (
                VLStatusCode.VL_REQUEST_ERROR,
                {
                    "url": urls[0],
                    "error": f"qwen2 vl image preprocess: type or version error, type:{type}, version:{self.vl_version}",
                },
                0,
            )
        # [T, 3, H, W]
        temporal, n_channel, h, w = images.shape

        # h, w, self.patch_size, self.merge_size 1365 2048 14 2
        resized_height, resized_width = smart_resize(
            h,
            w,
            factor=self.patch_size * self.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        images = (
            transforms.functional.resize(
                images,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
            )
            .float()
            .div(255)
        )

        # [T, 3, H, W]
        patch_image = self.image_transform(images)
        grid_t = patch_image.size(0) // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

        # resized_height, self.patch_size, resized_width 812 14 1204
        # grid_t, grid_h, grid_w 1 58 86

        llm_h, llm_w = grid_h // self.merge_size, grid_w // self.merge_size
        vit_seq_len = grid_t * grid_h * grid_w
        # [T/2, 2, 3, H/28, 2, 14, W/28, 2, 14]
        flatten_image = patch_image.reshape(
            grid_t,
            self.temporal_patch_size,
            n_channel,
            llm_h,
            self.merge_size,
            self.patch_size,
            llm_w,
            self.merge_size,
            self.patch_size,
        )
        # [T/2, H/28, W/28, 2, 2, 3, 2, 14, 14]
        flatten_image = flatten_image.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # [vit_seq_len, 3*2*14*14]
        vit_flatten_image = flatten_image.reshape(
            vit_seq_len,
            n_channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return (
            VLStatusCode.VL_SUCCESS,
            {
                "type": type,
                "image": vit_flatten_image.to(self.dtype),
                "grid_thw": [grid_t, grid_h, grid_w],
                "url": url_value,
                "height": first_h,
                "width": first_w,
            },
            self.get_vit_seq_len(resized_height, resized_width, temporal),
        )

    def batch_get_vision_info_once(self, configs: List):
        results = []
        height = 0
        width = 0
        if len(configs) % self.temporal_patch_size != 0:
            return [
                (
                    VLStatusCode.VL_REQUEST_ERROR,
                    {
                        "url": None,
                        "error": f"len(config):{len(configs)} mismatch, self.temporal_patch_size:{self.temporal_patch_size}",
                    },
                    0,
                )
            ]
        else:
            urls = [config.get("url") for config in configs]
            for i in range(0, len(urls), self.temporal_patch_size):
                if i == 0:
                    result = self.get_video_image_list(
                        tuple(urls[i : i + self.temporal_patch_size]),
                        configs[i].get("type"),
                        configs[i].get("fps", self.fps),
                        configs[i].get("min_pixels", self.min_pixels),
                        configs[i].get("max_pixels", self.max_pixels),
                        height=height,
                        width=width,
                    )
                    height = (
                        result[1].get("height")
                        if result[0] == VLStatusCode.VL_SUCCESS
                        else 0
                    )
                    width = (
                        result[1].get("width")
                        if result[0] == VLStatusCode.VL_SUCCESS
                        else 0
                    )
                    results.append(result)
                else:
                    results.append(
                        self.get_video_image_list(
                            tuple(urls[i : i + self.temporal_patch_size]),
                            configs[i].get("type"),
                            configs[i].get("fps", self.fps),
                            configs[i].get("min_pixels", self.min_pixels),
                            configs[i].get("max_pixels", self.max_pixels),
                            height=height,
                            width=width,
                        )
                    )
                # results.append(self.get_video_image_list(tuple(urls[i:i + self.temporal_patch_size]), configs[i].get('type'), configs[i].get('fps', self.fps), configs[i].get('min_pixels', self.min_pixels), configs[i].get('max_pixels', self.max_pixels)))
        # info = self.get_vision_info.cache_info()
        # print(f"cache info: {info}")
        return results

    def batch_get_video_image_list(self, config):
        results = []
        if len(config.get("url")) == 0:
            return [
                (
                    VLStatusCode.VL_REQUEST_ERROR,
                    {
                        "url": None,
                        "error": f"len(config):{len(config.get('url'))}, self.temporal_patch_size:{self.temporal_patch_size}",
                    },
                    0,
                )
            ]
        # copy last url to fill
        while len(config.get("url")) % self.temporal_patch_size != 0:
            config.get("url").append(config.get("url")[-1])
        urls = [url for url in config.get("url")]
        urls = urls[:768]
        height = 0
        width = 0
        for i in range(0, len(urls), self.temporal_patch_size):
            tuple_urls = tuple(urls[i : i + self.temporal_patch_size])
            type = config.get("type")
            fps = config.get("fps", self.fps)
            min_pixels = config.get("min_pixels", self.min_pixels)
            max_pixels = config.get("max_pixels", self.max_pixels)
            fallback_key = (tuple_urls, type, fps, min_pixels, max_pixels, 0, 0)
            if i == 0:
                result = self.get_video_image_list(
                    tuple_urls,
                    type,
                    fps,
                    min_pixels,
                    max_pixels,
                    height=height,
                    width=width,
                )
                height = (
                    result[1].get("height")
                    if result[0] == VLStatusCode.VL_SUCCESS
                    else 0
                )
                width = (
                    result[1].get("width")
                    if result[0] == VLStatusCode.VL_SUCCESS
                    else 0
                )
                results.append(result)
                # self.show_cache_info()
                self.local_cache.insert(fallback_key, result)
            else:
                cache_res = self.local_cache.get(fallback_key)
                if cache_res is not None:
                    results.append(self.local_cache.get(fallback_key))
                else:
                    results.append(
                        self.get_video_image_list(
                            tuple_urls,
                            type,
                            fps,
                            min_pixels,
                            max_pixels,
                            height=height,
                            width=width,
                        )
                    )
                    # self.show_cache_info()
        # info = self.get_vision_info.cache_info()
        # print(f"cache info: {info}")
        # logging.info(f"batch_get_video_image_list: {results}")
        return results

    def batch_get_vision_info(self, configs: List):
        results = []
        if len(configs) == 0:
            return [
                (
                    VLStatusCode.VL_REQUEST_ERROR,
                    {"url": None, "error": "len of config is 0"},
                    0,
                )
            ]
        else:
            # use executor.map to speed up
            # get_vision_info has lru_cache attribute, use multi-thread may cause cache miss
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     # 提取每个配置项，并将其作为参数传递给 get_vision_info
            #     results = list(executor.map(lambda config: self.get_vision_info(
            #         url=config.get('url'),
            #         type=config.get('type'),
            #         min_pixels=config.get('min_pixels', self.min_pixels),
            #         max_pixels=config.get('max_pixels', self.max_pixels),
            #         fps=config.get('fps', self.fps)
            #     ), configs))
            for config in configs:
                # 对于type video url list情况,需要特殊处理成 batch_get_vision_info_once,主要是为了兼容视频对话这种一对图片做缓存的场景
                if config.get("type") == "video" and isinstance(
                    config.get("url"), list
                ):
                    results.append(self.batch_get_video_image_list(config))
                    continue
                results.append(
                    self.get_vision_info(
                        config.get("url"),
                        config.get("type"),
                        config.get("fps", self.fps),
                        config.get("min_pixels", self.min_pixels),
                        config.get("max_pixels", self.max_pixels),
                    )
                )
        # info = self.get_vision_info.cache_info()
        # print(f"cache info: {info}")
        return results

    def get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def get_extra_audio_info(self, inputs):
        feature_attention_mask = inputs["attention_mask"]
        input_features = inputs["input_features"]
        audio_feat_lengths, audio_output_lengths = self.get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )
        # print(f"input_features.shape:{input_features.shape}, audio_feat_lengths:{audio_feat_lengths}, audio_output_lengths:{audio_output_lengths} sum:{feature_attention_mask.sum(-1)}")
        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=audio_feat_lengths.dtype,
                device=audio_feat_lengths.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand
        # print(f"padding_mask:{padding_mask.shape} seq_range:{seq_range}, lengths_expand:{lengths_expand} shape: {padding_mask.shape} seq_range shape: {seq_range.shape} lengths_expand shape: {lengths_expand.shape}")
        # audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
        #     batch_size, 1, max_seq_len, max_seq_len
        # )
        audio_attention_mask_ = padding_mask

        audio_attention_mask = audio_attention_mask_.to(dtype=torch.float16)
        audio_attention_mask[audio_attention_mask_] = float("-inf")
        return audio_attention_mask

    def get_audio_seq_len(self, audio_attention_mask):
        non_zero_indices = torch.nonzero(audio_attention_mask.squeeze()).squeeze()
        audio_seq_len = (
            non_zero_indices[0]
            if non_zero_indices.numel() > 0
            else audio_attention_mask.shape[1]
        )
        if isinstance(audio_seq_len, torch.Tensor):
            audio_seq_len = audio_seq_len.item()
        if self.model_type == "QWEN2-AL":
            return (audio_seq_len - 2) // 2 + 1
        else:
            audio_seq_len = (audio_seq_len - 1) // 2 + 1
            return (audio_seq_len - 1) // 2 + 1

    # get raw audio info and post-process as hie input
    @lru_cache(maxsize=10)
    def get_audio_info(self, audio_path, audio_waveform=None):
        status, inputs, error = self.get_raw_audio_info(audio_path, audio_waveform)
        if status != VLStatusCode.VL_SUCCESS:
            return VLStatusCode.VL_REQUEST_ERROR, {"url": audio_path, "error": error}, 0
        input_features = inputs["input_features"]
        audio_mask = self.get_extra_audio_info(inputs)
        return (
            VLStatusCode.VL_SUCCESS,
            {
                "type": "audio",
                "feature": input_features,
                "mask": audio_mask,
                "url": audio_path,
            },
            self.get_audio_seq_len(audio_mask),
        )

    def batch_get_audio_info(self, configs: List):
        results = []
        if len(configs) == 0:
            return [
                (
                    VLStatusCode.VL_REQUEST_ERROR,
                    {"url": None, "error": "len of configs is 0"},
                    0,
                )
            ]
        elif len(configs) == 1:
            result = self.get_audio_info(configs[0].get("url"))
            return [result]
        else:
            # use executor.map to speed up
            # get_vision_info has lru_cache attribute, use multi-thread may cause cache miss
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     # 提取每个配置项，并将其作为参数传递给 get_vision_info
            #     results = list(executor.map(lambda config: self.get_audio_info(
            #         url=config['url']
            #     ), configs))
            for config in configs:
                results.append(self.get_audio_info(config.get("url")))
        return results

    def get_raw_audio_info(self, audio_path, audio_waveform=None):
        # import librosa
        if audio_waveform is None:
            try:
                if audio_path.startswith("http://") or audio_path.startswith(
                    "https://"
                ):
                    response = requests.get(audio_path)
                    response.raise_for_status()
                    # audio_obj = BytesIO(response.content)
                    audio_obj = response.content
                else:

                    def read_audio_file_to_bytes(file_path: str) -> bytes:
                        with open(file_path, "rb") as audio_file:
                            return audio_file.read()

                    audio_obj = read_audio_file_to_bytes(audio_path)
            except requests.exceptions.RequestException as e:
                return (
                    VLStatusCode.VL_REQUEST_ERROR,
                    None,
                    f"Failed to download audio: {e}",
                )
            except IOError as e:
                return (
                    VLStatusCode.VL_FILE_ERROR,
                    None,
                    f"Failed to read audio file: {e}",
                )
            except Exception as e:
                return (
                    VLStatusCode.VL_OTHER_ERROR,
                    None,
                    f"Failed to process audio: {e}",
                )
            # audio_waveform, sr = librosa.load(audio_obj, sr=16000)
            audio_waveform = ffmpeg_read_audio(audio_obj, sampling_rate=16000)
        inputs = self.feature_extractor(
            audio_waveform,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding=self.padding,
        )
        return VLStatusCode.VL_SUCCESS, inputs, None


def get_image_preprocessor(**kwargs):
    return Preprocessor(**kwargs)
