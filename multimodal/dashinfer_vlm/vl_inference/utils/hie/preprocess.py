'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    preprocess.py
'''
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math
import torch
import requests
from ..qwen_vl_status import VLStatusCode
import base64
from io import BytesIO
from typing import Dict
import numpy as np
import subprocess


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
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
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


def get_vit_seq_len(h, w, patch_size=14, patch_emb_size=2):
    resized_height, resized_width = smart_resize(h, w)
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    seq_len = (
        grid_h * grid_w / (patch_emb_size * patch_emb_size)
    )  ## patch visual emb again
    return seq_len


def get_video_frame_count(url) -> int:
    import cv2

    video_capture = cv2.VideoCapture(url)

    if not video_capture.isOpened():
        print("Error: Unable to open video stream.")
        return 0
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()

    return frame_count


def fetch_image(ele: Dict) -> torch.Tensor:
    image = ele["image"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    if image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            "Unrecognized image input, support local path, http url, base64 and "
            "PIL.Image, got {image}"
        )
    image_tensor = transforms.functional.pil_to_tensor(image_obj.convert("RGB"))
    return image_tensor


def fetch_video(
    ele: Dict, ctx, nframe_factor=2, device=torch.device("cpu")
) -> torch.Tensor:
    # sniper hardware accel to decode video
    from sniper import sniper
    from torch.utils.dlpack import from_dlpack

    fps = 1.0
    if "fps" in ele:
        fps = float(ele["fps"])
    elif "nframes" in ele:
        frames = get_video_frame_count(ele["video"])
        if frames <= 0:
            raise ValueError("Invalid video file")
        fps = float(ele["nframes"]) / frames
    else:
        raise ValueError("Invalid config(either fps or nframes should be set)")
    # get from sniper
    config_decoder = {
        "url": ele["video"],
        "output_fps": str(fps),
        "auto_decode": "true" if device.type == "cpu" else "false",
        "soft_decode": "true" if device.type == "cpu" else "false",
    }
    decoder_client = sniper.codec.VideoDecoderClient(config_decoder)

    if decoder_client.Init() != sniper.codec.CodecStatus.CODEC_SUCCESS:
        raise ValueError("Failed to initialize decoder")
    if decoder_client.Start() != sniper.codec.CodecStatus.CODEC_SUCCESS:
        raise ValueError("Failed to start decoder")
    input_tensors = []
    while True:
        decoded_buf = decoder_client.GetVideoFrame()
        if decoded_buf is None:
            break
        tensor_dlpack = decoded_buf.GetData().ToDLPack()
        torch_tensor = from_dlpack(tensor_dlpack)
        input_tensors.append(torch_tensor)
    if len(input_tensors) == 0:
        raise None
    if len(input_tensors) % nframe_factor != 0:
        input_tensors = input_tensors[:-1]
    torch_out = torch.stack(input_tensors, dim=0)
    torch_out = torch_out.permute(0, 3, 1, 2)[:, [2, 1, 0], :, :]
    return torch_out


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


# fetch_video + decode
# def fetch_video(ele: Dict, nframe_factor=2, device=torch.device("cpu"), ctx) -> Tuple[torch.Tensor, bool]:
#     # sniper hardware accel to decode video
#     from sniper import sniper
#     from sniper.sniper import Tensor
#     from torch.utils.dlpack import from_dlpack, to_dlpack
#     fps = 1.0
#     if "fps" in ele:
#         fps = float(ele["fps"])
#     elif "nframes" in ele:
#         frames = get_video_frame_count(ele["video"])
#         if frames <= 0:
#             raise ValueError("Invalid video file")
#         fps = float(ele["nframes"]) / frames
#     else:
#         raise ValueError("Invalid config(either fps or nframes should be set)")
#     # get from sniper
#     config_decoder = {
#         "url": ele["video"],
#         "output_fps": str(fps),
#         "auto_decode": "true" if device.type == "cpu" else "false",
#         "soft_decode": "true" if device.type == "cpu" else "false"
#     }
#     decoder_client = sniper.codec.VideoDecoderClient(config_decoder)
#     config_builder = sniper.preprocess.ConfigBuilder()

#     if decoder_client.Init() != sniper.codec.CodecStatus.CODEC_SUCCESS:
#         raise ValueError("Failed to initialize decoder")
#     if decoder_client.Start() != sniper.codec.CodecStatus.CODEC_SUCCESS:
#         raise ValueError("Failed to start decoder")
#     input_tensors = []
#     ori_width = 0
#     ori_height = 0
#     while True:
#         decoded_buf = decoder_client.GetVideoFrame()
#         if decoded_buf is None:
#             break
#         ori_height = decoded_buf.height()
#         ori_width = decoded_buf.width()
#         input_tensors.append(decoded_buf.GetData())
#     min_pixels = ele.get("min_pixels", 28*28*4)
#     max_pixels = ele.get("max_pixels", 900000)
#     resized_height, resized_width = smart_resize(
#             ori_height,
#             ori_width,
#             factor=14 * 2,
#             min_pixels=min_pixels,
#             max_pixels=max_pixels,
#     )
#     config_preprocess = (
#         config_builder.EnableMeanNorm(enable=True)
#         .Mean(122.7709383, 116.7460125, 104.0937362)
#         .Norm(0.01459843, 0.01500777, 0.01422007)
#         .Resize(resized_width, resized_height, False, 0x0, sniper.preprocess.InterMode.CUBIC)
#         .Transpose([0, 3, 1, 2])
#         .ColorChannelSwitch([0, 1, 2])
#         .Build()
#     )
#     pre_processor = sniper.preprocess.PreProcessor(config_preprocess)
#     output_tensor = pre_processor.AllocateOutputTensor(len(input_tensors), ctx)
#     pre_processor.BatchProcess(ctx, input_tensors, None, output_tensor)
#     out_dlpack = output_tensor.ToDLPack()
#     torch_out = from_dlpack(out_dlpack)
#     return torch_out


class Preprocessor:
    def __init__(self, **kwargs):
        from transformers import WhisperFeatureExtractor

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
        self.context = {}
        self.feature_extractor = WhisperFeatureExtractor(feature_size=128)

    def get_patch_image(self, image_path, device=torch.device("cpu")):
        image = None
        try:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                image = Image.open(image_path)
        except requests.exceptions.RequestException:
            return VLStatusCode.VL_REQUEST_ERROR, None
        except IOError:
            return VLStatusCode.VL_FILE_ERROR, None
        except Exception:
            return VLStatusCode.VL_OTHER_ERROR, None
        image = image.convert("RGB")

        w, h = image.size
        resized_height, resized_width = smart_resize(h, w)

        # 将图像转换为Tensor并移动到CUDA上
        image_tensor = self.tensor_transform(image).to(device)

        resize_transform = transforms.Resize(
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        patch_image = self.image_transform(resize_transform(image_tensor))
        # image = transforms.functional.resize(
        #     image, [resized_height, resized_width],
        #     interpolation=InterpolationMode.BICUBIC
        # )
        # patch_image = image_transform(image)
        return VLStatusCode.VL_SUCCESS, patch_image.unsqueeze(0)  # batch

    def get_vision_info(self, ele, device=torch.device("cpu")):
        if "image" in ele:
            name = ele["image"][:128]
            images = fetch_image(ele)
            images = images.unsqueeze(0).expand(self.temporal_patch_size, -1, -1, -1)
        elif "video" in ele:
            # sniper hardware accel to decode video
            from sniper import sniper

            # update context
            if device not in self.context:
                if device.type == "cpu":
                    self.context[device] = sniper.HwContext.CPU()
                else:
                    self.context[device] = sniper.HwContext.GPU(device.index)
            name = ele["video"][:128]
            images = fetch_video(
                ele,
                ctx=self.context.get(device),
                nframe_factor=self.temporal_patch_size,
                device=device,
            )
            if images is None and device.type != "cpu":
                # use cpu to decode
                images = fetch_video(
                    ele,
                    ctx=self.context.get(device),
                    nframe_factor=self.temporal_patch_size,
                    device=torch.device("cpu"),
                )
            if images is None:
                return VLStatusCode.VL_REQUEST_ERROR, None, None
            num_frames = images.shape[0]
            factor = self.patch_size * self.merge_size
            min_pixels = (factor**2) * 52
            if ele.get("min_pixels") is None:
                ele["min_pixels"] = min_pixels
            max_tokens = ele.get("max_tokens", 6144)
            max_pixels = (factor**2) * min(
                768, (max_tokens / num_frames * self.temporal_patch_size)
            )
            if ele.get("max_pixels") is None:
                ele["max_pixels"] = max_pixels
            # print(f"fetch_video: time: {time.time()-start}")

        else:
            return VLStatusCode.VL_REQUEST_ERROR, None, None
        # [T, 3, H, W]
        _, n_channel, h, w = images.shape
        min_pixels = ele.get(
            "min_pixels",
            self.patch_size * self.patch_size * self.merge_size * self.merge_size * 4,
        )
        max_pixels = ele.get(
            "max_pixels",
            self.patch_size
            * self.patch_size
            * self.merge_size
            * self.merge_size
            * 5120,
        )
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=self.patch_size * self.merge_size,
            )
        else:
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
        grid_t * llm_h * llm_w
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

        info = {
            "name": name,
            "original_height": h,
            "original_width": w,
            "resized_height": resized_height,
            "resized_width": resized_width,
            "vit_grid_t": grid_t,
            "vit_grid_h": grid_h,
            "vit_grid_w": grid_w,
            "vit_seq_len": vit_seq_len,
            "llm_grid_t": grid_t,
            "llm_grid_h": llm_h,
            "llm_grid_w": llm_w,
        }
        return VLStatusCode.VL_SUCCESS, vit_flatten_image, info

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
        print(
            f"input_features.shape:{input_features.shape}, audio_feat_lengths:{audio_feat_lengths}, audio_output_lengths:{audio_output_lengths} sum:{feature_attention_mask.sum(-1)}"
        )
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
        print(
            f"padding_mask:{padding_mask.shape} seq_range:{seq_range}, lengths_expand:{lengths_expand} shape: {padding_mask.shape} seq_range shape: {seq_range.shape} lengths_expand shape: {lengths_expand.shape}"
        )
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
        return (audio_seq_len - 2) // 2 + 1

    # get raw audio info and post-process as hie input
    def get_audio_info(self, audio_path):
        status, inputs = self.get_raw_audio_info(audio_path)
        if status != VLStatusCode.VL_SUCCESS:
            return VLStatusCode.VL_REQUEST_ERROR, None, None
        input_features = inputs["input_features"]
        audio_mask = self.get_extra_audio_info(inputs)
        return VLStatusCode.VL_SUCCESS, input_features, audio_mask

    def get_raw_audio_info(self, audio_path):
        # import librosa
        try:
            if audio_path.startswith("http://") or audio_path.startswith("https://"):
                response = requests.get(audio_path)
                response.raise_for_status()
                # audio_obj = BytesIO(response.content)
                audio_obj = response.content
            else:

                def read_audio_file_to_bytes(file_path: str) -> bytes:
                    with open(file_path, "rb") as audio_file:
                        return audio_file.read()

                audio_obj = read_audio_file_to_bytes(audio_path)
        except requests.exceptions.RequestException:
            return VLStatusCode.VL_REQUEST_ERROR, None
        except IOError:
            return VLStatusCode.VL_FILE_ERROR, None
        except Exception:
            return VLStatusCode.VL_OTHER_ERROR, None
        # audio_waveform, sr = librosa.load(audio_obj, sr=16000)
        audio_waveform = ffmpeg_read_audio(audio_obj, sampling_rate=16000)
        inputs = self.feature_extractor(
            audio_waveform,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
        )
        # inputs = {
        #    'attention_mask': attention_mask
        #    'input_features': input_features
        # }
        return VLStatusCode.VL_SUCCESS, inputs


def get_image_preprocessor():
    return Preprocessor()
