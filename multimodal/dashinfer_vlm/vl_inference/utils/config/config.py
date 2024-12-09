'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    config.py
'''
class VitConfig:
    """Configuration for the vit.

    Args:
        model_path: path to vit onnx or hie model.
        precision: fp16/fp32 precision of the model.
        workers: numbers of the vit instances
        profile: profile data in running
    """

    def __init__(
        self,
        model_path: str = "",
        precision: str = "fp16",
        workers: int = 1,
        profile: bool = False,
        backend: str = "tensorrt",
    ) -> None:
        self.model_path = model_path
        self.precision = precision
        self.workers = workers
        self.profile = profile
        self.backend = backend
        self._verify_args()

    def _verify_args(self) -> None:
        if self.workers <= 0 or self.precision not in ["fp16", "fp32"]:
            raise ValueError(
                f"workers:{self.workers} should be >= 1, "
                f"or precision: {self.precision} shoule fp16 or fp32"
            )

    def check_valid_config(self) -> bool:
        if not self.model_path or not self.precision or not self.workers:
            return False
        return True


class CacheConfig:
    """Configuration for the vit cache.

    Args:
        url: redis url
        port: redis port
        passwd: redis passwd
        valid_cache_time: valid cache time in milliseconds
    """

    def __init__(
        self,
        url: str,
        port: int,
        passwd: str,
        valid_cache_time: int = 300000,
    ) -> None:
        self.url = url
        self.port = port
        self.passwd = passwd
        self.valid_cache_time = valid_cache_time

    def check_valid_config(self) -> bool:
        if not self.url or not self.port or not self.passwd:
            return False
        return True


QWEN_MODEL_TYPES = (
    "QWEN1-VL",
    "QWEN2-VL",
    "QWEN2-AL",
    "GUMMY-AL",
)

SPECIAL_TOKENS_DICT = {
    "IMAGE_BOS": 151857,
    "IMAGE_EOS": 151858,
    "IMAGE_TARGET": 151859,
    "IMAGE_BOS_V2": 151652,
    "IMAGE_EOS_V2": 151653,
    "IMAGE_TARGET_V2": 151859,
    "AUDIO_BOS": 151647,
    "AUDIO_EOS": 151648,
    "AUDIO_TARGET": 151646,
}
