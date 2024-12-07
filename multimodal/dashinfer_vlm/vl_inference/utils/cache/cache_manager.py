'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    cache_manager.py
'''
from .redis_cache import RedisCache
from .local_cache import LocalCache
from ..config import CacheConfig
from ..env import getenv
import hashlib
import concurrent.futures
import time
from typing import Union, List
import logging


class CacheManager:
    def __init__(self, cache_config: CacheConfig) -> None:
        if getenv("VL_REDIS_ENABLE", "0") == "1":
            self.redis_valid = True
        else:
            logging.info("*********redis cache disable***********")
            self.redis_valid = False
        if not cache_config or cache_config.check_valid_config() is False:
            self.redis_valid = False
            self.redis_cache = None
        if self.redis_valid is True:
            # redis cache
            self.redis_cache = RedisCache(
                host=cache_config.url,
                port=cache_config.port,
                passwd=cache_config.passwd,
                valid_cache_time=cache_config.valid_cache_time,
            )
        # local cache
        self.retry_redis = False
        if getenv("VL_LOCAL_ENABLE", "1") == "1":
            self.local_cache = LocalCache(
                max_cache_size=int(getenv("VL_LOCAL_MAX_CACHE", "1024"))
            )
            self.local_valid = True
        else:
            logging.info(
                "*********local cache disable, use tmp local cache for vit merge result***********"
            )
            self.local_cache_merge = LocalCache(
                max_cache_size=int(getenv("VL_LOCAL_MAX_CACHE", "1024"))
            )
            self.local_valid = False
        self.cache_retry_time = int(getenv("VL_CACHE_INVALID_TIME_MS_RETRY", "10000"))
        self.redis_max_time = 3000
        self.retry_start_time = 0

    def insert(self, md5_hash, input_tensor, **kwargs):
        # local cache
        # redis cache
        res = True
        # use_redis = kwargs.get("use_redis", False)
        use_redis = False
        if isinstance(md5_hash, str) and input_tensor is not None:
            grid_thw = kwargs.get("grid_thw", None)
            combined_value = {"tensor": input_tensor, "list": grid_thw}
            if self.local_valid:
                self.local_cache.insert(md5_hash, combined_value)
            else:
                self.local_cache_merge.insert(md5_hash, combined_value)
            if self.redis_valid and use_redis:
                res = self.redis_cache.insert(md5_hash, input_tensor, **kwargs)
        return res

    @staticmethod
    def get_md5_key(raw_key: str):
        md5_hasher = hashlib.md5()
        if isinstance(raw_key, str):
            md5_hasher.update(raw_key.encode())
        return md5_hasher.hexdigest()

    # add a static function to get md5 key only use url
    @staticmethod
    def get_md5_keys(image_urls):
        res = []
        if len(image_urls) >= 3:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                res = list(executor.map(CacheManager.get_md5_key, image_urls))
        else:
            for key in image_urls:
                res.append(CacheManager.get_md5_key(key))
        return res

    def get(self, md5_keys: Union[str, List], **kwargs) -> Union[List[dict], dict]:
        # images url to md5
        def get_torch_func(md5_hash, **kwargs):
            res = {}
            cache_start_time = time.time()
            cache_from_local = True
            local_cache_time = 0
            cache_tensor = None
            grid_list = None
            status = False
            use_redis = kwargs.get("use_redis", False)
            if not isinstance(md5_hash, str):
                return {"key": "", "result": None, "cache_time": 0, "status": True}
            if self.local_valid:
                cache_res = self.local_cache.get_tensor(md5_hash)
                status = True
                if cache_res is not None:
                    cache_tensor = cache_res.get("tensor", None)
                    grid_list = cache_res.get("list", None)
                local_cache_time = int((time.time() - cache_start_time) * 1000)
            else:
                cache_res = self.local_cache_merge.get_tensor(md5_hash)
                status = True
                if cache_res is not None:
                    cache_tensor = cache_res.get("tensor", None)
                    grid_list = cache_res.get("list", None)
            if (
                use_redis is True
                and cache_tensor is None
                and self.redis_valid
                and self.retry_redis
            ):
                cache_from_local = False
                redis_start_time = time.time()
                status, cache_tensor, grid_list = self.redis_cache.get_tensor(md5_hash)
                redis_end_time = int((time.time() - redis_start_time) * 1000)
                if status is False or redis_end_time > self.redis_max_time:
                    self.retry_start_time = time.time()
                    self.retry_redis = False
            if use_redis is True and self.redis_valid and self.retry_redis is False:
                if (
                    int((time.time() - self.retry_start_time) * 1000)
                    > self.cache_retry_time
                ):
                    self.retry_start_time = 0
                    if self.redis_cache is not None:
                        self.retry_redis = True
            if status is True and cache_tensor is not None:
                if grid_list is not None and grid_list != kwargs.get(
                    "grid_thw", grid_list
                ):
                    cache_tensor = None
                    grid_list = None
                elif grid_list is None:
                    if (
                        kwargs.get("vit_len", cache_tensor.shape[0])
                        != cache_tensor.shape[0]
                    ):
                        cache_tensor = None
                        grid_list = None
            res["key"] = md5_hash
            res["result"] = cache_tensor
            res["grid_thw"] = grid_list
            res["local"] = local_cache_time
            res["cache_time"] = int((time.time() - cache_start_time) * 1000)
            res["status"] = status
            res["source"] = "local" if cache_from_local is True else "global"
            return res

        res = []
        if isinstance(md5_keys, List):
            grid_thws = kwargs.get("grid_thws", None)
            vit_lens = kwargs.get("vit_lens", None)
            if grid_thws is None:
                grid_thws = [None] * len(md5_keys)
            if vit_lens is None:
                vit_lens = [None] * len(md5_keys)
            task_list = [
                (
                    md5_hash,
                    {
                        "grid_thw": grid_thw,
                        "vit_len": vit_len,
                        "use_redis": kwargs.get("use_redis", True),
                    },
                )
                for md5_hash, grid_thw, vit_len in zip(md5_keys, grid_thws, vit_lens)
            ]

            def wrapper(task):
                md5_hash, kwargs = task  # 解包元组
                return get_torch_func(md5_hash, **kwargs)  # 传递特定的 kwargs

            if len(md5_keys) >= 3:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        res = list(executor.map(wrapper, task_list))
            else:
                task_list = [
                    (md5_hash, {"grid_thw": grid_thw})
                    for md5_hash, grid_thw in zip(md5_keys, grid_thws)
                ]
                for task in task_list:
                    res.append(wrapper(task))
        else:
            res = get_torch_func(md5_keys, **kwargs)
        return res
        # for key in keys:
        #     cache_start_time = time.time()
        #     md5_hash = hashlib.md5(key.encode()).hexdigest()
        #     cache_tensor = self.redis_cache.get_tensor(md5_hash)
        #     res.append({"key":md5_hash, "result": cache_tensor, "cache_time": int((time.time() - cache_start_time)*1000)})
        # return res

    def get_vits_from_cache(self, **kwargs):
        keys = CacheManager.get_md5_keys(kwargs["urls"])
        return self.get(keys, **kwargs)

    def get_vit_from_cache(self, **kwargs):
        key = CacheManager.get_md5_key(kwargs["url"])
        return self.get(key, **kwargs)
