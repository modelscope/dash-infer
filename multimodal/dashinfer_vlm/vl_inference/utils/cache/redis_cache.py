'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    redis_cache.py
'''
import torch
import logging

# REDIS_URL='r-uf67m1z7q8n22bcjj2.redis.rds.aliyuncs.com'
# REDIS_PASSWORD='DashScope2023!'
# REDIS_PORT=6379
# cache valid time 5min


class RedisCache:
    def __init__(self, host, port=6379, passwd="", valid_cache_time=300000):
        from aquila_core import TensorStore

        logging.info(
            f"init redis cache host:{host}, port:{port}, valid_cache_time:{valid_cache_time}"
        )
        self.tensor_store = TensorStore(host=host, port=port, password=passwd)
        self.valid_cache_time = valid_cache_time

    def insert(self, key, input_tensor, **kwargs):
        sync = kwargs.get("sync", False)
        grid_thw = kwargs.get("grid_thw", None)
        if grid_thw:
            # cat grid_thw to input_tensor
            grid_tensor = torch.tensor(grid_thw, dtype=input_tensor.dtype)
            grid_tensor = grid_tensor.view(1, grid_tensor.shape[0]).expand(
                input_tensor.shape[0], grid_tensor.shape[0]
            )
            input_tensor = torch.cat((input_tensor, grid_tensor), dim=1)
        try:
            if sync is True:
                result = self.tensor_store.SetTorch(
                    key=key, tensor=input_tensor, timeout_ms=self.valid_cache_time
                )
            else:
                result = self.tensor_store.AsyncSetTorch(
                    key=key, tensor=input_tensor, timeout_ms=self.valid_cache_time
                )
            return result
        except Exception:
            logging.error("redis insert error")
            return False

    def get(self, key, **kwargs):
        for i in range(0, 1):
            try:
                result = self.tensor_store.GetTorch(key=key)
                grid_list = None
                if result:
                    seq_len, hz = result.shape
                    # grid list exists
                    if hz % 1024 == 3:
                        embs = result[:, : hz - 3]
                        grid_list = result[0, hz - 3 :].tolist()
                        result = embs
                return True, result, grid_list
            except Exception:
                logging.error("redis get error")
                break
        return False, None, None

    def get_tensor(self, key: str):
        return self.get(key)
