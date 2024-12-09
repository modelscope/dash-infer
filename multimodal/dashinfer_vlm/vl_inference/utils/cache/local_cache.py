'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    local_cache.py
'''
import logging
import threading
from cachetools import LRUCache


class LocalCache:
    def __init__(self, max_cache_size=1024, valid_cache_time=300000):
        logging.info(
            f"init local cache max_cache_size:{max_cache_size}, valid_cache_time:{valid_cache_time}"
        )
        # cache time is not supported in local cache
        self.max_cache_size = max_cache_size
        self.valid_cache_time = valid_cache_time
        self.lock = threading.Lock()
        self.lru_cache = LRUCache(maxsize=max_cache_size)

    def insert(self, key, data):
        with self.lock:
            self.lru_cache[key] = data
            return True

    def get(self, key):
        with self.lock:
            return self.lru_cache.get(key, None)

    def get_tensor(self, key: str):
        return self.get(key)
