'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    vit.py
'''
from typing import Optional, List
import enum
import torch
import time


class VitStatus(enum.Enum):
    """Status of vit"""

    VL_SUCCESS = 0
    VL_REQUEST_ERROR = 1
    VL_FILE_ERROR = 2
    VL_IMAGE_FORMAT_ERROR = 3
    VL_VIT_NUMS_MISMATCH_ERROR = 4
    VL_INPUT_FORMAT_ERROR = 5
    VL_OTHER_ERROR = 100
    INIT_WAITING = 101
    INIT_DONE = 102
    TEMINATE_WAITING = 103
    TERMINATE_DONE = 104


class Vit:
    """Stores the vit result, status and profiles."""

    def __init__(
        self,
        request_id: Optional[int] = 0,
        vit_id: Optional[int] = 0,
        status=VitStatus.INIT_WAITING,
        vit_preprocess_time=0,
        vit_forward_time=0,
        vit_cache_time=0,
        embs: Optional[torch.Tensor] = None,
        from_cache: bool = False,
        vit_grid_thw: Optional[List[int]] = None,
    ) -> None:
        self.vit_id = vit_id
        self.status = status
        self.vit_embs = embs
        self.vit_preprocess_time = vit_preprocess_time
        self.vit_forward_time = vit_forward_time
        self.vit_cache_time = vit_cache_time
        self.request_id = request_id
        self.from_cache = from_cache
        # vit grid thw
        self.vit_grid_thw = vit_grid_thw
        # time to put request
        self.start_time = time.time()

    def is_terminated(self) -> bool:
        return self.status == VitStatus.TEMINATE_WAITING

    def set_embs(self, embs: torch.Tensor) -> None:
        """Set vit embs."""
        self.vit_embs = embs

    def get_embs(self) -> torch.Tensor:
        """Get vit embs."""
        return self.vit_embs

    def get_vit_len(self) -> int:
        """Get vit length."""
        return self.vit_embs.shape[0] if self.vit_embs is not None else 0

    def get_vit_id(self) -> int:
        return self.vit_id

    def is_from_cache(self) -> bool:
        return self.from_cache

    def get_vit_grid_thw(self) -> List[int]:
        return self.vit_grid_thw


class VitGroup:
    """a group of vit."""

    def __init__(
        self,
        request_id: int,
        vits: Optional[List[Vit]] = None,
    ) -> None:
        self.request_id = request_id
        self.vits_dict = {}
        if vits is not None:
            self.vits_dict = {vit.vit_id: vit for vit in vits}

    def add_vit(self, vit: Vit) -> None:
        """Add vit to vit group."""
        self.vits_dict[vit.vit_id] = vit

    def get_forward_result(self) -> List[Vit]:
        return [vit for vit in self.vits_dict.values() if vit.is_from_cache() is False]

    def get_cache_result(self) -> List[Vit]:
        return [vit for vit in self.vits_dict.values() if vit.is_from_cache() is True]

    def clear(self) -> None:
        self.vits_dict.clear()

    def __len__(self) -> int:
        """Return the length of vits_dict."""
        return len(self.vits_dict)

    def __iter__(self):
        """Return the iterator object."""
        return iter(self.vits_dict.values())

    def __getitem__(self, key):
        return self.vits_dict[key]
