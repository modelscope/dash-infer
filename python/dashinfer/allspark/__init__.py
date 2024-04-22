#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    __init__.py
#
from .engine import Engine
from .engine import ClientEngine
from ._allspark import *
import os

__all__ = [
    "AsStatus",
    "AsModelConfig",
    "Engine",
    "GenerateRequestStatus",
    # functions
    "save_allsparky_dltensor_tofile",
    "set_global_header",
]
# since some pytorch will link same version openmp,
# without this env will be a segfault.
os.environ["KMP_DUPLICATE_LIB_OK"] = "YES"
