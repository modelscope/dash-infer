'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    vl_logger.py
'''
import logging
from datetime import datetime
from typing import Dict
from ..utils.env import getenv
from ..utils.env import DS_SVC_ID, DS_SVC_NAME
from ..utils.qwen_vl_status import Interval
from ..utils.env import setenv
import json

if len(DS_SVC_ID) == 0:
    # test env
    print("call setenv()")
    setenv()
logger = logging.getLogger("qwen-vl")
logger.setLevel(logging.INFO)
if getenv("LOCAL_TEST", "0") == "0":
    from aquila_core import LoggingHandler

    logger.addHandler(LoggingHandler())
else:
    ch = logging.StreamHandler()
    logger.addHandler(ch)


def _create_log(
    step: str,
    model: str,
    code: str,
    message: str,
    request_id: str,
    context: Dict,
    interval: Interval,
) -> str:
    """Convert parameters to a specific format

    :param step: for the convenience of sls query, use a name to represent the current step, such as tokenizer_start
    :param code: error code
    :param message: error message
    :param request_id:
    :param context: all context for easy troubleshooting. it is a "json.dumps" str
    :return:
    """
    # return '{} | {} | {} | {} | {} | {}'.format(step, code, message, request_id, context, cost)
    return json.dumps(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "step": step,
            "model": model,
            "code": code,
            "message": message,
            "request_id": request_id,
            "context": context,
            "interval": {"type": interval.type, "cost": interval.cost},
            "service_id": DS_SVC_ID,
            "service_name": DS_SVC_NAME,
            "ds_service_id": DS_SVC_ID,
            "ds_service_name": DS_SVC_NAME,
        },
        ensure_ascii=False,
    )


def _parse_args(**kwargs):
    step = kwargs.get("step", "")
    model = kwargs.get("model", "")
    code = kwargs.get("code", "")
    message = kwargs.get("message", "")
    request_id = kwargs.get("request_id", "")
    context = kwargs.get("context", "")
    interval = kwargs.get("interval", Interval())
    return step, model, code, message, request_id, context, interval


def logger_info(**kwargs):
    step, model, code, message, request_id, context, interval = _parse_args(**kwargs)
    logger.info(_create_log(step, model, code, message, request_id, context, interval))


def logger_error(**kwargs):
    step, model, code, message, request_id, context, interval = _parse_args(**kwargs)
    logger.error(
        _create_log(step, model, str(code), message, request_id, context, interval)
    )


class VlSlsStep:
    vl_preprocess_start = "vl_preprocess_start"
    vl_preprocess_error = "vl_preprocess_error"
    vl_preprocess_end = "vl_preprocess_end"
    vl_input_error = "vl_input_error"
    vl_cache_end = "vl_cache_end"
    vl_cache_error = "vl_cache_error"
    vl_vit_start = "vl_vit_start"
    vl_vit_error = "vl_vit_error"
    vl_vit_end = "vl_vit_end"
    vl_as_start = "vl_as_start"
    vl_as_first_token = "vl_as_first_token"
    vl_as_error = "vl_as_error"
    vl_as_stop = "vl_as_stop"
    vl_streaming_cache = "vl_streaming_cache"
    vl_streaming_truncate_start = "vl_streaming_truncate_start"
    vl_streaming_truncate_end = "vl_streaming_truncate_end"
