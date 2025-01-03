'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    server.py
'''
from ..vl_inference.utils.model_loader import HuggingFaceVLModel
from ..vl_inference.utils.hie.vit_preprocess import get_image_preprocessor
from ..vl_inference.utils.qwen_vl_status import VLStatusCode
from ..vl_inference.utils.env import getenv
from ..vl_inference.runtime.qwen_vl import QwenVl
from ..vl_inference.utils.config import VitConfig, CacheConfig
from dashinfer import allspark
from ..vl_inference.runtime.qwen_vl import VLRequest
from ..vl_inference.utils.env import setenv
import torch
from .protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    DeltaMessage,
    ErrorResponse,
    LogProbs,
    UsageInfo,
)
from .constants import (
    ErrorCode,
)
from .config import (
    get_model_context,
    add_context_args,
    load_args_into_config,
)
import asyncio
import argparse
import json
import os
from typing import Optional, Union, Dict, List, Any

import aiohttp
import fastapi
from fastapi import Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

from pydantic_settings import BaseSettings
import shortuuid
import tiktoken
import uvicorn
import logging
import sys

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format="[%(levelname)-7s]  %(message)s",
)


conv_template_map = {}

fetch_timeout = aiohttp.ClientTimeout(total=3 * 3600)


def init():
    setenv()
    context = get_model_context()
    print(context.context)
    model = context.get("model")
    tensor_parallel_size = context.get("parallel_size")
    chat_format = "CHATML"
    context.set("chat_format", chat_format)

    # -----------------------Convert Model------------------------
    home_dir = os.environ.get("HOME") or "/root"
    output_dir = os.path.join(home_dir, ".cache/as_model/", model.split("/")[-1])
    model_name = "model"
    data_type = context.get("dtype")

    model_loader = HuggingFaceVLModel(
        model,
        model_name,
        in_memory_serialize=False,
        user_set_data_type=data_type,
        trust_remote_code=True,
        vision_engine=context.get("vision_engine"),
        fp8=context.get("fp8"),
    )
    (
        model_loader.load_model(direct_load=False, load_format="auto")
        .serialize(model_output_dir=output_dir)
        .free_model()
    )
    as_graph_path = os.path.join(output_dir, model_name + ".asgraph")
    as_weight_path = os.path.join(output_dir, model_name + ".asparam")
    # vit_model_path = os.path.join(output_dir, model_name + ".onnx")
    vit_model_path = model_loader.vision_model_path

    context.set("output_dir", output_dir)
    context.set("model_name", model_name)
    # -----------------------Init Engine--------------------------
    config = Qwen2VLConfig.from_pretrained(
        model_loader.hf_model_path,
        trust_remote_code=True,
        revision=None,
        code_revision=None,
    )
    model_type = config.model_type.upper().replace("_", "-")
    tokenizer = model_loader.tokenizer

    context.set("eos_token_id", tokenizer.eos_token_id)

    context.set("model_type", model_type)
    context.set("tokenizer", tokenizer)
    cuda_devices = [str(i) for i in range(tensor_parallel_size)]
    compute_unit = "CUDA:" + ",".join(cuda_devices)
    # allspark model config
    as_model_config = allspark.AsModelConfig(
        model_name=model_name,
        model_path=as_graph_path,
        weights_path=as_weight_path,
        engine_max_length=context.get("max_length"),
        engine_max_batch=context.get("max_batch"),
        compute_unit=compute_unit,
        enable_prefix_cache=True if context.get("enable_prefix_cache") else False,
    )
    # vit config
    vit_config = VitConfig(
        model_path=vit_model_path,
        precision="fp16",
        workers=1,
        backend=context.get("vision_engine"),
    )
    # redis cache config
    cache_config = CacheConfig(
        url=getenv("VL_REDIS_URL", "127.0.0.1"),
        port=getenv("VL_REDIS_PORT", 6379),
        passwd=getenv("VL_REDIS_PASSWD", "1234"),
        valid_cache_time=getenv("VL_VALID_CACHE_TIME", 300000),
    )
    vl_engine = QwenVl(
        as_config=as_model_config,
        vit_config=vit_config,
        cache_config=cache_config,
        trt_vit_config=model_loader.vit_config,
    )
    preprocessor = get_image_preprocessor(
        workers=vit_config.workers,
        vl_version=2 if model_type == "QWEN2-VL" else 1,
        dtype=torch.float16,
    )
    context.set("preprocessor", preprocessor)
    context.set("vl_engine", vl_engine)


async def fetch_remote(url, pload=None, name=None):
    async with aiohttp.ClientSession(timeout=fetch_timeout) as session:
        async with session.post(url, json=pload) as response:
            chunks = []
            if response.status != 200:
                ret = {
                    "text": f"{response.reason}",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
                return json.dumps(ret)

            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)

    if name is not None:
        res = json.loads(output)
        if name != "":
            res = res[name]
        return res

    return output


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: Optional[List[str]] = None


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


# lock = asyncio.Lock()
@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def root(request: ChatCompletionRequest):
    task = asyncio.to_thread(_create_chat_completion, request)
    return await task


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).model_dump(), status_code=400
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))


# async def check_model(request) -> Optional[JSONResponse]:
#     controller_address = app_settings.controller_address
#     ret = None

#     models = await fetch_remote(controller_address + "/list_models", None, "models")
#     if request.model not in models:
#         ret = create_error_response(
#             ErrorCode.INVALID_MODEL,
#             f"Only {'&&'.join(models)} allowed now, your model {request.model}",
#         )
#     return ret


async def check_length(request, prompt, max_tokens, worker_addr):
    if (
        not isinstance(max_tokens, int) or max_tokens <= 0
    ):  # model worker not support max_tokens=None
        max_tokens = 1024 * 1024

    context_len = await fetch_remote(
        worker_addr + "/model_details", {"model": request.model}, "context_length"
    )
    token_num = await fetch_remote(
        worker_addr + "/count_token",
        {"model": request.model, "prompt": prompt},
        "count",
    )
    length = min(max_tokens, context_len - token_num)

    if length <= 0:
        return None, create_error_response(
            ErrorCode.CONTEXT_OVERFLOW,
            f"This model's maximum context length is {context_len} tokens. However, your messages resulted in {token_num} tokens. Please reduce the length of the messages.",
        )

    return length, None


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'top_p'",
        )
    if request.top_k is not None and (request.top_k > -1 and request.top_k < 1):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_k} is out of Range. Either set top_k to -1 or >=1.",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                logging.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            try:
                decoding = tiktoken.model.encoding_for_model(model_name)
            except KeyError:
                logging.warning("Warning: model not found. Using cl100k_base encoding.")
                model = "cl100k_base"
                decoding = tiktoken.get_encoding(model)
            inp = [decoding.decode(text) for text in inp]

    return inp


def create_openai_logprobs(logprob_dict):
    """Create OpenAI-style logprobs."""
    return LogProbs(**logprob_dict) if logprob_dict is not None else None


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


def get_images(url, preprocessor, min_pixels, max_pixels, fps=None):
    context = get_model_context()
    select_image = [url]
    select_image_path = []
    extra_info = []

    for i, image in enumerate(select_image):
        model_type = context.get("model_type")
        if model_type == "QWEN1-VL":
            status, pre_image, vit_len = preprocessor.get_patch_image(image)
            pre_image = pre_image["image"]
        elif model_type == "QWEN2-VL":
            vision_format = "video"  # TODO: video?
            status, pre_image, vit_len = preprocessor.get_vision_info(
                url=image,
                type=vision_format,
                fps=fps,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            extra_info.append(pre_image["grid_thw"])
            pre_image = pre_image["image"]
        else:
            status, pre_image, vit_len = preprocessor.get_audio_info(image)
            extra_info.append(pre_image["mask"])
            pre_image = pre_image["feature"]
            print(
                f"pre_image shape:{pre_image.shape}, atten_mask shape:{extra_info[i].shape}"
            )
        if status == VLStatusCode.VL_SUCCESS:
            select_image_path.append(image)
            select_image[i] = pre_image
    return select_image, select_image_path, extra_info


def get_vl_request(
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
    max_tokens: Optional[int],
    max_completion_tokens: Optional[int],
):
    context = get_model_context()
    from .conversation import Conversation, SeparatorStyle

    conv = Conversation(
        name="qwen2-vl-chatml",
        system_template="<|im_start|>system\n{system_message}",
        system_message="",
        messages=[],
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str="<|im_end|>",
    )

    # Some information is got from engine initialization
    vl_request = VLRequest(
        shortuuid.random(),
        context.get("tokenizer"),
        context.get("chat_format"),
        context.get("model_type"),
        context.get("preprocessor"),
        context.get("min_pixels"),
        context.get("max_pixels"),
    )

    if isinstance(messages, str):
        raise NotImplementedError(
            "VL Server does not support messages without vision inputs"
        )
    else:
        for message in messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.set_system_message(message["content"])
            elif msg_role == "user" or msg_role == "assistant":
                if isinstance(message["content"], list):
                    (text, image_list) = conv.get_content(message["content"])
                    if msg_role == "user":
                        conv.append_message(conv.roles[0], (text, image_list))
                    else:
                        conv.append_message(conv.roles[1], (text, image_list))
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f"Unkown role: {msg_role}")

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f"id: {vl_request.request_id}, prompt: {repr(prompt)}")
        input_ids = conv.get_input_tokens(prompt, context.get("tokenizer"))

        preprocess_req = conv.get_preprocess_req(
            min_pixels=context.get("min_pixels"), max_pixels=context.get("max_pixels")
        )
        vl_request.preprocess_req = preprocess_req
        generate_config = {
            "num_beams": 1,
            "num_return_sequences": 1,
            "temperature": temperature,
            "do_sample": True,
            "early_stopping": False if os.getenv("VLM_BENCHMARK") == "1" else True,
            "top_k": 1,
            "top_p": top_p,
            "max_length": (
                max_tokens + len(input_ids)
                if max_tokens is not None
                else max_completion_tokens + len(input_ids)
            ),
            "min_length": 5,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": 1.05,
            "length_penalty": 1,
            "stop_words_ids": [[151643], [151644], [151645]],
            "eos_token_id": context.get("eos_token_id"),
            "seed": 1234567,
        }
        print("generation config: ", generate_config)
        vl_request.gen_cfg = generate_config
        vl_request.context_length = len(input_ids)
        vl_request.input_tokens = [input_ids]

    vl_request.truncate_lengths = conv.get_truncate_length()

    return vl_request


async def get_worker_address(model_name: str) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address
    worker_addr = await fetch_remote(
        controller_address + "/get_worker_address", {"model": model_name}, "address"
    )

    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")
    logging.debug(f"model_name: {model_name}, worker_addr: {worker_addr}")
    return worker_addr


async def get_conv(model_name: str, worker_addr: str):
    conv_template = conv_template_map.get((worker_addr, model_name))
    if conv_template is None:
        conv_template = await fetch_remote(
            worker_addr + "/worker_get_conv_template", {"model": model_name}, "conv"
        )
        conv_template_map[(worker_addr, model_name)] = conv_template
    return conv_template


@app.get("/info", dependencies=[Depends(check_api_key)])
def info():
    context = get_model_context()
    return context.get("model_name")


def _create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    context = get_model_context()
    vl_request = get_vl_request(
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=request.max_tokens,
        max_completion_tokens=request.max_completion_tokens,
    )

    qwen_vl = context.get("vl_engine")
    gen = qwen_vl.forward(vl_request)
    if request.stream:

        def stream_gen():
            for gen_result in gen:
                if len(gen_result) == 5:
                    vl_status, id, as_status, output, prompt_nums = gen_result
                else:
                    vl_status, id, as_status, output = gen_result
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,  # TODO: deponed on request param n
                    delta=DeltaMessage(
                        content=context.get("tokenizer").decode(
                            output,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    ),
                    finish_reason=(
                        "stop"
                        if as_status == allspark.GenerateRequestStatus.GenerateFinished
                        else None
                    ),
                )
                chunk = ChatCompletionStreamResponse(
                    id=id, choices=[choice_data], model=context.get("model_name")
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    finish_reason = "stop"
    for gen_result in gen:
        if len(gen_result) == 5:
            vl_status, id, as_status, output, prompt_nums = gen_result
        else:
            vl_status, id, as_status, output = gen_result
        if vl_status != VLStatusCode.VL_SUCCESS:
            # vit/vl error
            logging.error("id: {} vl_status: {}".format(id, vl_status))
            finish_reason = None
        vl_request.output_ids.extend(output)
        # print(
        #     "id: {}, vl_status:{}, as_status: {}, output: {}".format(
        #         id,
        #         vl_status,
        #         as_status,
        #         (
        #             context.get("tokenizer").decode(
        #                 vl_request.output_ids,
        #                 skip_special_tokens=True,
        #                 clean_up_tokenization_spaces=False,
        #             )
        #             if len(vl_request.output_ids) > 0
        #             else "None"
        #         ),
        #     )
        # )
    output_text = context.get("tokenizer").decode(
                    vl_request.output_ids,
                    skip_special_tokens=True
                )
    choice = [
        ChatCompletionResponseChoice(
            index=0,  # TODO: depend on request param n
            message=ChatMessage(
                role="assistant",
                content=output_text,
            ),
            finish_reason=finish_reason,
        )
    ]
    usage = UsageInfo()
    # usage.prompt_tokens = vl_request.context_length
    usage.prompt_tokens = vl_request.as_context_len
    usage.completion_tokens = len(vl_request.output_ids)
    usage.total_tokens = vl_request.context_length + len(vl_request.output_ids)
    return ChatCompletionResponse(
        model=context.get("model_name"), choices=choice, usage=usage
    )


async def generate_completion(payload: Dict[str, Any], worker_addr: str):
    return await fetch_remote(worker_addr + "/worker_generate", payload, "")


async def get_embedding(payload: Dict[str, Any]):
    model_name = payload["model"]
    worker_addr = await get_worker_address(model_name)

    embedding = await fetch_remote(worker_addr + "/worker_get_embeddings", payload)
    return json.loads(embedding)


def create_openai_api_server():
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    add_context_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.api_keys = args.api_keys

    logging.info(f"args: {args}")
    return args


def main():
    args = create_openai_api_server()
    load_args_into_config(args)
    init()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
