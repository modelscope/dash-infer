'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    fastapi-server.py
'''
import os
import argparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from time import time
from typing import List, Optional
from dashinfer.allspark._allspark import GenerateRequestStatus
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.prompt_utils import PromptTemplate
from dashinfer.allspark.runtime_config import AsModelRuntimeConfigBuilder
import uvicorn
import json
import modelscope

# 解析命令行参数
parser = argparse.ArgumentParser(description="FastAPI server with custom options")
parser.add_argument('--use-modelscope', default=True, action='store_true', help='Download model from modelscope or not')
parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device to use for inference')
parser.add_argument('--model-name', default='Qwen/Qwen2.5-7B-Instruct', help='ModelScope name ')
parser.add_argument('--max-batch', type=int, default=8, help='Maximum batch size')
parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
parser.add_argument('--int8-kv-cache', action='store_true', help='Use int8 for kv-cache')
parser.add_argument('--host', default='0.0.0.0', help='Host to run the FastAPI server on')
parser.add_argument('--port', type=int, default=8000, help='Port to run the FastAPI server on')

args = parser.parse_args()

app = FastAPI()

# if use in memory serialize, change this Gflag to True
in_memory = False
init_quant = False
weight_only_quant = True
device_list = [0] if args.device == 'cuda' else []
fetch_output_mode = "async"
use_converted_model = False

output_base_folder="output_qwen"
tmp_dir = "model_output"

if args.use_modelscope:
    modelscope_model_name = args.model_name
    model_local_path = modelscope.snapshot_download(modelscope_model_name)
else:
    model_local_path = os.getenv("MODEL_PATH", "/model")

safe_model_name = "qwen"

model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name, in_memory_serialize=in_memory, trust_remote_code=True)
engine = allspark.Engine()


model_loader.read_model_config()

target_device = TargetDevice.CUDA if args.device == 'cuda' else TargetDevice.CPU

runtime_cfg_builder = None

if use_converted_model:
    model_convert_folder = os.path.join(output_base_folder, safe_model_name)
    runtime_cfg_builder = (AsModelRuntimeConfigBuilder().model_name("qwen").compute_unit(target_device, device_list).max_batch(args.max_batch))
    #runtime_cfg_builder.model_dir("DIR_TO_MODEL", "FILE_NAME")
    runtime_cfg_builder.max_length(args.max_length)
else:
    model_loader.load_model().serialize_to_path(engine, tmp_dir, enable_quant=init_quant, weight_only_quant=weight_only_quant, skip_if_exists=False).free_model();
    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, target_device, device_list, max_batch=args.max_batch)
    runtime_cfg_builder.max_length(args.max_length)

if args.int8_kv_cache:
    runtime_cfg_builder.kv_cache_mode(allspark.AsCacheMode.AsCacheQuantI8)

# set model dir if use converted model.
runtime_cfg = runtime_cfg_builder.build()

engine.install_model(runtime_cfg)

if in_memory:
    model_loader.free_memory_serialize_file()

engine.start_model(safe_model_name)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    prompt: str
    stream: Optional[bool] = False

async def generate_text(messages):
    input_str = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
    input_str = PromptTemplate.apply_chatml_template(input_str)
    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    gen_cfg.update({"top_k": 1, "repetition_penalty": 1.1})
    status, handle, queue = engine.start_request_text(safe_model_name, model_loader, input_str, gen_cfg)

    generated_ids = []
    status = queue.GenerateStatus()

    while (status == GenerateRequestStatus.Init
           or status == GenerateRequestStatus.Generating
           or status == GenerateRequestStatus.ContextFinished):
        elements = queue.Get()
        if elements is not None:
            generated_ids += elements.ids_from_generate
            yield model_loader.init_tokenizer().get_tokenizer().decode(elements.ids_from_generate)
        status = queue.GenerateStatus()
        if status == GenerateRequestStatus.GenerateFinished:
            break
        if status == GenerateRequestStatus.GenerateInterrupted:
            break

    engine.release_request(safe_model_name, handle)

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if request.stream:
        async def event_generator():
            chunk_id = 0
            async for delta in generate_text([Message(role="user", content=request.prompt)]):
                chunk = {
                    "id": f"cmpl-{os.urandom(16).hex()}-{chunk_id}",
                    "object": "text_completion",
                    "created": int(time()),
                    "model": safe_model_name,
                    "choices": [
                        {
                            "text": delta,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                chunk_id += 1

            # Send the final chunk with finish_reason
            final_chunk = {
                "id": f"cmpl-{os.urandom(16).hex()}",
                "object": "text_completion",
                "created": int(time()),
                "model": safe_model_name,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        result = ""
        async for delta in generate_text([Message(role="user", content=request.prompt)]):
            result += delta

        response = {
            "id": f"cmpl-{os.urandom(16).hex()}",
            "object": "text_completion",
            "created": int(time()),
            "model": safe_model_name,
            "choices": [
                {
                    "text": result,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(result.split()),
                "total_tokens": len(request.prompt.split()) + len(result.split())
            }
        }
        return JSONResponse(content=response)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    if request.stream:
        async def event_generator():
            chunk_id = 0
            async for token in generate_text(request.messages):
                chunk = {
                    "id": f"chatcmpl-{os.urandom(16).hex()}-{chunk_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time()),
                    "model": safe_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": token
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                chunk_id += 1

            # Send the final chunk with finish_reason
            final_chunk = {
                "id": f"chatcmpl-{os.urandom(16).hex()}",
                "object": "chat.completion.chunk",
                "created": int(time()),
                "model": safe_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        result = ""
        async for token in generate_text(request.messages):
            result += token

        response = {
            "id": "chatcmpl-" + os.urandom(16).hex(),
            "object": "chat.completion",
            "created": int(time()),
            "model": safe_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                "completion_tokens": len(result.split()),
                "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + len(result.split())
            }
        }

        return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
