'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    allspark_worker.py
'''
import argparse
import asyncio
import copy
import json
import os
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from dashinfer import allspark
from dashinfer.allspark import AsModelRuntimeConfigBuilder
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark import GenerateRequestStatus, AsStatus

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)


app = FastAPI()


def download_model(model_id, revision):
    source = "huggingface"
    if os.environ.get("FASTCHAT_USE_MODELSCOPE", "False").lower() == "true":
        source = "modelscope"

    logger.info(f"Downloading model {model_id} (revision: {revision}) from {source}")
    if source == "modelscope":
        from modelscope import snapshot_download

        model_dir = snapshot_download(model_id, revision=revision)
    elif source == "huggingface":
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=model_id)
    else:
        raise ValueError("Unknown source")

    logger.info(f"Save model to path {model_dir}")

    return model_dir


TMP_DIR = "../../model_output"

class DashInferWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        revision: str,
        no_register: bool,
        in_memory: bool,
        init_quant: bool,
        weight_only_quant: bool,
        data_type: str,
        max_length: int,
        device_list: List[int],
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: dash-infer worker..."
        )
        # check if model_path is existed at local path
        if not os.path.exists(model_path):
            model_path = download_model(model_path, revision)

        assert(len(self.model_names) == 1)
        safe_model_name = self.model_names[0].replace("/", "_")
        logger.info(
            f"Create a model {safe_model_name} with in_memory_serialize={in_memory}, user_set_data_type={data_type}"
        )
        model_loader = allspark.HuggingFaceModel(model_path, safe_model_name, in_memory_serialize=in_memory, trust_remote_code=True, user_set_data_type=data_type)
        model_loader.init_tokenizer()
        engine = allspark.Engine()

        if in_memory:
            (model_loader.load_model()
             .read_model_config()
             .serialize_to_memory(engine, enable_quant=init_quant, weight_only_quant=weight_only_quant)
             .free_model())
        else:
            (model_loader.load_model()
             .read_model_config()
             .serialize_to_path(engine, TMP_DIR, enable_quant=init_quant, weight_only_quant=weight_only_quant, skip_if_exists=False)
             .free_model())
            
        runtime_cfg_builder: AsModelRuntimeConfigBuilder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA, device_list, max_batch=8)
        runtime_cfg_builder.max_length = max_length
        runtime_cfg = runtime_cfg_builder.build()

        # install model to engine
        engine.install_model(runtime_cfg)

        if in_memory:
            model_loader.free_memory_serialize_file()
        
        # start the model inference
        engine.start_model(safe_model_name)

        self.context_len = max_length
        self.tokenizer = model_loader.tokenizer
        self.model_loader = model_loader
        self.gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
        self.engine = engine

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        temperature = params.get("temperature")
        top_k = params.get("top_k")
        top_p = params.get("top_p")
        repetition_penalty = params.get("repetition_penalty")
        presence_penalty = params.get("presence_penalty")
        max_new_tokens = params.get("max_new_tokens")
        stop_token_ids = params.get("stop_token_ids") or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        seed = params.get("seed")
        echo = params.get("echo", True)
        logprobs = params.get("logprobs")
        # not supported parameters
        frequency_penalty = params.get("frequency_penalty")
        stop = params.get("stop")
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)
        json_schema = params.get("json_schema", None)
        vocab = params.get("vocab", None)
        vocab_type = params.get("vocab_type", None)

        gen_cfg = copy.deepcopy(self.gen_cfg)
        if temperature is not None:
            gen_cfg.update({"temperature": float(temperature)})
        if top_k is not None:
            dashinfer_style_top_k = 0 if int(top_k) == -1 else int(top_k)
            gen_cfg.update({"top_k": dashinfer_style_top_k})
        if top_p is not None:
            gen_cfg.update({"top_p": float(top_p)})
        if repetition_penalty is not None:
            gen_cfg.update({"repetition_penalty": float(repetition_penalty)})
        if presence_penalty is not None:
            gen_cfg.update({"presence_penalty": float(presence_penalty)})
        if len(stop_token_ids) != 0:
            dashinfer_style_stop_token_ids = [[id] for id in set(stop_token_ids)]
            logger.info(
                f"dashinfer_style_stop_token_ids = {dashinfer_style_stop_token_ids}"
            )
            gen_cfg.update({"stop_words_ids": dashinfer_style_stop_token_ids})
        if seed is not None:
            gen_cfg.update({"seed": int(seed)})
        if logprobs is not None:
            gen_cfg.update({"logprobs": True})
            gen_cfg.update({"top_logprobs": int(logprobs)})
        if frequency_penalty is not None:
            logger.warning(
                "dashinfer worker does not support `frequency_penalty` parameter"
            )
        if stop is not None:
            logger.warning("dashinfer worker does not support `stop` parameter")
        if use_beam_search == True:
            logger.warning(
                "dashinfer worker does not support `use_beam_search` parameter"
            )
        if best_of is not None:
            logger.warning("dashinfer worker does not support `best_of` parameter")
        if json_schema is not None:
            logger.warning("dashinfer worker does not support `json_schema` parameter")
        if vocab is not None:
            logger.warning("dashinfer worker does not support `vocab` parameter")
        if vocab_type is not None:
            logger.warning("dashinfer worker does not support `vocab_type` parameter")

        logger.info(
            f"dashinfer engine helper creates request with context: {context}"
        )

        context_tokens = self.tokenizer.encode(context)
        context_len = len(context_tokens)

        # check if prompt tokens exceed the max_tokens
        max_tokens = gen_cfg["max_length"] if max_new_tokens is None else context_len + max_new_tokens
        if context_len > max_tokens:
            ret = {
                "text": f"This model's maximum generated tokens include context are {max_tokens}, However, your context resulted in {context_len} tokens",
                "error_code": ErrorCode.CONTEXT_OVERFLOW,
            }
            yield json.dumps(ret).encode() + b"\0"
            return

        gen_cfg.update({"max_length": int(max_tokens)})
        safe_model_name = params["model"].replace("/", "_")
        logger.info(
            f"start a request to model {safe_model_name} with gen_cfg = {gen_cfg}"
        )
        status, handle, queue = self.engine.start_request_ids(safe_model_name, self.model_loader, context_tokens, gen_cfg)
        if status != AsStatus.ALLSPARK_SUCCESS:
            ret = {
                "text": f"start request text failed with status: {status}",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            logger.error(f"start request to model {safe_model_name} failed")
            self.engine.release_request(safe_model_name, handle)
            yield json.dumps(ret).encode() + b"\0"
            return
        
        generated_ids = []
        while True:
            status = queue.GenerateStatus()
            if status == GenerateRequestStatus.Init:
                pass
            elif status == GenerateRequestStatus.Generating:
                elements = queue.Get()
                if elements is not None:
                    generated_ids += elements.ids_from_generate
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    ret = {
                        "text": context + generated_text if echo else generated_text,
                        "error_code": 0,
                        "usage": {
                            "prompt_tokens": context_len,
                            "completion_tokens": len(generated_ids),
                            "total_tokens": context_len + len(generated_ids),
                        },
                    }
                    yield json.dumps(ret).encode() + b"\0"
            elif status == GenerateRequestStatus.GenerateFinished:
                break
            else:
                ret = {
                    "text": f"generate failed with status: {status}",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
                yield json.dumps(ret).encode() + b"\0"
        self.engine.release_request(safe_model_name, handle)


    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())

def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = await worker.generate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://127.0.0.1:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://127.0.0.1:21001"
    )
    parser.add_argument("--model-path", type=str, default="qwen/Qwen-7B-Chat")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--in-memory", default=False, action="store_true")
    parser.add_argument("--init-quant", default=False, action="store_true")
    parser.add_argument("--weight-only-quant", default=False, action="store_true")
    parser.add_argument("--data-type", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="weights data type")
    parser.add_argument("--max-length", type=int, default=2048, help="max context length")
    parser.add_argument("-d", "--device-list", type=lambda s: list(map(int, s.split(","))), default=[0])
    

    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--revision",
        type=str,
        default="master",
        help="Modelscope model revision identifier",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
   

    args = parser.parse_args()
    worker = DashInferWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.revision,
        args.no_register,
        args.in_memory,
        args.init_quant,
        args.weight_only_quant,
        args.data_type,
        args.max_length,
        args.device_list,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

