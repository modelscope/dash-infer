'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_11_qwen2_json.py
'''
import os
import gc
import torch
import unittest
from concurrent.futures import ThreadPoolExecutor
import modelscope
import time
import sys
import json
from typing import Any, Optional, List
from dataclasses import field

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark import AsStatus
from transformers import AutoTokenizer

schemas = []
# schemas.append(r'''
#     {
#         "properties": {
#             "公司名称": {
#                 "type": "string"
#             },
#             "founding year": {
#                 "type": "integer"
#             },
#             "founding person": {
#                 "type": "string"
#             },
#             "founding city": {
#                 "type": "string"
#             },
#             "employees": {
#                 "type": "integer"
#             }
#         },
#         "required": [
#             "公司名称",
#             "founding year",
#             "founding person",
#             "founding city",
#             "employees"
#         ],
#         "type": "object"
#     }
# ''')
schemas.append(r'''
    {
        "title": "Complex JSON Schema",
        "description": "A very complex JSON schema with nested structures and multiple constraints.",
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 50
                    },
                    "age": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 120
                    },
                    "email": {
                        "type": "string",
                        "format": "email"
                    },
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {
                                "type": "string"
                            },
                            "city": {
                                "type": "string"
                            },
                            "state": {
                                "type": "string"
                            },
                            "postalCode": {
                                "type": "string",
                                "pattern": "^\\d{5}(?:-\\d{4})?$"
                            }
                        },
                        "required": ["street", "city", "state", "postalCode"]
                    }
                }
            },
            "orders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "items"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "pattern": "^[A-Z]{3}\\d{4}$"
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["product", "quantity"],
                                "properties": {
                                    "product": {
                                        "type": "string"
                                    },
                                    "quantity": {
                                        "type": "integer",
                                        "minimum": 1
                                    },
                                    "price": {
                                        "type": "number",
                                        "minimum": 0
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "preferences": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["en", "es", "fr", "de", "it"]
                    },
                    "notifications": {
                        "type": "boolean"
                    },
                    "marketing": {
                        "type": "boolean"
                    }
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "created": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "lastUpdated": {
                        "type": "string",
                        "format": "date-time"
                    }
                }
            }
        },
        "required": ["user", "orders"],
        "additionalProperties": false
    }
''')
schemas.append(r'''
    {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "null"}, 
                    {"type": "number"}, 
                    {"type": "string"}, 
                    {"type": "boolean"},
                    {"type": "object"}, 
                    {"type": "array"}
                ]
            }
        },
        "required": ["value"]
    }
''')
schemas.append(r'''
{
    "type": "object",
    "properties": {
        "productName": {
            "type": "string"
        },
        "productType": {
            "type": "string",
            "enum": ["Electronics", "Books", "Clothing", "Home & Kitchen"]
        },
        "price": {
            "type": "number",
            "minimum": 0
        },
        "placeOfProduction": {
            "type": "string",
            "enum": ["Thailand", "Japan", "Vietnam", "China", "Indonesia"]
        },
        "monthOfProduction": {
            "type": "string",
            "enum": ["05", "06", "07", "08", "09", "10"]
        },
        "yearOfProduction": {
            "type": "string",
            "enum": ["2020", "2021", "2022", "2023", "2024"]
        }
    },
    "required": ["productName", "productType", "price", "placeOfProduction", "monthOfProduction", "yearOfProduction"]
}
''')

class Request:
    id: int = -1
    input_text: Optional[str] = None

    # torch tensors:
    input_ids = None
    in_tokens = None
    output_ids = None
    output_tokens = None

    json_schema: Optional[str] = None
    output_text: Optional[str] = None

    status: Optional[int] = None
    gen_cfg: Optional[dict] = None
    start_time = None
    end_time = None
    handle: Any = field(default=None)
    queue: Any = field(default=None)

class Qwen2_TestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.model_name = "qwen/Qwen2-7B-Instruct"
        print("start test for ", self.model_name)
        # download from modelscope
        print("Downloading model from modelscope...")
        self.model_path = modelscope.snapshot_download(self.model_name)

    def tearDown(self):
        gc.collect()

    def print_output(self, request):
        msg = "***********************************\n"
        msg += f"* Answer for Request {request.id}\n"
        msg += "***********************************\n"
        msg += f"** encoded input, len: {request.in_tokens} **\n{request.input_ids}\n\n"
        msg += f"** encoded output, len: {request.output_tokens} **\n{request.output_ids}\n\n"
        msg += f"** text input **\n{request.input_text}\n\n"
        msg += f"** text output **\n{request.output_text}\n\n"
        elapsed_time = request.end_time - request.start_time
        msg += f"** elapsed time **\n{elapsed_time}\n\n"
        print(msg)


    def disabled_test_model_serialize(self):
        # prepare model loader
        safe_model_name = str(self.model_name).replace("/", "_")
        model_loader = allspark.HuggingFaceModel(
            self.model_path,
            safe_model_name,
            in_memory_serialize=False,
            user_set_data_type="float16",
            trust_remote_code=True,
        )

        engine = allspark.Engine()

        # load and serialize model
        model_loader.load_model(direct_load=False, load_format="auto")
        model_loader.serialize_to_path(
            engine,
            model_output_dir=self.model_path,
            enable_quant=False,
            weight_only_quant=False,
            skip_if_exists=True,
        )

    def process_one_request(self, request):
        output_ids = []
        gen_cfg = self.gen_cfg_builder.build()
        gen_cfg["response_format"] = {"type": "json_object", "json_schema": request.json_schema}
        #gen_cfg["response_format"] = {"type": "json_object"}
        request.start_time = time.time()
        status, request_handle, result_queue = self.engine.start_request(
            self.runtime_cfg.model_name,
            {
                "input_ids": torch.utils.dlpack.to_dlpack(request.input_ids),
            },
            generate_config=gen_cfg,
        )

        if status != allspark.AsStatus.ALLSPARK_SUCCESS:
            request.valid = False
            print("[Error] Request failed to start!")
            sys.exit(1)

        request.valid = True
        request.handle = request_handle
        request.queue = result_queue
        request.status = int(status)

        while True:
            # status = self.engine.get_request_status(self.model_name, request.queue)
            status = request.queue.GenerateStatus()
            request.status = int(status)

            if status == allspark.GenerateRequestStatus.Init:
                pass
            elif status == allspark.GenerateRequestStatus.Generating or status == allspark.GenerateRequestStatus.GenerateFinished:
                if status == allspark.GenerateRequestStatus.GenerateFinished:
                    request.end_time = time.time()
                # new_ids = self.engine.get_wait(self.model_name, request.queue)
                generated_elem = request.queue.Get()
                if generated_elem is not None:
                    new_ids = generated_elem.ids_from_generate
                    if (len(new_ids) > 0):
                        output_ids.extend(new_ids)

                    request.output_ids = output_ids
                    request.output_tokens = len(output_ids)
                    request.output_text = self.tokenizer.decode(request.output_ids, skip_special_tokens=True)

                if status == allspark.GenerateRequestStatus.GenerateFinished:
                    # jsos.loads will throw exception if request.output_text is not a valid JSON string
                    json_str = json.loads(request.output_text)
                    break
            elif status == allspark.GenerateRequestStatus.GenerateInterrupted:
                request.valid = False
                print("[Error] Request interrupted!")
                break
            else:
                request.valid = False
                print(f"[Error] Unexpected status: {status}")
                break

        self.engine.release_request(self.runtime_cfg.model_name, request_handle=request.handle)

    def run_allspark_continuous_batch(self, request_list):
        def done_callback(future):
            request = future.argument
            future.result()
            self.print_output(request)

        # create a threadpool
        executor = ThreadPoolExecutor(max_workers=self.batch_size)

        try:
            # submit all tasks to the threadpool
            futures = []
            for request in request_list:
                future = executor.submit(self.process_one_request, request)
                future.argument = request
                future.add_done_callback(done_callback)
                futures.append(future)
        finally:
            executor.shutdown(wait=True)
    
    def run_allspark_no_batching(self, request_list):
        for request in request_list:
            self.process_one_request(request)
            self.print_output(request)

    def test_model_generate(self):
        data_type = "float16"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left"
        )

        sample_prompts = [
            "小明在京东买了手机，他的相关信息",
            "介绍一下杭州",
            "使用JSON格式填写一部2024年7月在中国生产的价格为5999的iPhone 16的信息",
        ]

        # prepare requests
        requests = []
        req_id = 0
        for idx, prompt in enumerate(sample_prompts):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            torch_input = self.tokenizer([text], return_tensors="pt")
            input_ids = torch_input["input_ids"]
            request = Request()
            request.input_text = prompt
            request.input_ids = input_ids
            request.id = req_id
            request.in_tokens = input_ids.shape[-1]
            request.json_schema = schemas[idx]
            requests.append(request)
            req_id += 1

        # prepare model loader
        safe_model_name = str(self.model_name).replace("/", "_")
        model_loader = allspark.HuggingFaceModel(
            self.model_path,
            safe_model_name,
            in_memory_serialize=False,
            user_set_data_type=data_type,
            trust_remote_code=True,
        )

        self.engine = allspark.Engine()

        # load and serialize model
        model_loader.load_model(direct_load=False, load_format="auto")
        model_loader.serialize_to_path(
            self.engine,
            model_output_dir=self.model_path,
            enable_quant=False,
            weight_only_quant=False,
            skip_if_exists=True,
        )

        # prepare config
        runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(
            safe_model_name, TargetDevice.CUDA, device_ids=[0], max_batch=self.batch_size
        )
        runtime_cfg_builder.max_length(10240)
        runtime_cfg_builder.prefill_cache(False)
        self.runtime_cfg = runtime_cfg_builder.build()

        gen_cfg_updates = {
            "temperature": 0.1,
            "top_k": 1,
            "top_p": 0.1,
            #"seed": 1234,
            "max_length": 10240,
            #"repetition_penalty": 1.05,
            #"length_penalty": 1.0,
        }
        self.gen_cfg_builder = model_loader.create_reference_generation_config_builder(
            self.runtime_cfg
        )
        self.gen_cfg_builder.update(gen_cfg_updates)

        # build model
        self.engine.build_model_from_config_struct(self.runtime_cfg)
        self.assertEqual(
            self.engine.start_model(self.runtime_cfg.model_name), AsStatus.ALLSPARK_SUCCESS
        )

        self.run_allspark_continuous_batch(requests)
        # self.run_allspark_no_batching(requests)
        
        self.assertEqual(
            self.engine.stop_model(self.runtime_cfg.model_name), AsStatus.ALLSPARK_SUCCESS
        )


if __name__ == "__main__":
    unittest.main()
