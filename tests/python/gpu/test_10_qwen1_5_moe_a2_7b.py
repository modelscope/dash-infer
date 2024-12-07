'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_10_qwen1_5_moe_a2_7b.py
'''
import os
import gc
import torch
import unittest

import modelscope

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark import AsStatus
from transformers import AutoTokenizer


class Qwen1_5_MoETestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.model_name = "qwen/Qwen1.5-MoE-A2.7B-Chat"
        print("start test for ", self.model_name)
        # download from modelscope
        print("Downloading model from modelscope...")
        self.model_path = modelscope.snapshot_download(self.model_name)
        # whether to manully input prompts or not
        self.manual_input_prompt = False

    def tearDown(self):
        gc.collect()

    def disabled_test_moe_model_serialize(self):
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

    def test_moe_model_inference(self):
        data_type = "float16"
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left"
        )

        sample_prompts = [
            "老鼠生病了可以吃老鼠药吗？",
            "我的蓝牙耳机坏了，如何预约牙医？",
            "安徽的省会是南京还是蚌埠？",
        ]

        # prepare requests
        requests = []
        for prompt in sample_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer([text], return_tensors="pt")
            request = {"input_text": prompt, "torch_input": input_ids}
            requests.append(request)

        # prepare model loader
        safe_model_name = str(self.model_name).replace("/", "_")
        model_loader = allspark.HuggingFaceModel(
            self.model_path,
            safe_model_name,
            in_memory_serialize=False,
            user_set_data_type=data_type,
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

        # prepare config
        runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(
            safe_model_name, TargetDevice.CUDA, device_ids=[0, 1], max_batch=self.batch_size
        )
        runtime_cfg_builder.max_length(2048)
        runtime_cfg_builder.prefill_cache(False)
        runtime_cfg = runtime_cfg_builder.build()

        gen_cfg_updates = {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.9,
            "seed": 1234,
            "max_length": 1024,
            "repetition_penalty": 1.05,
            "length_penalty": 1.0,
        }
        gen_cfg_builder = model_loader.create_reference_generation_config_builder(
            runtime_cfg
        )
        gen_cfg_builder.update(gen_cfg_updates)
        gen_cfg = gen_cfg_builder.build()

        # build model
        engine.build_model_from_config_struct(runtime_cfg)
        self.assertEqual(
            engine.start_model(runtime_cfg.model_name), AsStatus.ALLSPARK_SUCCESS
        )

        # start requests
        if self.manual_input_prompt == True:
            while True:
                print("***" * 20)
                prompt = input("请输入一句prompt（输入end结束）：")
                if prompt == "end":
                    break
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    input_ids = tokenizer([text], return_tensors="pt")
                    request = {"input_text": prompt, "torch_input": input_ids}
                    status, request_handle, result_queue = engine.start_request(
                        runtime_cfg.model_name,
                        {
                            "input_ids": torch.utils.dlpack.to_dlpack(
                                request["torch_input"]["input_ids"]
                            ),
                            "attention_mask": torch.utils.dlpack.to_dlpack(
                                request["torch_input"]["attention_mask"]
                            ),
                        },
                        generate_config=gen_cfg,
                    )
                    print(status, request_handle, result_queue)
                    self.assertEqual(
                        engine.sync_request(runtime_cfg.model_name, request_handle),
                        AsStatus.ALLSPARK_SUCCESS,
                    )

                    while True:
                        status = result_queue.GenerateStatus()
                        if status == allspark.GenerateRequestStatus.GenerateFinished:
                            generated_elem = result_queue.Get()
                            output_ids = generated_elem.ids_from_generate
                            out_text = tokenizer.decode(
                                output_ids, skip_special_tokens=True
                            )
                            print("in_text: " + request["input_text"])
                            print("\n")
                            print("out_text: " + out_text)
                            print("***" * 20)
                            print("\n")
                            break
                        elif (
                            status == allspark.GenerateRequestStatus.GenerateInterrupted
                        ):
                            break

                    self.assertEqual(
                        engine.release_request(runtime_cfg.model_name, request_handle),
                        AsStatus.ALLSPARK_SUCCESS,
                    )

        else:
            for request in requests:
                status, request_handle, result_queue = engine.start_request(
                    runtime_cfg.model_name,
                    {
                        "input_ids": torch.utils.dlpack.to_dlpack(
                            request["torch_input"]["input_ids"]
                        ),
                        "attention_mask": torch.utils.dlpack.to_dlpack(
                            request["torch_input"]["attention_mask"]
                        ),
                    },
                    generate_config=gen_cfg,
                )
                print(status, request_handle, result_queue)
                self.assertEqual(
                    engine.sync_request(runtime_cfg.model_name, request_handle),
                    AsStatus.ALLSPARK_SUCCESS,
                )

                while True:
                    status = result_queue.GenerateStatus()
                    if status == allspark.GenerateRequestStatus.GenerateFinished:
                        generated_elem = result_queue.Get()
                        output_ids = generated_elem.ids_from_generate
                        out_text = tokenizer.decode(
                            output_ids, skip_special_tokens=True
                        )
                        print("***" * 20)
                        print("in_text: " + request["input_text"])
                        print("\n")
                        print("out_text: " + out_text)
                        print("***" * 20)
                        print("\n")
                        break
                    elif status == allspark.GenerateRequestStatus.GenerateInterrupted:
                        break

                self.assertEqual(
                    engine.release_request(runtime_cfg.model_name, request_handle),
                    AsStatus.ALLSPARK_SUCCESS,
                )

        self.assertEqual(
            engine.stop_model(runtime_cfg.model_name), AsStatus.ALLSPARK_SUCCESS
        )


if __name__ == "__main__":
    unittest.main()
