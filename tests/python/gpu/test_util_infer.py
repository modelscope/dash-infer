'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_util_infer.py
'''


import os
import unittest
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from concurrent.futures import ThreadPoolExecutor
import configparser

import modelscope

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice, RequestStatInfo
from dashinfer.allspark.engine import RoPEScaleMethod
from dashinfer.allspark.prompt_utils import PromptTemplate
from test_utils import LevenshteinCompare, CosineCompare, JaccardCompare, GenerateResultCompare
from dashinfer.allspark._allspark import AsStatus, GenerateRequestStatus, AsCacheMode


def get_auth():
    auth = None
    endpoint = None
    if os.environ.get("OSS_ACCESS_KEY_ID"):
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        endpoint = os.environ.get("OSS_ENDPOINT")
    else:
        config_path = os.path.expanduser('~/.ossutilconfig')
        if not os.path.exists(config_path):
            return None
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        if 'Credentials' not in cfg:
            return None
        conf = cfg['Credentials']
        access_id = conf.get('accessKeyID')
        access_secret = conf.get('accessKeySecret')
        if access_id is None or access_secret is None:
            return None
        endpoint = conf.get('endpoint')
        auth = oss2.Auth(access_id, access_secret)
    return auth, endpoint

def download_folder_from_oss(oss_folder_key, output_dir, max_workers=10, bucket="hci-team-private"):
    auth, endpoint = get_auth()
    
    print(f"Download model to {output_dir}", flush=True)
    bucket = oss2.Bucket(auth, endpoint, bucket)

    def download_object(object_info):
        try:
            object_name = object_info.key
            local_name = object_name[len(oss_folder_key):]
            local_file_path = os.path.join(output_dir, local_name)
            bucket.get_object_to_file(object_name, local_file_path)
        except Exception as e:
            print(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for object_info in oss2.ObjectIterator(bucket, prefix=oss_folder_key):
            executor.submit(download_object, object_info)

def func_test_model_with_reference(test_dict, engine=None, init_quant=False, weight_only_quant=False, test:unittest.TestCase=None,
                                   ms_download=True, in_memory=True, user_set_data_type="bfloat16",
                                   quant_config = None,
                                   cache_quant_mode = "16",
                                   set_engine_max_length = 2048,
                                   set_prefill_cache=False,
                                   user_runtime_config_dict = {},
                                   model_local_path="",  direct_load = False, load_format = None,  device_list=[0, 1], enable_sparsity_matmul=False) -> float:
    if load_format:
        assert direct_load == True
    tmp_dir = "model_output"
    print(test_dict)
    print(quant_config)
    print(f"kv cache bit width: {cache_quant_mode}")
    modelscope_name = test_dict["model_name"]
    compare: GenerateResultCompare = test_dict["compare"]
    threshold = test_dict["threshold"]
    gen_params_update = test_dict.get("generation_params", None)


    input_list = test_dict["input"]
    reference_list = test_dict["reference"]
    # download model
    model_model_path = ""
    if ms_download:
        model_model_path = modelscope.snapshot_download(modelscope_name)
    else:
        model_model_path = model_local_path

    safe_model_name = str(modelscope_name).replace("/", "_")

    model_loader = allspark.HuggingFaceModel(model_model_path, safe_model_name, in_memory_serialize=in_memory,
                                             user_set_data_type=user_set_data_type,
                                             trust_remote_code=True)
    if engine is None:
        engine = allspark.Engine()
    # convert model

    (model_loader.load_model(direct_load=direct_load, load_format=load_format)
     .read_model_config()
     # .set_rope_scale_method(RoPEScaleMethod.YaRN, {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"})
     .serialize(engine, model_output_dir=tmp_dir, enable_quant=init_quant, customized_quant_config=quant_config, weight_only_quant=weight_only_quant)
     .free_model())

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA,
                                                                               device_list, max_batch=1)
    # like change to engine max length to a smaller value
    runtime_cfg_builder.max_length(set_engine_max_length)
    runtime_cfg_builder.prefill_cache(set_prefill_cache)
    runtime_cfg_builder.enable_sparsity_matmul(enable_sparsity_matmul)

    runtime_cfg_builder.update_from_dict(user_runtime_config_dict)

    # like enable int8 kv-cache or int4 kv cache rather than fp16 kv-cache
    if cache_quant_mode != "16":
        if cache_quant_mode == "8":
            runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8)
        elif cache_quant_mode == "4":
            runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)

    runtime_cfg = runtime_cfg_builder.build()
    # install model to engine
    engine.install_model(runtime_cfg)

    # if in memory serialize, after install model, the serialize file can be free.
    if in_memory:
        model_loader.free_memory_serialize_file()

    # start the model inference
    engine.start_model(safe_model_name)

    for i in range(len(input_list)):
        input_str_origin = input_list[i]
        reference_text = reference_list[i]
        input_str = PromptTemplate.apply_chatml_template(input_str_origin)
        # generate a reference generate config.
        gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
        # change generate config base on this generation config, like change top_k = 1
        gen_cfg.update({"top_k": 1})
        if gen_params_update:
            gen_cfg.update(gen_params_update)

        try:

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_str_origin}
            ]
            templated_str = input_str_origin
            try:
                templated_str = model_loader.init_tokenizer().get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                print("Exception on token chat template.")
                templated_str = PromptTemplate.apply_chatml_template(input_str_origin)

            print(f"our template: {templated_str} input text: {input_str}")

            # 1. use text interface do inference.
            status, handle, queue = engine.start_request_text(safe_model_name,
                                                              model_loader,
                                                              templated_str,
                                                              gen_cfg)
            if test:
                test.assertEqual(status, AsStatus.ALLSPARK_SUCCESS)

            # sync will wait request finish, like a sync interface, but you can async polling the queue.
            # without this call, the model result will async running, result can be fetched by queue
            # until queue status become generate finished.

            engine.sync_request(safe_model_name, handle)

            # after sync, you can fetch all the generated id by this api, this api is a block api
            # will return when there new token, or generate is finished.
            generated_elem = queue.Get()
            # after get, engine will free resource(s) and token(s), so you can only get new token by this api.

            stat_dict = queue.RequestStatInfo()

            test.assertGreater(len(stat_dict.keys()), 1)

            req_stat = RequestStatInfo(stat_dict)
            print(f"request state {req_stat}, request key: {stat_dict}")

            # de-tokenize id to text
            output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_elem.ids_from_generate)
            print("---" * 20)
            print(
                f"test case: {modelscope_name} {quant_config} kv-cache: {cache_quant_mode} input:\n{input_str}  \n output text:\n{output_text}\n reference:\n{reference_text}")
            print(f"input token:\n {model_loader.init_tokenizer().get_tokenizer().encode(input_str)}")
            print(f"output token:\n {generated_elem.ids_from_generate}")
            print("---" * 20)
            # compare them with reference, and compare method.
            sim = compare.normal_similarity(output_text, reference_text, lang=test_dict["lang"])
            print(f"{modelscope_name} sim: {sim} threshold: {threshold}")
            if (test):
                test.assertGreaterEqual(sim, threshold, "similarity is not ok")

            engine.release_request(safe_model_name, handle)

            # inference method 2:  send id request.
            tokenizer = model_loader.init_tokenizer().get_tokenizer()
            encode_ids = tokenizer.encode(templated_str)

            print(encode_ids)
            if test:
                test.assertEqual(type(encode_ids), type([]))
            status2, handle2, queue2 = engine.start_request_ids(safe_model_name,
                                                                model_loader,
                                                                encode_ids,
                                                                gen_cfg)
            if test:
                test.assertEqual(AsStatus.ALLSPARK_SUCCESS, status2)

            generated_ids2 = []
            # async fetch output result.
            # looping until status is not ok
            print(f"2 request: status: {queue2.GenerateStatus()}")
            status = queue2.GenerateStatus()

            ## in following 3 status, it means tokens are generating
            while (status == GenerateRequestStatus.Init
                   or status == GenerateRequestStatus.Generating
                   or status == GenerateRequestStatus.ContextFinished):
                print(f"request: status: {queue2.GenerateStatus()}")
                elements = queue2.Get()
                if elements is not None:
                    print(f"new token: {elements.ids_from_generate}")
                    generated_ids2 += elements.ids_from_generate
                status = queue2.GenerateStatus()
                if status == GenerateRequestStatus.GenerateFinished:
                    break
                    # This means generated is finished.
                if status == GenerateRequestStatus.GenerateInterrupted:
                    break
                    # This means the GPU has no available resources; the request has been halted by the engine.
                    # The client should collect the tokens generated so far and initiate a new request later.

            if test:
                test.assertEqual(queue2.GenerateStatus(), GenerateRequestStatus.GenerateFinished)
            print(f"generated id: {queue2.GenerateStatus()}  {generated_ids2}")

            output_text2 = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids2)
            sim = compare.normal_similarity(output_text2, reference_text, lang=test_dict["lang"])

            print("---" * 20)

            print(
                f"[ids] [async] test case: {modelscope_name} quant_config:{quant_config} input:\n{input_str}  \n output:\n{output_text2}\n reference:\n{reference_text} simi:{sim}")
            print(f"input token-2:\n {model_loader.init_tokenizer().get_tokenizer().encode(input_str)}")
            print(f"output token-2:\n {generated_ids2}")
            print("---" * 20)

            print(f"{modelscope_name} sim_for_2: {sim} threshold: {threshold}")
            if test:
                test.assertGreaterEqual(sim, threshold, "similarity is out of range")

            engine.release_request(safe_model_name, handle2)
            # inference method 3:  send a multimedia request
        finally:
            engine.stop_model(safe_model_name)
            engine.release_model(safe_model_name)
            # let
        # FIXME: release module will hang.
        return sim
