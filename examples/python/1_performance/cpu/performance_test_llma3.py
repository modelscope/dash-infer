'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    performance_test_llma3.py
'''

import os
import numpy as np
import modelscope
import torch.utils.dlpack
import numpy as np
import time
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
import sys
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.prompt_utils import PromptTemplate
from dashinfer.allspark._allspark import AsStatus, GenerateRequestStatus, AsCacheMode
def check_transformers_version():
    import transformers
    required_version = "4.37.0"
    current_version = transformers.__version__

    if current_version < required_version:
        raise Exception(
            f"Transformers version {current_version} is lower than required version {required_version}. Please upgrade transformers to version {required_version}."
        )
        exit()

if __name__ == '__main__':
    check_transformers_version()
    # if use in memory serialize, change this flag to True
    in_memory = False
    init_quant= False
    weight_only_quant = True
    device_list=[0]
    fetch_output_mode = "async" # or "sync"
    modelscope_name ="LLM-Research/Meta-Llama-3-8B-Instruct"
    ms_version = DEFAULT_MODEL_REVISION
    output_base_folder="output_qwen"
    model_local_path=""
    tmp_dir = "../../model_output"


    model_local_path = modelscope.snapshot_download(modelscope_name, ms_version)
    safe_model_name = str(modelscope_name).replace("/", "_")

    model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name, user_set_data_type="float32", in_memory_serialize=in_memory, trust_remote_code=True)
    engine = allspark.Engine()

    model_convert_folder = os.path.join(output_base_folder, safe_model_name)

    if in_memory:
        (model_loader.load_model()
        .read_model_config()
        .serialize_to_memory(engine, enable_quant=init_quant, weight_only_quant=weight_only_quant)
        .export_model_diconfig(os.path.join(tmp_dir, "diconfig.yaml"))
        .free_model())
    else:
        (model_loader.load_model()
        .read_model_config()
        .serialize_to_path(engine, tmp_dir, enable_quant=init_quant, weight_only_quant=weight_only_quant,
                            skip_if_exists=False)
        .free_model())

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CPU,
                                                                            device_list, max_batch=8)
    # like change to engine max length to a smaller value
    runtime_cfg_builder.max_length(2048)

    # like enable int8 kv-cache or int4 kv cache rather than fp16 kv-cache
    # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8)

    # or u4
    # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)
    runtime_cfg = runtime_cfg_builder.build()

    # install model to engine
    engine.install_model(runtime_cfg)

    if in_memory:
        model_loader.free_memory_serialize_file()

    # start the model inference
    engine.start_model(safe_model_name)
    batch_size_list = [1, 2, 4, 8]
    output_len_list = [128]
    input_len_list = [128, 1200]
    generate_config={
    'uuid':"10000",
    'num_beams': 1,
    'num_return_sequences': 1,
    'temperature': 0.8,
    'do_sample': True,
    'early_stopping': True,
    'top_k': 1,
    'top_p': 0.8,
    }
    request_id = 0
    for output_len in output_len_list:
        for input_len in input_len_list:
            for batch_size in batch_size_list:
                sys.stdout.flush()
                in_ids = np.random.randint(low=1, high=3000, size=(1, input_len)).astype(np.int64)
                torch_dl_input = {"input_ids": torch.utils.dlpack.to_dlpack(torch.Tensor(in_ids).to(torch.int64).cpu())}
                generate_config["max_length"]=output_len + input_len
                total_request = batch_size
                pending_queue = []
                pending_handles = []
                input_len_list = []
                start = time.time()
                for i in range(total_request):
                    generate_config["uuid"]=str(request_id)
                    request_id +=1
                    status, request_handle, result_queue = engine.start_request(
                    safe_model_name,
                    torch_dl_input,
                    generate_config=generate_config)
                    input_len_list.append(in_ids[0])
                    pending_handles.append(request_handle)
                    pending_queue.append(result_queue)

                status = engine.sync_request(safe_model_name, None)
                end = time.time()
                total_input_len =0
                total_output_len =0
                for i in range(total_request):
                    result_queue = pending_queue[i]
                    out= engine.get_no_wait(safe_model_name,result_queue)
                    status = engine.get_request_status(safe_model_name,result_queue)
                    input = input_len_list[i]
                    total_input_len += len(input)
                    total_output_len += len(out)
                    status =  engine.get_request_status(safe_model_name,result_queue)
                    engine.release_request(safe_model_name,request_handle=pending_handles[i])
                print(engine.get_op_profiling_info(safe_model_name))
                total_time = end- start
                
                print("---" * 20)
                
                print(f"### batch_size: {batch_size}, output_len: {output_len}, input_len: {input_len}")
                print("total_time:",total_time)
                print("total_input_len = ", total_input_len ," total_output_len = ",total_output_len)
                print("avg_input_len = ", total_input_len/total_request ," avg_output_len = ",total_output_len/total_request)
                print("request_throughput",total_request/total_time)
                print("total_throughput:",(total_input_len + total_output_len)/total_time)
                print("output_throughput:",(total_output_len)/total_time)
    engine.stop_model(safe_model_name)
    engine.release_model(safe_model_name)