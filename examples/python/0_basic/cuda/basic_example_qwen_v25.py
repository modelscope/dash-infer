'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    basic_example_qwen_v25.py
'''

import os

import modelscope
from modelscope.utils.constant import DEFAULT_MODEL_REVISION

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
    device_list=[0,1]
    fetch_output_mode = "async" # or "sync"
    modelscope_name ="qwen/Qwen2.5-7B-Instruct"
    ms_version = DEFAULT_MODEL_REVISION
    output_base_folder="output_qwen"
    model_local_path=""
    tmp_dir = "../../model_output"


    model_local_path = modelscope.snapshot_download(modelscope_name, ms_version)
    safe_model_name = str(modelscope_name).replace("/", "_")

    model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name, user_set_data_type="bfloat16", in_memory_serialize=in_memory, trust_remote_code=True)
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

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA,
                                                                            device_list, max_batch=8)
    # like change to engine max length to a smaller value
    runtime_cfg_builder.max_length(256)

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

    input_list = ["你是谁？", "How to protect our planet and build a green future?"]
    for i in range(len(input_list)):
        input_str = input_list[i]
        input_str = PromptTemplate.apply_chatml_template(input_str)
        print("input_str = ", input_str)
        input_token  = model_loader.init_tokenizer().get_tokenizer().encode(input_str)
        print("input_token = ",input_token)
        # generate a reference generate config.
        gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
        # change generate config base on this generation config, like change top_k = 1
        gen_cfg.update({"top_k": 1})
        gen_cfg.update({"repetition_penalty": 1.1})
        #gen_cfg.update({"eos_token_id", 151645})
        status, handle, queue = engine.start_request_text(safe_model_name,
                                                        model_loader,
                                                        input_str,
                                                        gen_cfg)

        generated_ids = []
        if fetch_output_mode == "sync":
            # sync will wait request finish, like a sync interface, but you can async polling the queue.
            # without this call, the model result will async running, result can be fetched by queue
            # until queue status become generate finished.
            engine.sync_request(safe_model_name, handle)

            # after sync, you can fetch all the generated id by this api, this api is a block api
            # will return when there new token, or generate is finished.
            generated_elem = queue.Get()
            # after get, engine will free resource(s) and token(s), so you can only get new token by this api.
            generated_ids += generated_elem.ids_from_generate
        else:
            status = queue.GenerateStatus()

            ## in following 3 status, it means tokens are generating
            while (status == GenerateRequestStatus.Init
                or status == GenerateRequestStatus.Generating
                or status == GenerateRequestStatus.ContextFinished):
                print(f"2 request: status: {queue.GenerateStatus()}")
                elements = queue.Get()
                if elements is not None:
                    print(f"new token: {elements.ids_from_generate}")
                    generated_ids += elements.ids_from_generate
                status = queue.GenerateStatus()
                if status == GenerateRequestStatus.GenerateFinished:
                    break
                    # This means generated is finished.
                if status == GenerateRequestStatus.GenerateInterrupted:
                    break
                    # This means the GPU has no available resources; the request has been halted by the engine.
                    # The client should collect the tokens generated so far and initiate a new request later.




        # de-tokenize id to text
        output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
        print("---" * 20)
        print(
            f"test case: {modelscope_name} input:\n{input_str}  \n output:\n{output_text}\n")
        print(f"input token:\n {model_loader.init_tokenizer().get_tokenizer().encode(input_str)}")
        print(f"output token:\n {generated_ids}")

        engine.release_request(safe_model_name, handle)

    engine.stop_model(safe_model_name)
    engine.release_model(safe_model_name)