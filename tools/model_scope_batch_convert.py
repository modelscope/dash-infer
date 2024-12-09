'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model_scope_batch_convert.py
'''
import os

import modelscope
from modelscope import snapshot_download, HubApi
from modelscope.utils.constant import DEFAULT_MODEL_REVISION

from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.prompt_utils import PromptTemplate

list_of_model = [
    #{"ms_name": "qwen/Qwen2-1.5B", "version": DEFAULT_MODEL_REVISION},
    {"ms_name": "qwen/Qwen2-7B", "version": DEFAULT_MODEL_REVISION},
    {"ms_name": "qwen/Qwen2-1.5B", "version": DEFAULT_MODEL_REVISION},
    {"ms_name": "qwen/Qwen2-0.5B-Instruct", "version": DEFAULT_MODEL_REVISION},
    {"ms_name": "qwen/Qwen2-1.5B-Instruct", "version": DEFAULT_MODEL_REVISION},
    #{"ms_name": "qwen/Qwen2-0.5B", "version": DEFAULT_MODEL_REVISION},
]


def downlaod_ms_model_and_convert(modelscope_name, ms_version, output_base_folder, ms_token=""):
    if len(ms_token) > 0:
        api = HubApi()
        api.login(ms_token)

    model_model_path = modelscope.snapshot_download(modelscope_name, ms_version)
    safe_model_name = str(modelscope_name).replace("/", "_")

    model_loader = allspark.HuggingFaceModel(model_model_path, safe_model_name, trust_remote_code=True)
    engine = allspark.Engine()
    # convert model
    model_convert_folder = os.path.join(output_base_folder, safe_model_name)
    model_loader.load_model().serialize_to_path(engine, model_convert_folder).free_model()

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA,
                                                                               [0, 1], max_batch=1)
    # like change to engine max length to a smaller value
    runtime_cfg_builder.max_length(2048)

    # like enable int8 kv-cache or int4 kv cache rather than fp16 kv-cache
    # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8)

    # or u4
    # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)
    runtime_cfg = runtime_cfg_builder.build()
    # install model to engine
    engine.install_model(runtime_cfg)
    # start the model inference
    engine.start_model(safe_model_name)

    input_list = ["你是谁？"]
    for i in range(len(input_list)):
        input_str = input_list[i]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_str}
        ]
        #input_ids = model_loader.init_tokenizer().get_tokenizer().apply_chat_template(messages)
        #input_ids = input_ids

        input_str = PromptTemplate.apply_chatml_template(input_str)



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
        # sync will wait request finish, like a sync interface, but you can async polling the queue.
        # without this call, the model result will async running, result can be fetched by queue
        # until queue status become generate finished.
        engine.sync_request(safe_model_name, handle)

        # after sync, you can fetch all the generated id by this api, this api is a block api
        # will return when there new token, or generate is finished.
        generated_elem = queue.Get()
        # after get, engine will free resource(s) and token(s), so you can only get new token by this api.

        # de-tokenize id to text
        output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_elem.ids_from_generate)
        print("---" * 20)
        print(
            f"test case: {modelscope_name} input:\n{input_str}  \n output:\n{output_text}\n")
        print(f"input token:\n {model_loader.init_tokenizer().get_tokenizer().encode(input_str)}")
        print(f"output token:\n {generated_elem.ids_from_generate}")
        print("---" * 20)
        # compare them with reference, and compare method.


if __name__ == '__main__':
    base_dir = "convert_qwen2"

    login_token = os.getenv("MS_API_KEY")

    if str(login_token) == 0:
        raise ValueError("not found MS API key: set MS_API_KEY env var")

    for entry in list_of_model:
        #try:
        downlaod_ms_model_and_convert(entry["ms_name"], entry["version"], base_dir, login_token)
        print(f"{entry['ms_name']} convert and inference success")
        #except Exception as e:
        #    print(e.with_traceback())
        #    print(f"{entry['ms_name']} convert and inference failed !!!")
