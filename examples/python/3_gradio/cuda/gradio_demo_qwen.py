'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    gradio_demo_qwen.py
'''
import os
import sys
import copy
import time
import random
import threading
import subprocess
import gradio as gr
from typing import List, Optional, Tuple, Dict
import modelscope
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from dashinfer import allspark
from dashinfer.allspark.engine import TargetDevice
from dashinfer.allspark.prompt_utils import PromptTemplate
from dashinfer.allspark._allspark import AsStatus, GenerateRequestStatus, AsCacheMode
default_system = 'You are a helpful assistant.'


log_path = "outputs_gradio_demo_log"
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file = log_path + "/log_" + time.strftime("%Y%m%d_%H-%M-%S",
                                              time.localtime()) + ".jsonl"
log_file_lock = threading.Lock()

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
# from huggingface_hub import snapshot_download
# hf_model_path = snapshot_download(
#     repo_id="Qwen/Qwen-1_8B-Chat",
#     ignore_patterns=[r'.+\.bin$', r'.+\.safetensors$'])

from modelscope import snapshot_download

cmd = f"pip show dashinfer | grep 'Location' | cut -d ' ' -f 2"
package_location = subprocess.run(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  shell=True,
                                  text=True)
package_location = package_location.stdout.strip()
os.environ["AS_DAEMON_PATH"] = package_location + "/dashinfer/allspark/bin"
engine_max_batch = 8

###################################################

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


class Role:
    USER = 'user'
    SYSTEM = 'system'
    BOT = 'bot'
    ASSISTANT = 'assistant'
    ATTACHMENT = 'attachment'


def clear_session() -> History:
    return '', []


def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': Role.SYSTEM, 'content': system}]
    for h in history:
        messages.append({'role': Role.USER, 'content': h[0]})
        messages.append({'role': Role.ASSISTANT, 'content': h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == Role.SYSTEM
    system = messages[0]['content']
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([q['content'], r['content']])
    return system, history


def message_to_prompt(messages: Messages) -> str:
    prompt = ""
    for item in messages:
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        prompt += f"\n{im_start}{item['role']}\n{item['content']}{im_end}"
    prompt += f"\n{im_start}assistant\n"
    return prompt


def model_chat(query: Optional[str], history: Optional[History],
               system: str) -> Tuple[str, str, History]:
    if query is None:
        query = ''
    if history is None:
        history = []

    messages = history_to_messages(history, system)
    messages.append({'role': Role.USER, 'content': query})
    prompt = message_to_prompt(messages)
    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    input_str = prompt
    status, handle, queue = engine.start_request_text(safe_model_name,
                                                        model_loader,
                                                        input_str,
                                                        gen_cfg)
    engine.sync_request(safe_model_name, handle)
    generated_ids = []
    generated_elem = queue.Get()
    # after get, engine will free resource(s) and token(s), so you can only get new token by this api.
    generated_ids += generated_elem.ids_from_generate
    response = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
    # request_list = engine_helper.create_request([prompt], [gen_cfg])

    # request = request_list[0]
    # gen = engine_helper.process_one_request_stream(request)
    # for response in gen:
    role = Role.ASSISTANT
    system, history = messages_to_history(messages + [{'role': role, 'content': response}])
    yield '', history, system

    # json_str = engine_helper.convert_request_to_jsonstr(request)
    # with open(log_file, "a") as file:
    #     log_file_lock.acquire()
    #     try:
    #         file.write(json_str)
    #         file.write("\n")
    #     finally:
    #         log_file_lock.release()


###################################################

with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>Qwen-1_8B-Chat Bot👾</center>""")
    gr.Markdown("""<center><font size=4>Qwen-1_8B-Chat is the 1.8-billion parameter chat model of the Qwen series.</center>""")

    with gr.Row():
        with gr.Column(scale=3):
            system_input = gr.Textbox(value=default_system,
                                      lines=1,
                                      label='System')
        with gr.Column(scale=1):
            modify_system = gr.Button("🛠️ Set system prompt and clear history.", scale=2)
        system_state = gr.Textbox(value=default_system, visible=False)
    chatbot = gr.Chatbot(label='Qwen-1_8B-Chat')
    textbox = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 Clear history")
        sumbit = gr.Button("🚀 Send")

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_state],
                 outputs=[textbox, chatbot, system_input],
                 concurrency_limit=engine_max_batch)
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot],
                        concurrency_limit=engine_max_batch)
    modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot],
                        concurrency_limit=engine_max_batch)

# demo.queue(api_open=False).launch(height=800, share=True, server_name="127.0.0.1", server_port=7860)
# demo.queue(api_open=False).launch(root_path="/gradio-demo/", height=800, share=False, server_name="127.0.0.1", server_port=7860)
demo.queue(api_open=False).launch(height=800,
                                  share=False,
                                  server_name="127.0.0.1",
                                  server_port=7860)
