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

default_system = 'You are a helpful assistant.'

from dashinfer.helper import EngineHelper, ConfigManager

log_path = "outputs_gradio_demo_log"
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file = log_path + "/log_" + time.strftime("%Y%m%d_%H-%M-%S",
                                              time.localtime()) + ".jsonl"
log_file_lock = threading.Lock()

config_file = "../model_config/config_qwen_v10_1_8b.json"
config = ConfigManager.get_config_from_json(config_file)

# from huggingface_hub import snapshot_download
# hf_model_path = snapshot_download(
#     repo_id="Qwen/Qwen-1_8B-Chat",
#     ignore_patterns=[r'.+\.bin$', r'.+\.safetensors$'])

from modelscope import snapshot_download

hf_model_path = snapshot_download(
    "qwen/Qwen-1_8B-Chat",
    revision="v1.0.0",
    ignore_file_pattern=[r'.+\.bin$', r'.+\.safetensors$'])
print(f"hf_model_path: {hf_model_path}")

cmd = f"pip show dashinfer | grep 'Location' | cut -d ' ' -f 2"
package_location = subprocess.run(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  shell=True,
                                  text=True)
package_location = package_location.stdout.strip()
os.environ["AS_DAEMON_PATH"] = package_location + "/dashinfer/allspark/bin"
os.environ["AS_NUMA_NUM"] = str(len(config["device_ids"]))
os.environ["AS_NUMA_OFFSET"] = str(config["device_ids"][0])

engine_helper = EngineHelper(config)
engine_helper.verbose = True
engine_helper.init_tokenizer(hf_model_path)

if engine_helper.check_model_exist() == False:
    print("\nPlease run basic_example_qwen_v10.py first.\n")
    raise ValueError()

engine_helper.init_engine()
engine_max_batch = engine_helper.engine_config["engine_max_batch"]

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

    gen_cfg = copy.deepcopy(engine_helper.default_gen_cfg)
    gen_cfg["seed"] = random.randint(0, 10000)

    request_list = engine_helper.create_request([prompt], [gen_cfg])

    request = request_list[0]
    gen = engine_helper.process_one_request_stream(request)
    for response in gen:
        role = Role.ASSISTANT
        system, history = messages_to_history(messages + [{'role': role, 'content': response}])
        yield '', history, system

    json_str = engine_helper.convert_request_to_jsonstr(request)
    with open(log_file, "a") as file:
        log_file_lock.acquire()
        try:
            file.write(json_str)
            file.write("\n")
        finally:
            log_file_lock.release()


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
