'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    tokenizer_tool_for_qwen.py
'''
from pyhie import allspark

import modelscope
import argparse

def run_model(model_name, ids, text, download_by_modelscope):
    in_memory = True
    # 示例函数，用来运行模型
    print(f"Running model: {model_name} with ids: {ids}")
    safe_model_name = str(model_name).replace("/", "_")
    # 实际上你需要在这里实现你的模型逻辑
    if download_by_modelscope:
        model_name = modelscope.snapshot_download(model_name)
        # replace mmodel with path.

    model_loader = allspark.HuggingFaceModel(model_name, safe_model_name, in_memory_serialize=in_memory,
                                             trust_remote_code=True)

    tokenizer = model_loader.load_model().read_model_config().init_tokenizer().get_tokenizer()

    if ids:
        ids = args.ids.split(',')  # 将id分割成列表
        ids = list(map(int, ids))
        print(f"ids: {ids}")
        print(f"decoded text: \n {tokenizer.decode(ids)} \ninput: {ids}")
    if text:
        print(f"encode ids: \n{tokenizer.encode(text)} \n input text: {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model with specified parameters.')

    parser.add_argument('--model_path', type=str, required=False, help='The name of the model to run', default="qwen/Qwen2-0.5B-Instruct")
    parser.add_argument('--ids', type=str, required=False, help='The ids to process (comma-separated), like 1,2,3,4')
    parser.add_argument('--text', type=str, required=False, help='The text to process')
    parser.add_argument("--modelscope", type=bool, required=False, default=True, help="use modelscope download model")

    args = parser.parse_args()

    model_name = args.model_path
    text = args.text
    ids = args.ids
    print(f"ids: {ids} text:{text}")
    if not text and not ids:
        print("ids or text should provided.")
        exit(1)

    run_model(model_name, ids, text, args.modelscope)
