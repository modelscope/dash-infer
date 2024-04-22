import os
import re
import sys
import copy
import subprocess
import textwrap
import argparse
from pathlib import Path
import tqdm
import jsonlines

from dashinfer.helper import EngineHelper

"""
Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

python evaluate_chat_humaneval.py -f HumanEval.jsonl -o HumanEval_res.jsonl
git clone https://github.com/openai/human-eval
pip install -e human-eval
evaluate_functional_correctness HumanEval_res.jsonl
"""


def extract_code(text, entry_point):
    # 正则表达式匹配代码块
    code_block_pattern = re.compile(
        rf"```(?:[Pp]ython\n)?.*?def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL
    )
    code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            rf"def\s+{entry_point}.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)
    if code_block is None:
        code_block_pattern = re.compile(
            r"def.*?:\n(.*?)(?:\n(?!\n*(?:  |\t))|$)", re.DOTALL
        )
        code_block = code_block_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    # if no code block is found, assume the LM is simply filling the code
    return textwrap.indent(text, " " * 4)


def generate_sample(engine_helper, question, entry_point):
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" \
             + question + "<|im_end|>\n<|im_start|>assistant\n"
    gen_cfg = copy.deepcopy(engine_helper.default_gen_cfg)
    gen_cfg["max_length"] = 1000
    gen_cfg["top_k"] = 1 # use greedy decoding
    gen_cfg["repetition_penalty"] = 1.0 # disable repetition penalty
    request_list = engine_helper.create_request([prompt], [gen_cfg])
    request = request_list[0]
    engine_helper.process_one_request(request)
    response = request.out_text

    print(f"### question: {question}")
    print(f"### response: {response}")
    answer = extract_code(response, entry_point)
    print(f"### answer: {answer}")
    print("======================")
    return answer, response

def download_model(model_id, revision, source="modelscope"):
    print(f"Downloading model {model_id} (revision: {revision}) from {source}")
    if source == "modelscope":
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id, revision=revision)
    elif source == "huggingface":
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_id)
    else:
        raise ValueError("Unknown source")

    print(f"Save model to path {model_dir}")

    return model_dir

def prepare(args):
    config_file = args.config_file
    config = EngineHelper.get_config_from_json(config_file)

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
    config["model_path"] = args.model_path

    ## download model from modelscope
    original_model = {
        "source": "modelscope",
        "model_id": "qwen/Qwen-7B-Chat",
        "revision": "v1.1.9",
        "model_path": ""
    }
    original_model["model_path"] = download_model(original_model["model_id"],
                                                  original_model["revision"],
                                                  original_model["source"])

    ## init EngineHelper class
    engine_helper = EngineHelper(config)
    engine_helper.verbose = True
    engine_helper.init_tokenizer(original_model["model_path"])
    engine_helper.init_torch_model(original_model["model_path"])

    ## convert huggingface model to dashinfer model
    ## only one conversion is required
    if engine_helper.check_model_exist() == False:
        engine_helper.convert_model(original_model["model_path"])

    ## inference
    engine_helper.init_engine()

    return engine_helper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument('--model_path', type=str, default='~/dashinfer_models/')
    parser.add_argument('--config_file', type=str, default='../model_config/config_qwen_v10_7b_quantize.json')

    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default=None,
        help="data path to HumanEval.jsonl",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl"
    )

    args = parser.parse_args()
    engine_helper = prepare(args)

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            # use humanevalpack prompt
            signature = re.search(
                rf"def\s+({jobj['entry_point']}.*?):\s*\n", jobj["prompt"]
            ).group(1)
            description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", jobj["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
            prompt = (
                f"Write a Python function `{signature}` to solve the following problem:\n"
                f"{description}\n"
                f"{jobj['prompt']}"
            )

            task_id = jobj["task_id"]
            answer, response = generate_sample(
                engine_helper, prompt, jobj["entry_point"]
            )
            gen_jobjs = {"task_id": task_id, "completion": answer, "response": response}
            output.write(gen_jobjs)
    f_output.close()
