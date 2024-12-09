import os
import sys
import json
import re
import copy
import subprocess
from pathlib import Path
import argparse
import requests
import math
import numpy as np
import tqdm
from datasets import load_from_disk, load_dataset

from dashinfer.helper import EngineHelper, ConfigManager

"""
python eval/evaluate_chat_gsm8k.py [--use-fewshot]
"""

INVALID_ANS = "[invalid]"

def doc_to_text(doc, use_fewshot):
    if use_fewshot:
        context = (
            "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\n"
            "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\n"
            "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\n"
            "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n\n"
            "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\n"
            "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n\n"
            "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\n"
            "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n\n"
            f"Question: {doc['question']}\nLet's think step by step"
        )
    else:
        context = doc["question"]
    return context

def generate_sample(engine_helper, question):
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
    print("=============")
    return response

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit

def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))

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
    config = ConfigManager.get_config_from_json(config_file)

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
    config["convert_config"]["do_dynamic_quantize_convert"] = True

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
    parser.add_argument('--config_file', type=str, default='../model_config/config_qwen_v10_7b.json')

    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )
    parser.add_argument("--use-fewshot", action="store_true")

    args = parser.parse_args()
    engine_helper = prepare(args)

    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)  # or:
    else:
        dataset = load_dataset("gsm8k", "main")

    test = dataset["test"]

    f_output = open(args.sample_output_file, "w", encoding="utf-8")
    tot_length = test.num_rows
    acc_res = []
    for doc in tqdm.tqdm(test):
        context = doc_to_text(doc, args.use_fewshot)
        completion = generate_sample(engine_helper, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()
        acc_res.append(acc)

    f_output.close()
    print("4-shot Acc: " if args.use_fewshot else "Zero-shot Acc", np.mean(acc_res))
