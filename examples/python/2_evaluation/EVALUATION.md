## Information

The evaluation scripts in the current directory are developed based on [Qwen github](https://github.com/QwenLM/Qwen/tree/main/eval).

commit:

```
commit 3ad0c83bb9a31fb3ccb8b70b6ec5a92caa4f7170 (HEAD -> main, origin/main, origin/HEAD)
Merge: 11e0087 a6085c2
Author: Jianhong Tu <37433392+tuhahaha@users.noreply.github.com>
Date:   Fri Mar 15 20:52:29 2024 +0800

    Merge pull request #1155 from JianxinMa/main

    bugfix streaming mode of openai_api.py
```

## CEVAL

```Shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir -p data
mkdir data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip -d data
cd ../../

# Qwen-7B-Chat (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
pip install thefuzz
python evaluate_chat_ceval.py -d data/ceval/
```

## MMLU

```Shell
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

# Qwen-7B-Chat (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
pip install thefuzz
python evaluate_chat_mmlu.py -d data/mmlu/data/
```

## HumanEval

Get the HumanEval.jsonl file from [here](https://github.com/openai/human-eval/tree/master/data)

```Shell
git clone https://github.com/openai/human-eval
pip install -e human-eval

# Qwen-7B-Chat
python evaluate_chat_humaneval.py -f HumanEval.jsonl -o HumanEval_res_chat.jsonl
evaluate_functional_correctness HumanEval_res_chat.jsonl
```

When installing package human-eval, please note its following disclaimer:

This program exists to run untrusted model-generated code. Users are strongly encouraged not to do so outside of a robust security sandbox. The execution call in execution.py is deliberately commented out to ensure users read this disclaimer before running code in a potentially unsafe manner. See the comment in execution.py for more information and instructions.

## GSM8K

```Shell
# Qwen-7B-Chat (We only provide 0-shot reproduction scripts. 5-shot results are obtained by OpenCompass (https://github.com/InternLM/opencompass).)
python evaluate_chat_gsm8k.py # zeroshot
```
