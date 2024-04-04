model_path=../outputs/
tokenizer_path=/root/.cache/modelscope/hub/qwen/Qwen-7B-Chat/
config_file=../model_config/config_qwen_v10_7b_quantize.json

# rm -rf data/mmlu/data
# rm -rf data/ceval/data
# rm -rf data/gsm8k_res.jsonl
# rm -rf data/HumanEval_res.jsonl
# rm -rf log_*.txt

# tar -xvf data/mmlu/data.tar -C data/mmlu
# unzip data/ceval/ceval-exam.zip -d data/ceval/data

python evaluate_chat_ceval.py -d data/ceval/data/ --debug --overwrite \
--model_path ${model_path} --tokenizer_path ${tokenizer_path} --config_file ${config_file} \
2>&1 | tee log_ceval.txt

python evaluate_chat_gsm8k.py -o data/gsm8k_res.jsonl \
--model_path ${model_path} --tokenizer_path ${tokenizer_path} --config_file ${config_file} \
2>&1 | tee log_gsm8k.txt

python evaluate_chat_humaneval.py -f data/HumanEval.jsonl -o data/HumanEval_res.jsonl \
--model_path ${model_path} --tokenizer_path ${tokenizer_path} --config_file ${config_file} \
2>&1 | tee log_humaneval.txt

evaluate_functional_correctness HumanEval_res.jsonl \
2>&1 | tee log_humaneval_eval.txt

python evaluate_chat_mmlu.py -d data/mmlu/data/ --debug \
--model_path ${model_path} --tokenizer_path ${tokenizer_path} --config_file ${config_file} \
2>&1 | tee log_mmlu2.txt
