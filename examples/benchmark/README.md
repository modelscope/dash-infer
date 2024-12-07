
# 准备工作

## Dataset

从这个git里面下载：
`git clone https://code.alibaba-inc.com/HCI/dashscope-data`

## 模型下载
可以使用本地下载好的HF格式的模型，或者使用 modelscope的ID去下载模型，如果可以访问hf，可以使用hf的的路径。 

把这个地址填到 --model_path 这个参数即可。

# Benchmark工具

## 使用数据集测试

使用 `--test_data_path` 这个参数进行选择数据集，`--test_dataset_id` 来选择数据集类型， 默认sample 100条，可以通过--test_sample_size 进行改变。

使用方法：


例如， 一个7b模型 1qps, 单卡 压测：
` python3 ./examples/benchmark/benchmark_throughput.py  --model_path=qwen/Qwen2-7B-Instruct --modelscope True  --test_qps=1 --test_dataset_path=/home/jiejing.zjj/workspace/llm_evaluation/dataset/type0_online_data.json --test_dataset_id=0 --engine_max_batch=10 --test_sample_size=10`

例如， 一个7b 模型, 双卡 最大batch size压测。

`python3 ./examples/benchmark/benchmark_throughput.py  --model_path=qwen/Qwen2-7B-Instruct --modelscope True  --test_qps=250 --test_dataset_path=/home/jiejing.zjj/workspace/llm_evaluation/dataset/type0_online_data.json --test_dataset_id=0 --engine_max_batch=380 --engine_max_length=819 --device_ids=0,1 --test_sample_size=310 --test_max_output=20`


## 使用随机数进行测试
使用随机数会进行生成固定长度的测试。
注意:  随机数据会使用输入长度 = engine_max_length - test_max_output, 输出长度： test_max_output 
`python3 ./examples/benchmark/benchmark_throughput.py  --model_path=qwen/Qwen2-7B-Instruct --modelscope True  --test_qps=250 --test_random_input --engine_max_batch=100 --engine_max_length=800 --device_ids=0,1 --test_sample_size=310 --test_max_output=400 `

