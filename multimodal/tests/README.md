# DashInfer-VLM Benchmark

This document presents performance data and reproduction steps for dashinfer-vlm, along with methods for comparing it to other open-source multimodal frameworks.

## Evaluation Method
### Download dataset
You need prepare conversations and images to run the model benchmarks. Here are public datasets from [OpenGVLab/InternVL-Chat-V1-2-SFT-Data](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data). The api benchmark will combine prompts and images into requests, then send them to the API endpoint using the OpenAI client.
- [docvqa_train_10k.jsonl](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data/resolve/main/opensource/docvqa_train_10k.jsonl)
- [share_textvqa.zip](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data/resolve/main/data/share_textvqa.zip)

### Launch Server
**vLLM:**
```
vllm serve qwen/Qwen2-VL-2B-Instruct/ --allowed-local-media-path <image_folder> --limit-mm-per-prompt image=10
```
_Note: to make sure the same output token length, we set `ignore_eos=True` in `vllm/sampling_params.py`_

**dashinfer-vlm** (set `VLM_BENCHMARK=1` to specify generated token lengths, i.e. ignore_eos=True):
```
VLM_BENCHMARK=1 dashinfer_vlm_serve --model qwen/Qwen2-VL-2B-Instruct --host 127.0.0.1
```
You could enable prefix and fp8 to accelerate inference. See full options using `dashinfer_vlm_serve -h`.

### Run Benchmark
Using OpenAI client:
```
python benchmark_openai_api.py --prompt-file <json_file> \
--image-folder <image_folder> \
--req-nums 100 \
--batch-size 32 \
--image-nums-mean 3 \
--image-nums-range 1 \
--response-mean 120 \
--response-len-range 64
```

The above command generates 100 requests from prompt_file and images_folder. Each request contains 3 $\pm$ 1 images and outputs 120 $\pm$ 64 tokens. The requests are sent to the server in parallel using 8 threads.

 To benchmark multi-turn conversations, add `--multi-turn <num>` in above command.


## Model Performance
### Qwen2-VL

The steps to reproduce the performance of Qwen2-VL 2B/7B:
1. Launch api serve as the [Evaluation Method](#evaluation-method) section.
2. Benchmark command:
```
python benchmark_openai_api.py --prompt-file docvqa_train_10k.jsonl \
--image-folder `pwd`/share_textvqa \
--req-nums 100 \
--image-nums-mean 3 \
--image-nums-range 1 \
--response-len-range 64 \
--response-mean 120 \
--batch-size <1 | 32> \
--multi-turn <0 | 2>
```

#### Qwen/Qwen2-VL-2B-Instruct on A100

| Optimizations  | #Device | Dtype | Prompt Tokens (avg) | Output Tokens (avg) | # Previous Conversations |  #Concurrent | vLLM (0.6.4) QPS | DashInfer-VLM QPS | Speedup (dashinfer-vlm/vllm)
| :------: | :----: | :------: | :---- | :----: | :----: | :----: | :----: | :----: | :----: |
|  -   | 1 | FP16   | 2807 | 120 | 0 | 1 | 0.80 | 0.78 | 0.98
|  -   | 1 | FP16   | 2807 | 120 | 0 | 32 | 2.29 | 3.85 | 1.68
|  -   | 1 | FP16   | 2953 | 120 | 2 | 32 | 2.21 | 6.05 | 2.73
|  enable_prefix_cache   | 1 | FP16   | 2953 | 120 | 2 | 32 | x | 7.98 | x

#### Qwen/Qwen2-VL-7B-Instruct on A100

| Optimizations  | #Device | Dtype | Prompt Tokens (avg) | Output Tokens (avg) | # Previous Conversations |  #Concurrent | vLLM (0.6.4) QPS | DashInfer-VLM QPS | Speedup (dashinfer-vlm/vllm)
| :------: | :----: | :------: | :---- | :----: | :----: | :----: | :----: | :----: | :----: |
|  -   | 1 | FP16   | 2807 | 120 | 0 | 1 | 0.47 | 0.44 | 0.93
|  -   | 1 | FP16   | 2807 | 120 | 0 | 32 | 1.63 | 2.20 | 1.35
|  -   | 1 | FP16   | 2953 | 120 | 2 | 32 | 1.50 | 2.84 | 1.89
|  enable_prefix_cache   | 1 | FP16   | 2953 | 120 | 2 | 32 | x | 4.68 | x

#### Qwen/Qwen2-VL-72B-Instruct on 4xA100

| Optimizations  | #Device | Dtype | Prompt Tokens (avg) | Output Tokens (avg) | # Previous Conversations |  #Concurrent | vLLM (0.6.4) QPS | DashInfer-VLM QPS | Speedup (dashinfer-vlm/vllm)
| :------: | :----: | :------: | :---- | :----: | :----: | :----: | :----: | :----: | :----: |
|  -   | 1 | FP16   | 2807 | 120 | 0 | 1 | 0.2 | 0.15 | 0.75
|  -   | 1 | FP16   | 2807 | 120 | 0 | 32 | 0.79 | 0.99 | 1.26
|  -   | 1 | FP16   | 2953 | 120 | 2 | 32 | 0.82 | 1.12 | 1.36
|  enable_prefix_cache   | 1 | FP16   | 2953 | 120 | 2 | 32 | x | 1.99 | x
