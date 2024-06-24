# llama.cpp Performance Test

1. run offical docker image
```shell
docker run -it --entrypoint=/bin/bash ghcr.io/ggerganov/llama.cpp:full
```
2. download Meta-Llama-3-8B-Instruct model from modelscope
```shell
apt-get install git-lfs -y && git lfs install
git clone https://www.modelscope.cn/LLM-Research/Meta-Llama-3-8B-Instruct.git models/Meta-Llama-3-8B-Instruct
```

3. install Python dependencies
```shell
python3 -m pip install -r requirements.txt
```

4. convert the model to ggml FP16 format
```shell
python3 convert-hf-to-gguf.py models/Meta-Llama-3-8B-Instruct
```
5. quantize the model to 8-bits
```shell
./llama-quantize ./models/Meta-Llama-3-8B-Instruct/ggml-model-f16.gguf ./models/Meta-Llama-3-8B-Instruct/ggml-model-Q8_0.gguf Q8_0
```

6. do latency benchmark
- fp16 format
```shell
./llama-bench -m ./models/Meta-Llama-3-8B-Instruct/ggml-model-f16.gguf -t 48 --numa isolate -fa 1 -r 1  -p 0 -n 0 -pg 1024,1 -pg 8196,1 -pg 16384,1 -pg 32768,1 -o csv
```
- int8 format
```shell
./llama-bench -m ./models/Meta-Llama-3-8B-Instruct/ggml-model-Q8_0.gguf -t 48 --numa isolate -fa 1 -r 1  -p 0 -n 0 -pg 1024,1 -pg 8196,1 -pg 16384,1 -pg 32768,1 -o csv
```

7. do throughput benchmark
- fp16 format
```shell
for bs in 1 2 4 8;
do
  ./llama-parallel -m ./models/Meta-Llama-3-8B-Instruct/ggml-model-f16.gguf -t 48 --numa isolate -fa  -c 1024 -n 128 -np $bs -ns $bs
done
```
- int8 format
```shell
for bs in 1 2 4 8;
do
  ./llama-parallel -m ./models/Meta-Llama-3-8B-Instruct/ggml-model-Q8_0.gguf -t 48 --numa isolate -fa  -c 1024 -n 128 -np $bs -ns $bs
done
```
