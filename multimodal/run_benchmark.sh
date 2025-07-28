python tests/benchmark_openai_api.py --prompt-file tests/data/docvqa_train_10k.jsonl --image-folder `pwd`/tests/data/share_textvqa/images/ --req-nums 100 \
	--batch-size 32 \
	--image-nums-mean 3 \
	--image-nums-range 1  \
	--response-mean 120 \
	--response-len-range 64 \

