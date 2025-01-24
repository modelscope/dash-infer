'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    config.py
'''
from collections import defaultdict


class ModelContext:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelContext, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.context = defaultdict()

    def set(self, key, value):
        self.context[key] = value

    def get(self, key):
        return self.context.get(key)


model_contexts = {}


def load_args_into_config(args):
    global model_contexts
    context = ModelContext()
    for key, value in vars(args).items():
        context.set(key, value)
    model_contexts["default"] = context
    return context


def get_model_context(name="default"):
    global model_contexts
    return model_contexts.get(name)


def add_context_args(parser):
    group = parser.add_argument_group("VL Model Engine", "model context")
    group.add_argument("--model", type=str, required=True, help="model name or path")
    group.add_argument(
        "--vision_engine",
        type=str,
        default="tensorrt",
        choices=["tensorrt", "transformers"],
        help="engine to run vision model",
    )
    group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help="device (Default: cuda)",
    )
    group.add_argument("--max_length", type=int, default=32000, help="model max length")
    group.add_argument("--max_batch", type=int, default=128, help="max batch")
    group.add_argument(
        "--parallel_size",
        type=int,
        default=1,
        help="number of devices used to run engine",
    )
    group.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help="Enables prefix caching.",
    )
    group.add_argument(
        "--quant-type",
        default=None,
        choices=["gptq", "gptq_weight_only", "a8w8", "a16w4", "a16w8", "fp8"],
        help="The default strategy of GPTQ models is activation quantization. To disable activation quantization, please use gptq_weight_only mode. The quantization type 'axwy' means x-bit activations and y-bit weights, which will use dynamic quantization for models that haven't undergone quantization fine-tuning"
    )
    group.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    group.add_argument(
        "--min-pixels",
        default=4*28*28,
        type=int,
        help="The min pixels of the image to resize the image.",
    )
    group.add_argument(
        "--max-pixels",
        default=16384*28*28,
        type=int,
        help="The max pixels of the image to resize the image.",
    )
