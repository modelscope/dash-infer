'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_01_m6_7b_worker.py
'''
from dashinfer import allspark
import torch.utils.dlpack
import numpy as np
import os

CURRENT_PATH = os.path.split(__file__)[0:-1][0]

MAX_LENGTH = 8192
MAX_BATCHSIZE = 1


def build_as_model_config(model_name, device_type, device_ids):
    as_model_path = os.path.join(CURRENT_PATH, model_name)
    as_model_config = allspark.AsModelConfig(
        model_name=model_name,
        model_path=os.path.join(as_model_path, model_name + ".asgraph"),
        weights_path=os.path.join(as_model_path, model_name + ".asparam"),
        device_type=device_type,
        device_ids=device_ids,
        engine_max_length=MAX_LENGTH,
        engine_max_batch=MAX_BATCHSIZE,
    )
    return as_model_config


def run_model_sync(engine, model_name, in_ids, generate_config):
    in_mask = (np.array(in_ids) != 0).astype(np.int64)
    torch_input = {
        "input_ids": torch.Tensor(in_ids).to(torch.int64),
        "attention_mask": torch.Tensor(in_mask).to(torch.int64),
    }

    generate_config["async"] = False

    out_ids = engine.run_text_generation(model_name, {
        "input_ids":
        torch.utils.dlpack.to_dlpack(torch_input["input_ids"]),
        "attention_mask":
        torch.utils.dlpack.to_dlpack(torch_input["attention_mask"]),
    },
                                         generate_config=generate_config)

    if "generated_ids" in out_ids:
        out_ids = torch.utils.dlpack.from_dlpack(out_ids["generated_ids"])
        out_list = out_ids.cpu().numpy().tolist()
        np.save(os.path.join(model_name, "out_sync.npy"),
                out_ids.cpu().numpy())
        return out_list


def test_model():
    model_name = "m6_7b_a16w8"
    device_type = "CPU"
    device_ids = [0]

    generate_config = {
        'num_beams': 1,
        'num_return_sequences': 1,
        'temperature': 1.0,
        'do_sample': True,
        'early_stopping': True,
        'top_k': 1,
        'top_p': 0.5,
        'max_length': MAX_LENGTH,
        # 'stop_words_ids': [[151643], [151644], [151645]],  # qwen_15w
        'eos_token_id': 151643,
        'seed': 1234,
        'loop_context': True
    }

    in_ids = [[101211, 9370, 65770, 105542, 101314, 11319]]

    as_model_config = build_as_model_config(model_name, device_type,
                                            device_ids)

    engine = allspark.Engine()
    engine.build_model_from_config_struct(as_model_config)
    engine.set_matmul_precision("medium")  # highest/high/medium

    out_sync = run_model_sync(engine, model_name, in_ids, generate_config)


if __name__ == "__main__":
    test_model()
