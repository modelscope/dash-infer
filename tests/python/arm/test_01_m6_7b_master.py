'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_01_m6_7b_master.py
'''
import unittest
from dashinfer import allspark
from dashinfer.allspark.quantization import QuantizeConfig
import torch.utils.dlpack
import numpy as np
import os
import shutil
import subprocess

CURRENT_PATH = os.path.split(__file__)[0:-1][0]


def process_7b_torch_model(pth_path):
    model = torch.load(pth_path, map_location=lambda storage, loc: storage)
    module = model['model']["language_model"]["encoder"]
    module["word_embeddings"] = model['model']["language_model"]["embedding"][
        "word_embeddings"]["weight"]
    module["lm_head.weight"] = model["model"]["language_model"][
        "output_layer"]['weight']
    for key in module.keys():
        module[key] = module[key].float()
    torch_model = module
    return torch_model


def build_a16w8_model(torch_model,
                      model_name,
                      model_type,
                      model_config,
                      device_type,
                      device_ids,
                      FLOAT_TYPE="float32"):
    FLOAT_TYPE = FLOAT_TYPE.lower()
    ACT_TYPE = "bfloat16"
    INT_TYPE = "uint8"

    engine = allspark.Engine()
    engine.set_device_type(device_type)
    engine.set_device_ids(device_ids)
    engine.build_model_from_torch(
        model_name=model_name,
        model_type=model_type,
        torch_model=torch_model,
        data_type=FLOAT_TYPE,
        multigpu_mode=1,
        model_config=model_config,
        is_generate=True,
        derive_type="lmhead",
        do_dynamic_quantize_convert=True,  # a16w8
        quant_config=QuantizeConfig(activation_type=ACT_TYPE, weight_type=INT_TYPE, extra_option={
            "SubChannel": True,
            "GroupSize": 64
        }),
        save_dir=os.path.join(CURRENT_PATH, model_name))
    return engine


class M6_7B_A16W8_ARM_TestCase(unittest.TestCase):

    def setUp(self):
        self.models_root_path = os.environ.get("ALLSPARK_TESTCASE_PATH")
        self.m6_7b_pt_model_path = os.path.join(
            self.models_root_path, "testcase",
            "m6_7b_a16w8/m6_7b_0802_chat.pt")

        # ModelType
        self.model_type = "M6v3"
        # Config
        self.build_model_config = {
            "layer_norm_eps": 1e-5,
            "layernorm_epsilon": 1e-5,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
        }
        # TorchModel
        self.torch_model = process_7b_torch_model(self.m6_7b_pt_model_path)

    def test_m6_7b_a16w8_arm(self):
        model_name = "m6_7b_a16w8"
        device_type = "CPU"
        device_ids = [0]

        out_ids_ref = [[
            101211, 9370, 65770, 105542, 101314, 11319, 151645, 198, 105678,
            9370, 65770, 36993, 20412, 104130, 1773, 151645, 198, 151643
        ]]

        engine = build_a16w8_model(self.torch_model, model_name,
                                   self.model_type, self.build_model_config,
                                   device_type, device_ids)

        return_code = subprocess.call(
            "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 ./mpirun --map-by numa:pe=64 -np 2 sh -c 'python3 test_01_m6_7b_worker.py'",
            shell=True)

        out_sync = np.load(os.path.join(model_name, "out_sync.npy")).tolist()

        self.assertEqual(out_ids_ref, out_sync)

        # Clean
        shutil.rmtree(os.path.join(CURRENT_PATH, model_name))


if __name__ == "__main__":
    unittest.main()
