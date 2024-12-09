'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    qwen_v20.py
'''
from .qwen_v15 import *

class Qwen_v20(Qwen_v15):
    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super(Qwen_v15, self).__init__("Qwen_v20", data_type, **kwargs)
        self.model.inputs.append(
            make_tensor("input_ids", np.empty(shape=(0, 0), dtype=np.int64))
        )
        self.model.inputs.append(
            make_tensor("attention_mask", np.empty(shape=(0, 0), dtype=np.int64))
        )
        self.covert_namespace_qweight_to_weight(torch_model)
        self.model.outputs.append(make_tensor("last_hidden_state"))
        self.is_generate = kwargs.get("is_generate", True)
        self.weight_real_names = set()
        for v in torch_model:
            self.weight_real_names.add(v)
        self._build_graph(self.model_config, derive_type)
        start_time = time.time()
        if not self.only_convert_lora:
            self._trans_weight(torch_model)
        self._trans_lora_weight(self._trans_weight)
        print("parse weight time: ", time.time() - start_time)
