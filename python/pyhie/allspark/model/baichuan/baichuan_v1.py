'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    baichuan_v1.py
'''
from .baichuan_v2 import *


class Baichuan_v1(Baichuan_v2):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super(Baichuan_v2, self).__init__("Baichuan_v1", data_type, **kwargs)
        self.model.inputs.append(
            make_tensor("input_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.model.inputs.append(
            make_tensor("attention_mask", np.empty(shape=(0, 0),
                                                   dtype=np.int64)))
        self.model.outputs.append(make_tensor("last_hidden_state"))
        self.is_generate = kwargs.get('is_generate', True)
        self.weight_real_names = set()
        for v in torch_model:
            self.weight_real_names.add(v)
        self._build_graph(self.model_config, derive_type)
        start_time = time.time()
        if not self.only_convert_lora:
            self._trans_weight(torch_model)
        print("parse weight time: ", time.time() - start_time)
