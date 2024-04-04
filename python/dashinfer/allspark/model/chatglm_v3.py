#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    chatglm_v3.py
#
from .chatglm_v2 import *


class ChatGLM_v3(ChatGLM_v2):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super(ChatGLM_v2, self).__init__("ChatGLM_v3", data_type, **kwargs)
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
        self.invfreq_type_ = 2
        self._build_graph(self.model_config, derive_type)
        start_time = time.time()
        self._trans_weight(torch_model)
        print("parse weight time: ", time.time() - start_time)
