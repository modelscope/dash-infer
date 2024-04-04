#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    model_base.py
#
from .allspark_pb2 import *
import torch
import numpy as np
from io import BytesIO
from os import path, makedirs
from dashinfer import allspark
import time
import json
import re
"""
define a tensor
"""


def make_tensor(name, data=None):
    tensor = TensorProto()
    tensor.name = name
    if data is not None:
        tensor.data = trans_to_allsparkz(data)
    return tensor


def get_activation(name):
    if name == "RELU" or name == "relu":
        return RELU
    elif name == "TANH" or name == "tanh":
        return TANH
    elif name == "GELU_ERF" or name == "gelu_erf":
        return GELU_ERF
    elif name == "GELU_TANH" or name == "gelu_tanh":
        return GELU_TANH
    elif name == "SILU" or name == "silu":
        return SILU
    else:
        return UNARYTYPE_UNDEFINED


class GenerateConfig(object):

    def __init__(
        self,
        num_beams=1,
        do_sample=True,
        early_stopping=False,
        temperature=1.0,
        top_k=50,
        top_p=0.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        min_length=5,
        max_length=20,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        eos_token_id=0,
        fusion_in_decoder=False,
        bad_words_ids=None,
    ) -> None:
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.min_length = min_length
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_return_sequences = num_return_sequences
        self.eos_token_id = eos_token_id
        self.fusion_in_decoder = fusion_in_decoder
        self.bad_words_ids = bad_words_ids


"""
define a operator
"""


class Operator(object):

    def __init__(self, op_type, op_name, inputs, op_attr={}):
        self.op = OperatorProto()
        self.op.op_type = op_type
        self.op.op_name = op_name
        self.set_op_attr(op_attr)
        if isinstance(inputs, list) and inputs:
            self.op.inputs.extend(inputs)
        else:
            self.op.inputs.append(inputs)

    def set_op_attr(self, op_attr):
        for key in op_attr:
            self.op.attr[key] = torch.tensor(op_attr[key]).numpy().tobytes()

    def __call__(self):
        return self.op


class Embedding(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Embedding", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".word_embeddings"))
        self.op.weights.append(make_tensor(op_name + ".position_embeddings"))
        if "token_embedding" in op_attr and op_attr["token_embedding"]:
            self.op.weights.append(
                make_tensor(op_name + ".token_type_embeddings"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class EmbeddingT5(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("EmbeddingT5", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".word_embeddings"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class RichEmbedding(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("RichEmbedding", op_name, inputs, op_attr)
        self.op.outputs.append(inputs[1])


class LayerNorm(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("LayerNorm", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".gamma"))
        self.op.weights.append(make_tensor(op_name + ".beta"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class LayerNormNoBeta(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("LayerNormNoBeta", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".gamma"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Gemm(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Gemm", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".weight"))
        if "with_bias" in op_attr and op_attr["with_bias"] == False:
            pass
        else:
            self.op.weights.append(make_tensor(op_name + ".bias"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class DynamicQuantizeMatmul(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("DynamicQuantizeMatmul", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".weight"))
        self.op.weights.append(make_tensor(op_name + ".weight.scale"))
        self.op.weights.append(make_tensor(op_name + ".weight.zero_point"))
        self.op.weights.append(make_tensor(op_name + ".weight.redsum"))
        if "with_bias" in op_attr and op_attr["with_bias"] == False:
            pass
        else:
            self.op.weights.append(make_tensor(op_name + ".bias"))
        self.op.outputs.append(
            make_tensor(op_name + ".out",
                        np.empty(shape=(0, 0), dtype=np.float32)))


class Quantize(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Quantize", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))
        self.op.outputs.append(make_tensor(op_name + ".out.scale"))
        self.op.outputs.append(make_tensor(op_name + ".out.zero_point"))
        self.op.outputs.append(make_tensor(op_name + ".out.redsum"))


class MultiHeadAttention(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("MultiHeadAttention", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class MultiQueryAttention(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("MultiQueryAttention", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class EncdecAttention(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("EncdecAttention", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Binary(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Binary", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Unary(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Unary", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Mul(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Mul", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Cast(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Cast", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class TransMask(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("TransMask", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))
        self.op.outputs.append(make_tensor("batch_offset"))


class RelativePE(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("RelativePE", op_name, inputs, op_attr)
        self.op.weights.append(
            make_tensor(op_name + ".relative_attention_bias"))
        self.op.outputs.append(
            make_tensor(op_name + ".out", np.empty(shape=(0, 0))))


class ALiBiPE(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("ALiBiPE", op_name, inputs, op_attr)
        self.op.outputs.append(
            make_tensor(op_name + ".out", np.empty(shape=(0, 0))))


# """
#  @brief: 构建decoder最初的dec_id, 一共有两种方式:
#   1) 从input_ids复制获得, 输入输出如下:
#      inputs :
#        > input_ids [batch, seq_len]
#      outputs :
#        > dec_ids [batch, seq_len]
#        > max_dec_ids [batch, max_length]
#   2) 从start_ids broadcast获得, 无输入, 输出如下:
#      op_attr : {"start_id" : xxx}
#      outputs :
#        > dec_ids [batch, 1]
#        > max_dec_ids [batch, max_length]
# """
class PreProcessId(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("PreProcessId", op_name, inputs, op_attr)
        self.op.outputs.append(
            make_tensor("dec_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.op.outputs.append(
            make_tensor("max_dec_ids", np.empty(shape=(0, 0), dtype=np.int64)))


class PostProcessId(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("PostProcessId", op_name, inputs, op_attr)
        self.op.outputs.append(
            make_tensor("generated_ids", np.empty(shape=(0, 0),
                                                  dtype=np.int64)))


class UpdateId(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("UpdateId", op_name, inputs, op_attr)
        self.op.outputs.append(
            make_tensor("max_dec_ids", np.empty(shape=(0, 0), dtype=np.int64)))


class Sample(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Sample", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class BeamSearch(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("BeamSearch", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))
        self.op.outputs.append(
            make_tensor("next_beam_idx", np.empty(shape=(0, 0),
                                                  dtype=np.int32)))
        self.op.outputs.append(
            make_tensor("hyps_ids", np.empty(shape=(0, 0), dtype=np.int64)))


class GenerateOp(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("GenerateOp", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))
        self.op.outputs.append(
            make_tensor("next_beam_idx", np.empty(shape=(0, 0),
                                                  dtype=np.int32)))
        self.op.outputs.append(
            make_tensor("hyps_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.op.outputs.append(
            make_tensor("hyps_beam_score",
                        np.empty(shape=(0, 0), dtype=np.float32)))


class AllGather(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("AllGather", op_name, inputs, op_attr)
        # self.op.outputs.append(self.op.inputs[0])
        self.op.outputs.append(make_tensor(op_name + ".out"))


class AllReduce(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("AllReduce", op_name, inputs, op_attr)
        # self.op.outputs.append(make_tensor(op_name + ".out"))
        self.op.outputs.append(self.op.inputs[0])


class Split(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Split", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Rotary(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Rotary", op_name, inputs, op_attr)
        if "use_weight" in op_attr and op_attr["use_weight"] == True:
            self.op.weights.append(make_tensor(op_name + ".inv_freq"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class RotaryMulQuery(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("RotaryMulQuery", op_name, inputs, op_attr)
        if "use_weight" in op_attr and op_attr["use_weight"] == True:
            self.op.weights.append(make_tensor(op_name + ".inv_freq"))
        self.op.outputs.append(make_tensor(op_name + ".out"))


class Chunk(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("Chunk", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class ChunkBinary(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("ChunkBinary", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


class GetLastLine(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("GetLastLine", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))


"""
define a base model
"""


class Model(object):

    def __init__(
            self,
            model_type,
            data_type,
            model_config,
            multinode_mode=1,
            do_binary_add_fused=True,
            do_dynamic_quantize_convert=False,
            quant_config=None,
            enable_i8cache_mha=False,  # deprecated, please use mha_kv_cache_config instead
            mha_kv_cache_config=None,
            weights_path="",
            use_dynamic_ntk=False,
            use_logn_attn=False,
            model_sequence_length=2048,
            seqlen_extrapolation=1.0,
            rotary_base=10000):
        Model.dtype_dict = {
            "float32": FLOAT32,
            "float16": FLOAT16,
            "bfloat16": BFLOAT16
        }
        self.model = TransformerProto()
        self.model.model_type = model_type
        self.dtype = Model.dtype_dict[data_type]
        self.model_config = model_config
        self.multinode_mode = multinode_mode
        self.do_binary_add_fused = do_binary_add_fused
        self.do_dynamic_quantize_convert = do_dynamic_quantize_convert
        self.weights = {}
        self.weight_name_map = {}
        self.sparse_map = {}
        self.split_map = {}
        self.quantize_map = {}
        self.quant_config = quant_config
        self.enable_i8cache_mha = enable_i8cache_mha
        self.mha_kv_cache_config = mha_kv_cache_config
        self.weights_path = weights_path
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.model_sequence_length = model_sequence_length
        self.seqlen_extrapolation = seqlen_extrapolation
        self.rotary_base = rotary_base

    def __call__(self):
        return self.model

    def _load_weights(self):
        for key, data in self.weights.items():
            tensor_proto = self.model.weights[key]
            tensor_proto.name = key
            tensor_proto.data = data

        for orig_name in list(weights):
            name = orig_name
            normalized_name = re.sub(r'^.*(\.\d+\.)(.+)$', r'layers\1\2', name)
            weights[normalized_name] = weights[name]
            del weights[name]

        # for tensors without .weight or .bias, all regarded as .weight
        for orig_name in list(weights):
            if not re.match(r'\.(weight|bias)$', orig_name):
                name = orig_name + '.weight'
                weights[name] = weights[orig_name]
                del weights[orig_name]

        for name in list(weights):
            if weights[name].dtype != self.dtype:
                if self.dtype == FLOAT16:
                    weights[name] = weights[name].half()
                elif self.dtype == BFLOAT16:
                    weights[name] = weights[name].bfloat16()
                elif self.dtype == FLOAT32:
                    weights[name] = weights[name].float()
                else:
                    raise Exception(f"not supported target dtype {self.dtype}")

        return weights

    def make_gemm_op(self, gemm_name, inputs, op_attr={}):
        return Gemm(gemm_name, inputs, op_attr)

    def _add_layer(self, layer_type, graph, in_tensor):
        Model.switcher = {
            "pooler": self._add_pooler_layer,
            "lmhead": self._add_lmhead_layer,
        }
        func = Model.switcher.get(layer_type, "Invalid")
        if func != "Invalid":
            func(graph, in_tensor)
        else:
            raise RuntimeError("Invalid additional layer type")

    def _add_pooler_layer(self, graph, in_tensor):
        out_gemm = Gemm("pooler.dense", in_tensor, {
            "is_pooler": True,
            "activation": TANH
        })()
        graph.ops.extend([out_gemm])

    def _add_lmhead_layer(self, graph, in_tensor):
        if self.multinode_mode != 0:
            getlastline = GetLastLine("get_last_line", in_tensor)()
            lmhead = Gemm("lm_head", getlastline.outputs[0], {
                "with_bias": False,
                "splitk": True
            })()
            lmhead.outputs[0].name = "logits"
            all_reduce_lmhead = AllReduce("all_reduce_lmhead",
                                          lmhead.outputs[0])()
            graph.ops.extend([getlastline, lmhead, all_reduce_lmhead])
            self.split_map.update({
                "lm_head.weight": HSPLIT,
            })
        else:
            getlastline = GetLastLine("get_last_line", in_tensor)()
            lmhead = Gemm("lm_head", getlastline.outputs[0],
                          {"with_bias": False})()
            graph.ops.extend([getlastline, lmhead])

    def validate_weight_dtype(self, w_name, tensor, self_dtype_str):
        if str(tensor.dtype) != "torch." + self_dtype_str:
            raise ValueError(
                "DataType not match, [weight dtype: {}] vs [model dtype:{}]".
                format(str(tensor.dtype), "torch." + self_dtype_str))


# numpy转allsparkz， 兼容量化模型以及部分旧模型转换所以保留
def trans_to_allsparkz(input,
                       data_mode=DENSE,
                       split_mode=NOSPLIT,
                       group_list=[]):
    input_dict = {
        np.float64: ('f', 8),
        np.float32: ('f', 4),
        np.float16: ('f', 2),
        np.uint64: ('u', 8),
        np.uint32: ('u', 4),
        np.uint16: ('u', 2),
        np.uint8: ('u', 1),
        np.int64: ('i', 8),
        np.int32: ('i', 4),
        np.int16: ('i', 2),
        np.int8: ('i', 1),
        bool: ('b', 1)
    }
    if not input.dtype.type in input_dict:
        return None
    input_dtype = input_dict[input.dtype.type][0]
    input_word_size = input_dict[input.dtype.type][1]
    input_shape = list(input.shape)
    input_str = input.tobytes()
    py_attr = {
        'sparse_type': data_mode,
        'split_mode': split_mode,
        'shape': input_shape,
        'dtype': input_dtype,
        'word_size': input_word_size,
        'group_list': group_list,
    }
    output_str = allspark.save_allsparky(input_str, py_attr)
    return output_str


def save_torch_to_allsparky(weights_path,
                            name,
                            input,
                            data_mode=DENSE,
                            split_mode=NOSPLIT,
                            group_list=[]):
    input_dict = {
        torch.float64: ('f', 8),
        torch.float32: ('f', 4),
        torch.float16: ('f', 2),
        torch.bfloat16: ('b', 2),
        torch.uint8: ('u', 1),
        torch.int64: ('i', 8),
        torch.int32: ('i', 4),
        torch.int16: ('i', 2),
        torch.int8: ('i', 1),
        bool: ('b', 1)
    }
    if not input.dtype in input_dict:
        return None
    input_dtype = input_dict[input.dtype][0]
    input_word_size = input_dict[input.dtype][1]
    input_shape = list(input.shape)
    dldata = torch.utils.dlpack.to_dlpack(input.to('cpu'))
    py_attr = {
        'sparse_type': data_mode,
        'split_mode': split_mode,
        'shape': input_shape,
        'dtype': input_dtype,
        'word_size': input_word_size,
        'group_list': group_list,
    }
    allspark.save_allsparky_dltensor_tofile(weights_path, name, dldata,
                                            py_attr)


def set_global_header(weights_path):
    allspark.set_global_header(weights_path)
