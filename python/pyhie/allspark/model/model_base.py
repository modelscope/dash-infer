'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    model_base.py
'''
from .._allspark import *
import torch
import numpy as np
from io import BytesIO
from os import path, makedirs
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

def get_invfreq_type(name):
    if name == "YARN" or name == "yarn":
        return yarn
    else:
        return base_rotary


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


class GemmLoraCapsule(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("GemmLoraCapsule", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".weight"))
        if "with_bias" in op_attr and op_attr["with_bias"] == False:
            pass
        else:
            self.op.weights.append(make_tensor(op_name + ".bias"))
        ''' optimized into .aslora file
        if "lora_scaling" not in op_attr:
            self.set_op_attr({'lora_scaling': 1.0})
        '''
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

class UnaryGLU(Operator):
    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("UnaryGLU", op_name, inputs, op_attr)
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

class CalcExpert(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("CalcExpert", op_name, inputs, op_attr)
        self.op.outputs.append(make_tensor(op_name + ".out"))

class MOE(Operator):

    def __init__(self, op_name, inputs, op_attr={}):
        super().__init__("MOE", op_name, inputs, op_attr)
        self.op.weights.append(make_tensor(op_name + ".gate_up_proj.weight"))
        # self.op.weights.append(make_tensor(op_name + ".up_proj.weight"))
        self.op.weights.append(make_tensor(op_name + ".down_proj.weight"))
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
            multigpu_mode=1,
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
            rotary_base=10000,
            lora_cfg=None):
        Model.dtype_dict = {
            "float32": FLOAT32,
            "float16": FLOAT16,
            "bfloat16": BFLOAT16
        }
        self.model = TransformerProto()
        self.model.model_type = model_type
        self.dtype = Model.dtype_dict[data_type]
        self.model_config = model_config
        self.multigpu_mode = multigpu_mode
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

        self.gen_lora_op = False
        self.lora_names = []
        self.only_convert_lora = False
        if type(lora_cfg) == type(dict()):
            self.gen_lora_op = True
            self.lora_base_dir = lora_cfg.get('input_base_dir', '.')
            self.lora_names = lora_cfg.get('lora_names', [])
            self.lora_weight_name_map = {}
            self.lora_sparse_map = {}
            self.lora_split_map = {}
            self.lora_quantize_map = {}
            self.lora_name_scaling_map = {} 
            self.only_convert_lora = lora_cfg.get('lora_only', False)
            self._load_lora_cfg(self.lora_names)

        self.adapted_quant_method = self.quant_config.extra_option.get(
            'AdaptedQuantMethod') if self.quant_config != None else None

    def __call__(self):
        return self.model

    def _load_weights(self):
        for key, data in self.weights.items():
            tensor_proto = self.model.weights[key]
            tensor_proto.name = key
            tensor_proto.data = data

    def _get_lora_path(self, lora_name):
        return path.join(path.dirname(self.weights_path),
                         lora_name + '.aslora')

    def _load_lora_cfg(self, lora_names):
        for lora_name in lora_names:
            lora_input_dir = path.join(self.lora_base_dir, lora_name)
            if not path.exists(lora_input_dir):
                raise Exception(f"LoRA dir {lora_input_dir} not found!")
            lora_cfg_file = path.join(lora_input_dir, 'adapter_config.json')
            if path.exists(lora_cfg_file):
                with open(lora_cfg_file, 'r') as f:
                    cfg = json.load(f)
                    cfg_lora_alpha = cfg.get("lora_alpha")
                    cfg_lora_r = cfg.get("lora_r")  or cfg.get("r")
            else:
                raise Exception(f"LoRA cfg file {lora_cfg_file} not found!")
            if cfg_lora_r == None or cfg_lora_alpha == None:
                raise Exception(
                    f"{lora_name}: missing lora_r/r, lora_alpha, please check your lora adapter_config.json file"
                )
            self.lora_name_scaling_map[lora_name] = float(cfg_lora_alpha) / float(cfg_lora_r)

    def _load_lora_name_map(self, lora_dir):
        ret = {}
        if not path.exists(lora_dir):
            raise Exception(f"LoRA dir {lora_input_dir} not found!")
        lora_cfg_file = path.join(lora_dir, 'adapter_config.json')
        if path.exists(lora_cfg_file):
            with open(lora_cfg_file, 'r') as f:
                cfg = json.load(f)
                ret = cfg.get("lora_layer_name_map", {})
        return ret

    # 根据LoRA目录中的adapter_config.json和 adapter_model.bin ， 提取出所有的LoRA tensors，乘以lora_scaling系数，并将tensors的名字进行规范化
    def _load_lora_torchweights(self, lora_name):
        lora_input_dir = path.join(self.lora_base_dir, lora_name)
        if not path.exists(lora_input_dir):
            raise Exception(f"LoRA dir {lora_input_dir} not found!")
        lora_layer_name_map = self._load_lora_name_map(lora_input_dir)

        # 加载原始LoRA weights
        lora_input_binfile = path.join(lora_input_dir, 'adapter_model.bin')
        weights = torch.load(lora_input_binfile,
                             map_location=torch.device('cpu'))
        '''
        for k in list(weights):
            if weights[k].dtype == torch.float32:
                weights[k] = weights[k].half()
        '''

        # 对weights名称进行规范化
        for train_name, std_name in lora_layer_name_map.items():
            for old_name in list(weights):
                if train_name in old_name:
                    new_name = old_name.replace(train_name, std_name)
                    weights[new_name] = weights[old_name]
                    del weights[old_name]

        for orig_name in list(weights):
            name = orig_name
            #normalized_name = re.sub(r'^.*(\.\d+\.)(.+)$', r'model.layers\1\2', name)
            normalized_name = re.sub(r'\.(layers|layer|h)(\.\d+\.)(.+)$', r'.layers\2\3', name)
            if normalized_name != name:
                weights[normalized_name] = weights[name]
                del weights[name]

        # for tensors without .weight or .bias, all regarded as .weight
        for orig_name in list(weights):
            if not re.match(r'.*\.(weight|bias)$', orig_name):
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
                    raise Exception(
                        f"not supported LoRA targe dtype {self.dtype}")
            lora_scaling = self.lora_name_scaling_map[lora_name]
            if  lora_scaling != 1.0 and '.lora_B.weight' in name:
                weights[name] *= lora_scaling

        return weights

    def _make_lora_split_map(self):
        for lora_name in self.lora_names:
            self.lora_split_map[lora_name] = {}
            for k, v in self.split_map.items():
                lora_k = re.sub(r'(?=\.(weight|bias)$)', '.lora_A', k)
                self.lora_split_map[lora_name][lora_k] = NOSPLIT if v in [
                    VSPLIT, QKVSPLIT, GROUP_VSPLIT
                ] else v  # loraA 在各种竖切方式下，与base保持一致，此时loraB不切割
                lora_k = re.sub(r'(?=\.(weight|bias)$)', '.lora_B', k)
                self.lora_split_map[lora_name][
                    lora_k] = NOSPLIT if v == HSPLIT else v  # 横切满足乘法分配律， loraB在HSPLIT模式下横切，此时loraA不切割

    # for lora quantization
    def _make_lora_quant_map(self):
        for lora_name in self.lora_names:
            self.lora_quantize_map[lora_name] = {}
            for k in list(self.quantize_map):
                lora_k = re.sub(r'(?=\.weight$)', '.lora_A', k)
                self.lora_quantize_map[lora_name][lora_k] = 1
                lora_k = re.sub(r'(?=\.weight$)', '.lora_B', k)
                self.lora_quantize_map[lora_name][lora_k] = 1

    def _make_lora_weight_name_map(self, unified_lora_torch_keys,
                                   base_weight_name_map):
        ret = {}
        affected_inner_names = set()
        amend_lora_torch_names = set()
        for lora_torch_k in unified_lora_torch_keys:
            tmp_k = re.sub(r'\.lora_.', '', lora_torch_k)
            matched = False
            for inner_name, torch_name in base_weight_name_map.items():
                if isinstance(torch_name, list):  # attention qkv weights
                    for name in torch_name:
                        if name == tmp_k:
                            affected_inner_names.add(inner_name)
                            matched = True
                            break
                else:
                    if torch_name == tmp_k:
                        affected_inner_names.add(inner_name)
                        matched = True
                        break
            '''
            if not matched:
                amend_lora_torch_names.add(lora_torch_k)
            '''
            #assert matched, f"lora key {lora_torch_k} cannot match any in base"

        for inner_name in affected_inner_names:
            base_torch_name = base_weight_name_map[inner_name]

            lora_inner_name = re.sub(r'(?=\.(weight|bias)$)', '.lora_A',
                                     inner_name)
            if isinstance(base_torch_name, list):
                lora_torch_name = [
                    re.sub(r'(?=\.(weight|bias)$)', '.lora_A', name)
                    for name in base_torch_name
                ]
            else:
                lora_torch_name = re.sub(r'(?=\.(weight|bias)$)', '.lora_A',
                                         base_torch_name)
            ret[lora_inner_name] = lora_torch_name

            lora_inner_name = re.sub(r'(?=\.(weight|bias)$)', '.lora_B',
                                     inner_name)
            if isinstance(base_torch_name, list):
                lora_torch_name = [
                    re.sub(r'(?=\.(weight|bias)$)', '.lora_B', name)
                    for name in base_torch_name
                ]
            else:
                lora_torch_name = re.sub(r'(?=\.(weight|bias)$)', '.lora_B',
                                         base_torch_name)
            ret[lora_inner_name] = lora_torch_name

        return ret

    # save all lora weights to files
    def _trans_lora_weight(self, trans_func):
        for lora_name in self.lora_names:
            lora_weights = self._load_lora_torchweights(lora_name)
            trans_func(lora_weights, lora_name)

    def make_gemm_op(self, gemm_name, inputs, op_attr={}):
        if self.gen_lora_op:
            return GemmLoraCapsule(gemm_name, inputs, op_attr)
        else:
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
        # self.weight_name_map.update(
        #     {
        #         "pooler.dense.weight": self.name_adapter.fullname("pooler.dense.weight"),
        #         "pooler.dense.bias": self.name_adapter.fullname("pooler.dense.bias"),
        #     }
        # )
        # if self.multigpu_mode!=0:
        #     all_gather_pooler = AllGather("all_gather_pooler",out_gemm.outputs[0])()
        #     graph.ops.append(all_gather_pooler)
        #     self.split_map.update(
        #     {
        #         "pooler.dense.weight": VSPLIT,
        #         "pooler.dense.bias": VSPLIT,
        #     }
        # )
    def _add_lmhead_layer(self, graph, in_tensor):
        if self.multigpu_mode != 0:
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

    def convert_namespace_hf2megatron_for_gptq(self, orig_model):
        if self.adapted_quant_method != "GPTQ":
            return orig_model
        state_dict = orig_model
        for old_name in list(state_dict.keys()):
            if old_name.startswith('transformer.wte'):
                new_name = old_name.replace('transformer.wte.weight',
                                            'word_embeddings')
            elif old_name.startswith('transformer.h'):
                new_name = old_name.replace('transformer.h', 'layers')
                if new_name.find('ln_1') != -1:
                    new_name = new_name.replace('ln_1', 'input_layernorm')
                if new_name.find('attn.c_attn') != -1:
                    new_name = new_name.replace(
                        'attn.c_attn', 'self_attention.query_key_value')
                if new_name.find('attn.c_proj') != -1:
                    new_name = new_name.replace('attn.c_proj',
                                                'self_attention.dense')
                if new_name.find('ln_2') != -1:
                    new_name = new_name.replace('ln_2',
                                                'post_attention_layernorm')
                if new_name.find('mlp.c_proj') != -1:
                    new_name = new_name.replace('mlp.c_proj',
                                                'mlp.dense_4h_to_h')
            elif old_name.startswith('transformer.ln_f'):
                new_name = old_name.replace('transformer.ln_f',
                                            'final_layernorm')
            else:
                new_name = old_name
            new_name = re.sub(r'\.qweight$', '.weight', new_name)
            state_dict[new_name] = state_dict.pop(old_name)
        return state_dict

    def covert_namespace_qweight_to_weight(self, orig_model):
        if self.adapted_quant_method not in ["GPTQ", "GPTQ_NO_PACK"]:
            return orig_model
        state_dict = orig_model
        for old_name in list(state_dict.keys()):
            new_name = old_name
            new_name = re.sub(r'\.qweight$', '.weight', new_name)
            state_dict[new_name] = state_dict.pop(old_name)
        return state_dict
    
    def validate_weight_dtype(self, w_name, tensor, self_dtype_str):
        if self.adapted_quant_method == 'GPTQ':
            if (re.match(r'.*\.(weight|qzeros|g_idx)$', w_name) and tensor.dtype
                    != torch.int32) or (re.match(r'.*\.scales$', w_name) and str(
                        tensor.dtype) != self_dtype_str):
                raise ValueError(f"GPTQ DataType not correct for {w_name}")
        if self.adapted_quant_method == 'GPTQ_NO_PACK':
            if (re.match(r'.*\.(scales)$', w_name) and str(tensor.dtype)
                    != self_dtype_str):
                raise ValueError(f"GPTQ_NO_PACK DataType not correct for {w_name}, require {self_dtype_str} but found {str(tensor.dtype)}")
        elif str(tensor.dtype) != "torch." + self_dtype_str:
            raise ValueError(
                f"DataType not match, tensor name: {w_name} [weight dtype: {str(tensor.dtype)}] vs [model dtype:{'torch.'+self_dtype_str}]")


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
    # print("input_dtype: {} input_word_size: {} input_shape:{} input_str size: {}".format(input_dtype, input_word_size, input_shape, len(input_str)))
    output_str = save_allsparky(input_str, py_attr)
    # output_str = allspark.save_allsparky(input_str, data_mode, split_mode, input_shape, input_dtype, input_word_size)
    return output_str


# Torch.tensor转allsparkz，之后默认都用该接口
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
        # torch.uint64:('u',8), torch.uint32:('u',4), torch.uint16:('u',2),
        torch.uint8: ('u', 1),
        torch.int64: ('i', 8),
        torch.int32: ('i', 4),
        torch.int16: ('i', 2),
        torch.int8: ('i', 1),
        torch.float8_e4m3fn: ('f', 1),
        bool: ('b', 1)
    }
    if not input.dtype in input_dict:
        return None
    input_dtype = input_dict[input.dtype][0]
    input_word_size = input_dict[input.dtype][1]
    input_shape = list(input.shape)
    # dlpack don't support float8 now, so store as int8 type
    if input.dtype == torch.float8_e4m3fn: 
        input = input.view(torch.int8)
    dldata = torch.utils.dlpack.to_dlpack(input.to('cpu'))
    # allspark.h中的TensorAttribute 对齐
    py_attr = {
        'sparse_type': data_mode,
        'split_mode': split_mode,
        'shape': input_shape,
        'dtype': input_dtype,
        'word_size': input_word_size,
        'group_list': group_list,
    }

    # print("input_dtype: {} input_word_size: {} input_shape:{} input_str size: {}".format(input_dtype, input_word_size, input_shape, len(input_str)))
    save_allsparky_dltensor_tofile(weights_path, name, dldata,
                                            py_attr)


