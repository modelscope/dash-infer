#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    qwen_v10.py
#
from .model_base import *
from .utils import WeightNameAdapter
from dashinfer.allspark.quantization import *
from .quantization_utils import *


class Qwen_v10(Model):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super().__init__("Qwen_v10", data_type, **kwargs)
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

        self.weight_std_names = [
            # globals
            "wte.weight",  #0
            "position_embeddings",
            "ln_f.weight",  #2
            "final_layernorm.bias",
            # layers
            "ln_1.weight",  #4
            "input_layernorm.bias",
            "attn.c_attn.weight",  #6
            "attn.c_attn.bias",
            "attn.c_proj.weight",  #8
            "dense.bias",
            "ln_2.weight",  #10
            "post_attention_layernorm.bias",
            "w1.weight",  #12
            "w1.bias",
            "w2.weight",  #14
            "w2.bias",
            "c_proj.weight",  #16
            "dense_4h_to_h.bias",
            "rotary_emb.inv_freq",
        ]
        self.pattern_rules = {
            0: r"\btransformer\.%s\b",
            4: r"\btransformer\.h\.\d+\..*\b%s\b",
        }
        self._build_graph(self.model_config, derive_type)

        start_time = time.time()
        self._trans_weight_hf(self.model_config, torch_model)

        self._trans_weight(torch_model)
        print("parse weight time: ", time.time() - start_time)

    def _trans_weight_hf(self, torch_cfg, torch_model):

        def fix_query_key_value_ordering(key, param, num_splits, num_heads,
                                         hidden_size):
            # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
            # for compatibility with later versions of NVIDIA Megatron-LM.
            # The inverse operation is performed inside Megatron-LM to read checkpoints:
            # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
            # If param is the weight tensor of the self-attention block, the returned tensor
            # will have to be transposed one more time to be read by HuggingFace GPT2.
            # print("================= {} ====================".format(key))
            input_shape = param.size()
            # print(f"input_shape: {input_shape}")
            # other versions store [num_heads * num_splits * hidden_size, :]
            saved_shape = (num_splits, num_heads,
                           hidden_size) + input_shape[1:]
            param = param.view(*saved_shape)
            param = param.transpose(0, 1).contiguous()
            param = param.view(*input_shape)
            return param

        num_heads = torch_cfg.get('num_attention_heads', 0)
        hidden_size_per_head = torch_cfg.get('size_per_head', 128)

        new_model = torch_model
        for key, val in torch_model.items():
            # print(key, val.size())
            if (key.find("attn.c_attn") != -1) and (key.endswith("weight")
                                                    or key.endswith("bias")):
                out_val = fix_query_key_value_ordering(key, val, 3, num_heads,
                                                       hidden_size_per_head)
                new_model[key] = out_val
            else:
                continue
        return new_model

    def _build_graph(self, torch_cfg, derive_type):
        cfg = self.model.model_conf
        cfg.ln_eps = torch_cfg.get('layernorm_epsilon', 1e-6)
        cfg.num_heads = torch_cfg.get('num_attention_heads', 0)
        cfg.dec_layer = torch_cfg.get('num_hidden_layers', 0)
        cfg.size_per_head = torch_cfg.get('size_per_head', 128)
        cfg.dtype = self.dtype
        cfg.is_generate = self.is_generate
        weight_std_names = self.weight_std_names

        self.name_adapter = WeightNameAdapter(weight_std_names,
                                              self.weight_real_names,
                                              self.pattern_rules)
        self.weight_name_map = {
            "embedding.word_embeddings":
            self.name_adapter.fullname(weight_std_names[0]),
            "final.layernorm.gamma":
            self.name_adapter.fullname(weight_std_names[2]),
        }
        decoder_name_map = {
            "attention.layernorm.gamma": weight_std_names[4],
            "attention.self.weight": weight_std_names[6],
            "attention.self.bias": weight_std_names[7],
            "attention.output.dense.weight": weight_std_names[8],
            "ffn.layernorm.gamma": weight_std_names[10],
            "ffn.intermediate.dense1.weight": weight_std_names[12],
            "ffn.intermediate.dense2.weight": weight_std_names[14],
            "ffn.output.dense.weight": weight_std_names[16],
        }
        for i in range(cfg.dec_layer):
            for key in decoder_name_map:
                self.weight_name_map["decoder.layer.{}.{}".format(
                    i, key)] = self.name_adapter.fullname(
                        decoder_name_map[key]).format(i)
        if self.multinode_mode != 0:
            self.split_map = {}
            self.split_map["embedding.word_embeddings"] = VSPLIT
            self.split_map["embedding.word_embeddings_H"] = HSPLIT
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)
                self.split_map[prefix + "attention.self.weight"] = QKVSPLIT
                self.split_map[prefix + "attention.self.bias"] = QKVSPLIT
                self.split_map[prefix +
                               "attention.output.dense.weight"] = HSPLIT
                self.split_map[prefix + "attention.output.dense.bias"] = HSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense1.weight"] = VSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense1.bias"] = VSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense2.weight"] = VSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense2.bias"] = VSPLIT
                self.split_map[prefix + "ffn.output.dense.weight"] = HSPLIT
                self.split_map[prefix + "ffn.output.dense.bias"] = HSPLIT
        if self.do_dynamic_quantize_convert:
            if self.quant_config != None:
                if self.quant_config.quantize_mode in [
                        QuantizeConfig.QuantMode.A16W8
                ]:
                    self.quantize_map = {}
                    for i in range(cfg.dec_layer):
                        prefix = "decoder.layer.{}.".format(i)
                        self.quantize_map[prefix + "attention.self.weight"] = 1
                        self.quantize_map[prefix +
                                          "attention.output.dense.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.intermediate.dense1.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.intermediate.dense2.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.output.dense.weight"] = 1
                else:
                    raise ValueError("not support quantize_mode",
                                     (str(self.quant_config.quantize_mode)))
            else:
                raise ValueError("not find quant_config")
        do_binary_add_fused = self.do_binary_add_fused
        if self.do_dynamic_quantize_convert:
            do_binary_add_fused = False
        ##############################################################################################
        self.model.graph_names.extend(["decoder"])
        graph = self.model.graphs["decoder"]
        mask = TransMask(
            "transmask",
            self.model.inputs[1],
            {"sequence_mask": True},
        )()
        embedding = EmbeddingT5("embedding", self.model.inputs[0],
                                {"token_embedding": False})()
        graph.ops.extend([mask, embedding])
        if self.multinode_mode != 0:
            all_gather_embedding = AllGather("all_gather_embedding",
                                             embedding.outputs[0])()
            graph.ops.append(all_gather_embedding)
            rich_embedding = RichEmbedding(
                "rich_embedding",
                [self.model.inputs[0], all_gather_embedding.outputs[0]])()
            graph.ops.append(rich_embedding)
        else:
            rich_embedding = RichEmbedding(
                "rich_embedding",
                [self.model.inputs[0], embedding.outputs[0]])()
            graph.ops.append(rich_embedding)

        for i in range(cfg.dec_layer):
            prefix = "decoder.layer.{}.".format(i)
            # attention
            first_ln = LayerNormNoBeta(
                prefix + "attention.layernorm",
                graph.ops[-1].outputs[0],
                {"eps": cfg.ln_eps},
            )()
            attn_self_gemm = self.make_gemm_op(prefix + "attention.self",
                                               first_ln.outputs[0])()
            rotary_attributes = {"num_heads": cfg.num_heads}
            if self.use_dynamic_ntk and hasattr(self, "model_sequence_length"):
                rotary_attributes["ntk_model_embed"] = int(
                    self.model_sequence_length)
            if hasattr(self, "seqlen_extrapolation"):
                rotary_attributes["seqlen_extrapolation"] = float(
                    self.seqlen_extrapolation)
            if hasattr(self, "rotary_base"):
                rotary_attributes["rotary_base"] = float(self.rotary_base)
            if self.use_logn_attn and hasattr(self, "model_sequence_length"):
                rotary_attributes["logn_model_embedding"] = int(
                    self.model_sequence_length)
            rotary_embedding = Rotary(
                prefix + "rotary",
                [attn_self_gemm.outputs[0], mask.outputs[1]],
                rotary_attributes,
            )()
            mha_attribtues = {"num_heads": cfg.num_heads}
            mha = MultiHeadAttention(
                prefix + "attention",
                [rotary_embedding.outputs[0], mask.outputs[0]],
                mha_attribtues,
            )()
            attn_op_list = []
            ffn_op_list = []
            ffn_ln = None
            if do_binary_add_fused:
                attn_out_gemm = self.make_gemm_op(
                    prefix + "attention.output.dense",
                    [mha.outputs[0], graph.ops[-1].outputs[0]], {
                        "with_bias": False,
                        "binary_type": ADD
                    })()
                attn_op_list = [
                    first_ln, attn_self_gemm, rotary_embedding, mha,
                    attn_out_gemm
                ]
                # ffn
                ffn_ln = LayerNormNoBeta(prefix + "ffn.layernorm",
                                         attn_out_gemm.outputs[0],
                                         {"eps": cfg.ln_eps})()
            else:
                attn_out_gemm = self.make_gemm_op(
                    prefix + "attention.output.dense", mha.outputs[0],
                    {"with_bias": False})()
                attn_add = Binary(
                    prefix + "attention_add",
                    [attn_out_gemm.outputs[0], graph.ops[-1].outputs[0]],
                    {"binary_type": ADD})()
                attn_op_list = [
                    first_ln, attn_self_gemm, rotary_embedding, mha,
                    attn_out_gemm, attn_add
                ]
                # ffn
                ffn_ln = LayerNormNoBeta(prefix + "ffn.layernorm",
                                         attn_add.outputs[0],
                                         {"eps": cfg.ln_eps})()
            ffn_intermediate1 = self.make_gemm_op(
                prefix + "ffn.intermediate.dense1", ffn_ln.outputs[0],
                {"with_bias": False})()
            ffn_intermediate2 = self.make_gemm_op(
                prefix + "ffn.intermediate.dense2", ffn_ln.outputs[0],
                {"with_bias": False})()
            ffn_intermediate_geglu = Binary(
                prefix + "ffn_geglu",
                [ffn_intermediate1.outputs[0], ffn_intermediate2.outputs[0]],
                {"binary_type": SWIGLU},
            )()
            if do_binary_add_fused:
                ffn_out = self.make_gemm_op(prefix + "ffn.output.dense", [
                    ffn_intermediate_geglu.outputs[0], attn_out_gemm.outputs[0]
                ], {
                    "binary_type": ADD,
                    "with_bias": False
                })()
                ffn_op_list = [
                    ffn_ln, ffn_intermediate1, ffn_intermediate2,
                    ffn_intermediate_geglu, ffn_out
                ]
                if self.multinode_mode != 0:
                    all_reduce_attention = AllReduce(
                        prefix + "attention.all_reduce_attention",
                        attn_out_gemm.outputs[0])()
                    all_reduce_ffn = AllReduce(
                        prefix + "attention.all_reduce_ffn",
                        ffn_out.outputs[0])()
                    attn_op_list.append(all_reduce_attention)
                    ffn_op_list.append(all_reduce_ffn)
            else:
                ffn_out = self.make_gemm_op(prefix + "ffn.output.dense",
                                            ffn_intermediate_geglu.outputs[0],
                                            {"with_bias": False})()
                final_add = Binary(
                    prefix + "final_add",
                    [ffn_out.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [
                    ffn_ln, ffn_intermediate1, ffn_intermediate2,
                    ffn_intermediate_geglu, ffn_out, final_add
                ]
                if self.multinode_mode != 0:
                    all_reduce_attention = AllReduce(
                        prefix + "attention.all_reduce_attention",
                        attn_out_gemm.outputs[0])()
                    all_reduce_ffn = AllReduce(
                        prefix + "attention.all_reduce_ffn",
                        ffn_out.outputs[0])()
                    attn_op_list.insert(-1, all_reduce_attention)
                    ffn_op_list.insert(-1, all_reduce_ffn)
            # final
            graph.ops.extend(attn_op_list + ffn_op_list)

        #deocder over
        final_layernorm = LayerNormNoBeta("final.layernorm",
                                          graph.ops[-1].outputs[0],
                                          {"eps": cfg.ln_eps})()
        graph.ops.append(final_layernorm)
        graph.ops[-1].outputs[0].name = "last_hidden_state"
        # Quantize
        if self.do_dynamic_quantize_convert:
            for op in graph.ops:
                quantize_op(op, self.quant_config, self.quantize_map)
        ##############################################################################################
        if derive_type == None:
            raise RuntimeError(
                "derive type [{}] is not supported.".format(derive_type))
        elif derive_type == "lmhead":
            self._add_layer("lmhead", graph, graph.ops[-1].outputs[0])
            self.weight_name_map.update({
                "lm_head.weight": "lm_head.weight",
            })
            graph.ops[-1].outputs[0].name = "logits"
            self.model.outputs[0].CopyFrom(graph.ops[-1].outputs[0])
        else:
            raise RuntimeError(
                "derive type [{}] is not supported.".format(derive_type))
        ##############################################################################################
        if self.is_generate:
            self.model.graph_names.insert(0, "pre_graph")
            self.model.graph_names.append("gen_graph")
            gen_graph = self.model.graphs["gen_graph"]
            self.model.graph_names.append("post_graph")
            self.model.outputs[0].CopyFrom(
                make_tensor("generated_ids",
                            np.empty(shape=(0, 0), dtype=np.int64)))
            pre_graph = self.model.graphs["pre_graph"]
            preprocess_ids = PreProcessId(
                "preprocess_id",
                self.model.inputs[0],
            )()
            update_id_first = UpdateId("update_id_first",
                                       preprocess_ids.outputs[0])()
            pre_graph.ops.extend(
                [preprocess_ids, update_id_first, graph.ops[0]])
            del graph.ops[0]
            #########################################################
            for op in graph.ops:
                if op.op_type == "EmbeddingT5":
                    op.inputs[0].CopyFrom(preprocess_ids.outputs[0])
                elif op.op_type == "MultiHeadAttention":
                    op.op_type = "DecOptMHA"
            gen_op = GenerateOp(
                "generate",
                [graph.ops[-1].outputs[0], preprocess_ids.outputs[1]],
            )()
            update_id = UpdateId(
                "update_id", [preprocess_ids.outputs[0], gen_op.outputs[1]])()
            postprocess_ids = PostProcessId(
                "postprocess_id", [update_id.outputs[0], gen_op.outputs[2]])()
            for op in graph.ops:
                if op.op_type == "DecOptMHA":
                    op.inputs.append(gen_op.outputs[1])
            gen_op.outputs[0].CopyFrom(preprocess_ids.outputs[0])
            gen_graph.ops.extend([gen_op, update_id])
            #########################################################
            post_graph = self.model.graphs["post_graph"]
            post_graph.ops.append(postprocess_ids)

    def _trans_weight(self, torch_weight):
        weights_path = self.weights_path
        weight_name_map = self.weight_name_map
        split_map = self.split_map
        sparse_map = self.sparse_map
        quantize_map = self.quantize_map

        self_dtype_str = [
            k for k, v in Model.dtype_dict.items() if v == self.dtype
        ][0]
        for key, torch_name in weight_name_map.items():
            tensor = torch_weight[torch_name]
            start_time = time.time()
            if re.match(r'.*\.attention\.self\.bias', key):
                tensor_reshape = tensor.reshape(
                    self.model.model_conf.num_heads, 3,
                    tensor.shape[0] // 3 // self.model.model_conf.num_heads)
                tensor = torch.permute(tensor_reshape, (1, 0, 2)).reshape(-1)
            if re.match(r'.*\.attention\.self\.weight', key):
                tensor_reshape = tensor.reshape(
                    self.model.model_conf.num_heads, 3, tensor.shape[0] // 3 //
                    self.model.model_conf.num_heads * tensor.shape[1])
                tensor_reshape = torch.permute(tensor_reshape,
                                               (1, 0, 2)).reshape(
                                                   (tensor.shape[0], -1))
                tensor = torch.permute(tensor_reshape, (1, 0)).contiguous()
            elif key.rfind(".weight") != -1:
                tensor = torch.permute(tensor, (1, 0)).contiguous()

            self.validate_weight_dtype(key, tensor, self_dtype_str)
            mode = DENSE if key not in sparse_map else sparse_map[key]
            split_mode = NOSPLIT if key not in split_map else split_map[key]
            quantize_mode = False if key not in quantize_map else quantize_map[
                key]
            if quantize_mode == False:
                save_torch_to_allsparky(weights_path, key, tensor, mode,
                                        split_mode)
            else:
                if self.quant_config.quantize_mode in [
                        QuantizeConfig.QuantMode.A16W8
                ]:
                    # Check whether splitting is possible under the SubChannel settings
                    if (self.quant_config.extra_option["SubChannel"]
                            == True) and (split_mode == HSPLIT):
                        K = tensor.shape[0]
                        group_size = self.quant_config.extra_option[
                            "GroupSize"]
                        if K % group_size != 0:
                            raise ValueError(
                                f"SubChannel: the model is not splittable under current Groupsize value {group_size}. "
                                f"Weight Tensor Name: {key}, Split Mode: HSPLIT, Shape: {tensor.shape}"
                            )
                    qdata, scale, zero_point = quantize_gemm_weigt_a16wX_torch(
                        tensor, self.quant_config)
                    save_torch_to_allsparky(weights_path, key, qdata, mode,
                                            split_mode)
                    if self.quant_config.extra_option[
                            "SubChannel"] == True or split_mode != HSPLIT:
                        save_torch_to_allsparky(weights_path, key + ".scale",
                                                scale, mode, split_mode)
                        save_torch_to_allsparky(weights_path,
                                                key + ".zero_point",
                                                zero_point, mode, split_mode)
                    else:
                        save_torch_to_allsparky(weights_path, key + ".scale",
                                                scale, mode, NOSPLIT)
                        save_torch_to_allsparky(weights_path,
                                                key + ".zero_point",
                                                zero_point, mode, NOSPLIT)
            if (torch_name != "word_embeddings"):
                torch_weight[torch_name] = torch.Tensor(0)
        set_global_header(weights_path)
