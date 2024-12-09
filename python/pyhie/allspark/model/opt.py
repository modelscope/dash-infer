'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    opt.py
'''
from .model_base import *
from .utils import WeightNameAdapter, quantiza_data_by_numpy


class OPT(Model):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super().__init__("OPT", data_type, **kwargs)
        self.model.inputs.append(
            make_tensor("input_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.model.inputs.append(
            make_tensor("attention_mask", np.empty(shape=(0, 0),
                                                   dtype=np.int64)))
        self.model.outputs.append(make_tensor("last_hidden_state"))
        self.is_generate = True

        self.weight_real_names = set()
        for v in torch_model:
            self.weight_real_names.add(v)
        self._build_graph(self.model_config, derive_type)
        if not self.only_convert_lora:
            self._trans_weight(torch_model)

    def _build_graph(self, torch_cfg, derive_type):
        cfg = self.model.model_conf

        cfg.ln_eps = torch_cfg.get('layernorm_epsilon', 1e-5)
        cfg.num_heads = torch_cfg.get('num_attention_heads', 12)
        cfg.dec_layer = torch_cfg.get('num_hidden_layers', 12)
        cfg.activation = get_activation(
            torch_cfg.get('activation_function', "RELU"))
        cfg.is_generate = self.is_generate

        weight_std_names = [
            # globals
            "embed_tokens",
            "embed_positions",
            # layers
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "q_proj.bias",
            "k_proj.bias",
            "v_proj.bias",
            "out_proj.weight",
            "out_proj.bias",
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
        ]

        name_adapter = WeightNameAdapter(weight_std_names,
                                         self.weight_real_names,
                                         pattern_rules={
                                             0: r"\b%s\b",
                                             2: r"\blayers\.\d+\..*\b%s\b",
                                         })
        self.weight_name_map = {
            "embedding.word_embeddings":
            name_adapter.fullname(weight_std_names[0]),
            "embedding.position_embeddings":
            name_adapter.fullname(weight_std_names[1]),
            "final.layernorm.gamma":
            "model.decoder.final_layer_norm.weight",
            "final.layernorm.beta":
            "model.decoder.final_layer_norm.bias",
        }
        decoder_name_map = {
            "attention.layernorm.gamma":
            weight_std_names[10],
            "attention.layernorm.beta":
            weight_std_names[11],
            "attention.self.weight":
            [weight_std_names[2], weight_std_names[3], weight_std_names[4]],
            "attention.self.bias":
            [weight_std_names[5], weight_std_names[6], weight_std_names[7]],
            "attention.output.dense.weight":
            weight_std_names[8],
            "attention.output.dense.bias":
            weight_std_names[9],
            "ffn.layernorm.gamma":
            weight_std_names[16],
            "ffn.layernorm.beta":
            weight_std_names[17],
            "ffn.intermediate.dense.weight":
            weight_std_names[12],
            "ffn.intermediate.dense.bias":
            weight_std_names[13],
            "ffn.output.dense.weight":
            weight_std_names[14],
            "ffn.output.dense.bias":
            weight_std_names[15],
        }
        for i in range(cfg.dec_layer):
            for key in decoder_name_map:
                real_name = decoder_name_map[key]
                if isinstance(real_name, list):
                    self.weight_name_map["decoder.layer.{}.{}".format(
                        i, key)] = [(name_adapter.fullname(v).format(i))
                                    for v in real_name]
                else:
                    self.weight_name_map["decoder.layer.{}.{}".format(
                        i, key)] = name_adapter.fullname(real_name).format(i)
        if self.multigpu_mode != 0:
            self.split_map = {}
            self.split_map["embedding.word_embeddings"] = VSPLIT
            self.split_map["embedding.position_embeddings"] = VSPLIT
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)
                self.split_map[prefix + "attention.self.weight"] = QKVSPLIT
                self.split_map[prefix + "attention.self.bias"] = QKVSPLIT
                self.split_map[prefix +
                               "attention.output.dense.weight"] = HSPLIT
                self.split_map[prefix + "attention.output.dense.bias"] = HSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense.weight"] = VSPLIT
                self.split_map[prefix + "ffn.intermediate.dense.bias"] = VSPLIT
                self.split_map[prefix + "ffn.output.dense.weight"] = HSPLIT
                self.split_map[prefix + "ffn.output.dense.bias"] = HSPLIT
        if self.do_dynamic_quantize_convert:
            self.quantize_map = {}
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)
                self.quantize_map[prefix + "attention.self.weight"] = 1
                self.quantize_map[prefix + "attention.output.dense.weight"] = 1
                self.quantize_map[prefix + "ffn.intermediate.dense.weight"] = 1
                self.quantize_map[prefix + "ffn.output.dense.weight"] = 1
        ##############################################################################################
        self.model.graph_names.extend(["decoder"])
        graph = self.model.graphs["decoder"]
        mask = TransMask(
            "transmask",
            self.model.inputs[1],
            {"sequence_mask": True},
        )()
        embedding = Embedding("embedding",
                              [self.model.inputs[0], mask.outputs[1]],
                              {"offset": 2})()
        graph.ops.extend([mask, embedding])
        if self.multigpu_mode != 0:
            all_gather_embedding = AllGather("all_gather_embedding",
                                             embedding.outputs[0])()
            graph.ops.append(all_gather_embedding)
        for i in range(cfg.dec_layer):
            prefix = "decoder.layer.{}.".format(i)
            # attention
            first_ln = LayerNorm(
                prefix + "attention.layernorm",
                graph.ops[-1].outputs[0],
                {"eps": cfg.ln_eps},
            )()
            attn_self_gemm = Gemm(prefix + "attention.self",
                                  first_ln.outputs[0])()
            mha = MultiHeadAttention(
                prefix + "attention",
                [attn_self_gemm.outputs[0], mask.outputs[0]],
                {"num_heads": cfg.num_heads},
            )()
            attn_out_gemm = Gemm(prefix + "attention.output.dense",
                                 mha.outputs[0])()
            attn_add = Binary(
                prefix + "attention_add",
                [attn_out_gemm.outputs[0], graph.ops[-1].outputs[0]],
                {"binary_type": ADD},
            )()
            attn_op_list = [
                first_ln, attn_self_gemm, mha, attn_out_gemm, attn_add
            ]
            # ffn
            ffn_ln = LayerNorm(prefix + "ffn.layernorm", attn_add.outputs[0],
                               {"eps": cfg.ln_eps})()
            ffn_intermediate = Gemm(
                prefix + "ffn.intermediate.dense",
                ffn_ln.outputs[0],
                {"activation": cfg.activation},
            )()
            ffn_out = Gemm(prefix + "ffn.output.dense",
                           ffn_intermediate.outputs[0])()
            final_add = Binary(
                prefix + "final_add",
                [ffn_out.outputs[0], attn_add.outputs[0]],
                {"binary_type": ADD},
            )()
            ffn_op_list = [ffn_ln, ffn_intermediate, ffn_out, final_add]
            if self.multigpu_mode != 0:
                all_reduce_attention = AllReduce(
                    prefix + "attention.all_reduce_attention",
                    attn_out_gemm.outputs[0])()
                all_reduce_ffn = AllReduce(prefix + "attention.all_reduce_ffn",
                                           ffn_out.outputs[0])()
                attn_op_list.insert(-1, all_reduce_attention)
                ffn_op_list.insert(-1, all_reduce_ffn)
            # final
            graph.ops.extend(attn_op_list + ffn_op_list)
        final_layernorm = LayerNorm("final.layernorm",
                                    graph.ops[-1].outputs[0],
                                    {"eps": cfg.ln_eps})()
        graph.ops.append(final_layernorm)
        graph.ops[-1].outputs[0].name = "last_hidden_state"
        if self.do_dynamic_quantize_convert:
            for op in graph.ops:
                if op.op_type == "Gemm":
                    op.op_type = "DynamicQuantizeMatmul"
                    op.weights.insert(
                        1, make_tensor(op.op_name + ".weight.scale"))
                    op.weights.insert(
                        2, make_tensor(op.op_name + ".weight.zero_point"))
                    op.weights.insert(
                        3, make_tensor(op.op_name + ".weight.redsum"))
        ##############################################################################################
        if derive_type == None:
            return
        elif derive_type == "lmhead":
            # for k, v in self.weight_name_map.items():
            #     self.weight_name_map[k] = "transformer." + v
            self._add_layer("lmhead", graph, graph.ops[-1].outputs[0])
            self.weight_name_map.update({
                "lm_head.weight":
                name_adapter.fullname(weight_std_names[0]),
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
                if op.op_type == "Embedding":
                    op.op_type = "DecOptEmbedding"
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
        self_dtype_str = [
            k for k, v in Model.dtype_dict.items() if v == self.dtype
        ][0]
        for key, torch_name in self.weight_name_map.items():
            if isinstance(torch_name, list):  # attention qkv weights
                npy_tensor = (torch.concat(
                    [torch_weight[name] for name in torch_name]).cpu().numpy())
            else:
                npy_tensor = torch_weight[torch_name].cpu().numpy()
            if key.find("weight") != -1:
                npy_tensor = np.copy(npy_tensor.transpose(1, 0), order="C")
            if str(npy_tensor.dtype) != self_dtype_str:
                raise ValueError(
                    "DataType not match, [weight dtype: {}] vs [model dtype:{}]"
                    .format(str(npy_tensor.dtype), self_dtype_str))
            mode = DENSE
            split_mode = NOSPLIT if key not in self.split_map else self.split_map[
                key]
            quantize_mode = False if key not in self.quantize_map else self.quantize_map[
                key]
            if quantize_mode == False:
                self.weights[key] = trans_to_allsparkz(npy_tensor, mode,
                                                       split_mode)
            else:
                qdata, scale, zero_point, redsum = quantiza_data_by_numpy(
                    npy_tensor, "int8", -2)
                self.weights[key] = trans_to_allsparkz(qdata, mode, split_mode)
                self.weights[key + ".scale"] = trans_to_allsparkz(
                    scale, mode, split_mode)
                self.weights[key + ".zero_point"] = trans_to_allsparkz(
                    zero_point, mode, split_mode)
                self.weights[key + ".redsum"] = trans_to_allsparkz(
                    redsum, mode, split_mode)
