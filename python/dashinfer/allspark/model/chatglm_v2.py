#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    chatglm_v2.py
#
from .model_base import *
from .utils import WeightNameAdapter
from dashinfer.allspark.quantization import *
from .quantization_utils import *
import math


class ChatGLM_v2(Model):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super().__init__("ChatGLM_v2", data_type, **kwargs)
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
        self.invfreq_type_ = 1
        self._build_graph(self.model_config, derive_type)
        self._trans_weight(torch_model)
        start_time = time.time()
        print("parse weight time: ", time.time() - start_time)

    def _build_graph(self, torch_cfg, derive_type):
        cfg = self.model.model_conf
        cfg.ln_eps = torch_cfg.get('layernorm_epsilon', 1e-5)
        cfg.num_heads = torch_cfg.get('num_attention_heads', 12)
        cfg.dec_layer = torch_cfg.get('num_layers', 12)
        hidden_size_ = torch_cfg.get('hidden_size', 4096)
        cfg.kv_channels = torch_cfg.get('kv_channels', 128)
        cfg.multi_query_group_num = torch_cfg.get('multi_query_group_num', 2)
        rope_ratio_ = float(torch_cfg.get('rope_ratio', 1))
        cfg.is_generate = self.is_generate

        cfg.size_per_head = torch_cfg.get('size_per_head', 128)
        cfg.dtype = self.dtype

        weight_std_names = [
            # globals
            "word_embeddings",  #0
            "rotary_pos_emb.inv_freq",
            "final_layernorm.weight",  #2
            "final_layernorm.bias",
            # layers
            "input_layernorm.weight",  #4
            "input_layernorm.bias",
            "query_key_value.weight",  #6
            "query_key_value.bias",
            "dense.weight",  #8
            "dense.bias",
            "post_attention_layernorm.weight",  #10
            "post_attention_layernorm.bias",
            "dense_h_to_4h.weight",
            "dense_h_to_4h.bias",
            "dense_4h_to_h.weight",
            "dense_4h_to_h.bias",
        ]

        self.name_adapter = WeightNameAdapter(weight_std_names,
                                              self.weight_real_names)
        self.weight_name_map = {
            "embedding.word_embeddings":
            self.name_adapter.fullname(weight_std_names[0]),
            # "embedding.position_embeddings": self.name_adapter.fullname(weight_std_names[1]),
            "final.layernorm.gamma":
            self.name_adapter.fullname(weight_std_names[2]),
            # "final.layernorm.beta": self.name_adapter.fullname(weight_std_names[3]),
        }
        decoder_name_map = {
            "attention.layernorm.gamma": weight_std_names[4],
            # "attention.layernorm.beta": weight_std_names[5],
            "attention.self.weight": weight_std_names[6],
            "attention.self.bias": weight_std_names[7],
            "attention.output.dense.weight": weight_std_names[8],
            # "attention.output.dense.bias": weight_std_names[9],
            "ffn.layernorm.gamma": weight_std_names[10],
            # "ffn.layernorm.beta": weight_std_names[11],
            "ffn.intermediate.dense.weight": weight_std_names[12],
            # "ffn.intermediate.dense.bias": weight_std_names[13],
            "ffn.output.dense.weight": weight_std_names[14],
            # "ffn.output.dense.bias": weight_std_names[15],
            "rotary.inv_freq": weight_std_names[1],
        }
        for i in range(cfg.dec_layer):
            for key in decoder_name_map:
                self.weight_name_map["decoder.layer.{}.{}".format(
                    i, key
                )] = self.name_adapter.fullname(decoder_name_map[key]).format(
                    i)  #"layers.{}.{}".format(i, decoder_name_map[key])
        if self.multinode_mode != 0:
            self.split_map = {}
            self.split_map["embedding.word_embeddings"] = VSPLIT
            self.split_map["embedding.word_embeddings_H"] = HSPLIT
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)

                self.split_map[prefix + "attention.self.weight"] = GROUP_VSPLIT
                self.split_map[prefix + "attention.self.bias"] = GROUP_VSPLIT
                self.split_map[prefix +
                               "attention.output.dense.weight"] = HSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense.weight"] = VSPLIT
                self.split_map[prefix + "ffn.output.dense.weight"] = VSPLIT
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
                                          "ffn.intermediate.dense.weight"] = 1
                        self.quantize_map[prefix +
                                          "ffn.output.dense.weight"] = 1
                else:
                    raise ValueError("not support quantize_mode",
                                     (str(self.quant_config.quantize_mode)))
            else:
                raise ValueError("not find quant_config")

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

            rotary_embedding = RotaryMulQuery(
                prefix + "rotary",
                [attn_self_gemm.outputs[0], mask.outputs[1]],
                {
                    "num_heads": cfg.num_heads,
                    "multi_query_group_num": cfg.multi_query_group_num,
                    "rope_ratio": rope_ratio_,
                    "hidden_size": hidden_size_,
                    "use_weight": False,
                    "invfreq_type": self.invfreq_type_,
                    "rotary_type": 3
                },
            )()

            mqa = MultiQueryAttention(
                prefix + "attention",
                [rotary_embedding.outputs[0], mask.outputs[0]],
                {
                    "num_heads": cfg.num_heads,
                    "hidden_size": hidden_size_,
                    "kv_channels": cfg.kv_channels,
                    "multi_query_group_num": cfg.multi_query_group_num
                },
            )()
            attn_out_gemm = self.make_gemm_op(
                prefix + "attention.output.dense", mqa.outputs[0],
                {"with_bias": False})()

            if self.multinode_mode == 0:
                attn_add = Binary(
                    prefix + "attention_add",
                    [attn_out_gemm.outputs[0], graph.ops[-1].outputs[0]],
                    {"binary_type": ADD},
                )()
                attn_op_list = [
                    first_ln, attn_self_gemm, rotary_embedding, mqa,
                    attn_out_gemm, attn_add
                ]
            else:
                all_reduce_attention = AllReduce(
                    prefix + "attention.all_reduce_attention",
                    attn_out_gemm.outputs[0])()

                attn_add = Binary(
                    prefix + "attention_add",
                    [
                        all_reduce_attention.outputs[0],
                        graph.ops[-1].outputs[0]
                    ],
                    {"binary_type": ADD},
                )()
                attn_op_list = [
                    first_ln, attn_self_gemm, rotary_embedding, mqa,
                    attn_out_gemm, all_reduce_attention, attn_add
                ]

            # ffn
            ffn_ln = LayerNormNoBeta(prefix + "ffn.layernorm",
                                     attn_add.outputs[0],
                                     {"eps": cfg.ln_eps})()
            ffn_intermediate = self.make_gemm_op(
                prefix + "ffn.intermediate.dense",
                ffn_ln.outputs[0],
                {"with_bias": False},
            )()

            if self.multinode_mode == 0:
                ffn_chunk = ChunkBinary(prefix + "ffn.chunk",
                                        ffn_intermediate.outputs[0],
                                        {"binary_type": SWIGLU})()
                ffn_out = self.make_gemm_op(prefix + "ffn.output.dense",
                                            ffn_chunk.outputs[0],
                                            {"with_bias": False})()
                final_add = Binary(
                    prefix + "final_add",
                    [ffn_out.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [
                    ffn_ln, ffn_intermediate, ffn_chunk, ffn_out, final_add
                ]
            else:
                all_gather_ffn_intermediate = AllGather(
                    prefix + "all_gather_ffn_intermediate",
                    ffn_intermediate.outputs[0])()
                ffn_chunk = ChunkBinary(prefix + "ffn.chunk",
                                        all_gather_ffn_intermediate.outputs[0],
                                        {"binary_type": SWIGLU})()
                ffn_out = self.make_gemm_op(prefix + "ffn.output.dense",
                                            ffn_chunk.outputs[0],
                                            {"with_bias": False})()
                all_gather_ffn_out = AllGather(prefix + "all_gather_ffn_out",
                                               ffn_out.outputs[0])()
                final_add = Binary(
                    prefix + "final_add",
                    [all_gather_ffn_out.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [
                    ffn_ln, ffn_intermediate, all_gather_ffn_intermediate,
                    ffn_chunk, ffn_out, all_gather_ffn_out, final_add
                ]

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
            pass
        elif derive_type == "lmhead":
            self._add_layer("lmhead", graph, graph.ops[-1].outputs[0])
            self.weight_name_map.update({
                "lm_head.weight":
                "transformer.output_layer.weight",
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
                elif op.op_type == "MultiQueryAttention":
                    op.op_type = "DecOptMQA"

            gen_op = GenerateOp(
                "generate",
                [graph.ops[-1].outputs[0], preprocess_ids.outputs[1]],
            )()
            update_id = UpdateId(
                "update_id", [preprocess_ids.outputs[0], gen_op.outputs[1]])()
            postprocess_ids = PostProcessId(
                "postprocess_id", [update_id.outputs[0], gen_op.outputs[2]])()
            for op in graph.ops:
                if op.op_type == "DecOptMHA" or op.op_type == "DecOptMQA":
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
            tensor = torch_weight[torch_name].cpu()
            # print("trans_tensor: {}, {}".format(key, torch_name))
            start_time = time.time()
            if key.find("weight") != -1:
                tensor = torch.permute(tensor, (1, 0)).contiguous()
            if str(tensor.dtype) != "torch." + self_dtype_str:
                raise ValueError(
                    "DataType not match, [weight dtype: {}] vs [model dtype:{}]"
                    .format(str(tensor.dtype), "torch." + self_dtype_str))
            mode = DENSE if key not in sparse_map else sparse_map[key]
            split_mode = NOSPLIT if key not in split_map else split_map[key]
            if split_mode != GROUP_VSPLIT:
                group_list = []
            else:
                group_list = [
                    self.model.model_conf.num_heads *
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.kv_channels
                ]
            quantize_mode = False if key not in quantize_map else quantize_map[
                key]
            if quantize_mode == False:
                save_torch_to_allsparky(weights_path, key, tensor, mode,
                                        split_mode, group_list)
            else:
                if self.quant_config.quantize_mode == QuantizeConfig.QuantMode.Dynamic:
                    raise ValueError(
                        "QuantizeConfig.QuantMode.Dynamic not support now")
                elif self.quant_config.quantize_mode == QuantizeConfig.QuantMode.A16W8:
                    qdata, scale, zero_point = quantize_gemm_weight_a16w8_torch(
                        tensor, self.quant_config)
                    save_torch_to_allsparky(weights_path, key, qdata, mode,
                                            split_mode, group_list)
                    if self.quant_config.extra_option[
                            "SubChannel"] == True or split_mode != HSPLIT:
                        save_torch_to_allsparky(weights_path, key + ".scale",
                                                scale, mode, split_mode,
                                                group_list)
                        save_torch_to_allsparky(weights_path,
                                                key + ".zero_point",
                                                zero_point, mode, split_mode,
                                                group_list)
                    else:
                        save_torch_to_allsparky(weights_path, key + ".scale",
                                                scale, mode, NOSPLIT,
                                                group_list)
                        save_torch_to_allsparky(weights_path,
                                                key + ".zero_point",
                                                zero_point, mode, NOSPLIT,
                                                group_list)
            if torch_name not in [
                    "word_embeddings", "transformer.rotary_pos_emb.inv_freq"
            ]:
                torch_weight[torch_name] = torch.Tensor(0)
        set_global_header(weights_path)
