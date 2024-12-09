'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    starcoder.py
'''
from .model_base import *
from .utils import WeightNameAdapter
from ..quantization import *
from .quantization_utils import *
import math


class StarCoder(Model):

    def __init__(self, torch_model, data_type, derive_type, **kwargs):
        super().__init__("StarCoder", data_type, **kwargs)
        self.model.inputs.append(
            make_tensor("input_ids", np.empty(shape=(0, 0), dtype=np.int64)))
        self.model.inputs.append(
            make_tensor("attention_mask", np.empty(shape=(0, 0),
                                                   dtype=np.int64)))
        self.model.outputs.append(make_tensor("last_hidden_state"))
        self.is_generate = kwargs.get("is_generate", True)
        self.weight_real_names = set()
        for v in torch_model:
            self.weight_real_names.add(v)
        self._build_graph(self.model_config, derive_type)
        start_time = time.time()
        if not self.only_convert_lora:
            self._trans_weight(torch_model)
        print("parse weight time: ", time.time() - start_time)

    def _build_graph(self, torch_cfg, derive_type):
        cfg = self.model.model_conf
        cfg.ln_eps = torch_cfg.get("layernorm_epsilon", 1e-5)
        cfg.num_heads = torch_cfg.get("n_head", 48)
        cfg.dec_layer = torch_cfg.get("n_layer", 40)
        hidden_size_ = torch_cfg.get("n_embd", 6144)
        cfg.kv_channels = torch_cfg.get("kv_channels", 128)
        cfg.kv_channels = hidden_size_ // cfg.num_heads
        cfg.multi_query_group_num = torch_cfg.get("multi_query_group_num", 1)
        cfg.is_generate = self.is_generate
        cfg.data_mode = self.data_mode
        weight_std_names = [
            # globals
            "wte.weight",
            "wpe.weight",
            "ln_f.weight",
            "ln_f.bias",
            # layers
            "ln_1.weight",  #4
            "ln_1.bias",
            "attn.c_attn.weight",  #6
            "attn.c_attn.bias",
            "attn.c_proj.weight",  #8
            "attn.c_proj.bias",
            "ln_2.weight",  #10
            "ln_2.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
            # final
            "lm_head.weight",
        ]

        self.name_adapter = WeightNameAdapter(
            weight_std_names,
            self.weight_real_names,
            pattern_rules={
                0: r"\b%s\b",
                4: r"\bh\.\d+\..*\b%s\b",
                16: r"\b%s\b",
            },
        )
        self.weight_name_map = {
            "embedding.word_embeddings":
            self.name_adapter.fullname(weight_std_names[0]),
            "embedding.position_embeddings":
            self.name_adapter.fullname(weight_std_names[1]),
            "final.layernorm.gamma":
            self.name_adapter.fullname(weight_std_names[2]),
            "final.layernorm.beta":
            self.name_adapter.fullname(weight_std_names[3]),
        }
        decoder_name_map = {
            "attention.layernorm.gamma": weight_std_names[4],
            "attention.layernorm.beta": weight_std_names[5],
            "attention.self.weight": weight_std_names[6],
            "attention.self.bias": weight_std_names[7],
            "attention.output.dense.weight": weight_std_names[8],
            "attention.output.dense.bias": weight_std_names[9],
            "ffn.layernorm.gamma": weight_std_names[10],
            "ffn.layernorm.beta": weight_std_names[11],
            "ffn.intermediate.dense.weight": weight_std_names[12],
            "ffn.intermediate.dense.bias": weight_std_names[13],
            "ffn.output.dense.weight": weight_std_names[14],
            "ffn.output.dense.bias": weight_std_names[15],
        }
        for i in range(cfg.dec_layer):
            for key in decoder_name_map:
                self.weight_name_map["decoder.layer.{}.{}".format(
                    i, key
                )] = self.name_adapter.fullname(decoder_name_map[key]).format(
                    i)  # "layers.{}.{}".format(i, decoder_name_map[key])
        if self.multigpu_mode != 0:
            self.split_map = {}
            self.split_map["embedding.word_embeddings"] = VSPLIT
            self.split_map["embedding.word_embeddings_H"] = HSPLIT
            self.split_map["embedding.position_embeddings"] = VSPLIT
            for i in range(cfg.dec_layer):
                prefix = "decoder.layer.{}.".format(i)
                self.split_map[prefix + "attention.self.weight"] = MQA_VSPLIT
                self.split_map[prefix + "attention.self.bias"] = MQA_VSPLIT
                self.split_map[prefix +
                               "attention.output.dense.weight"] = HSPLIT
                self.split_map[prefix + "attention.output.dense.bias"] = HSPLIT
                self.split_map[prefix +
                               "ffn.intermediate.dense.weight"] = VSPLIT
                self.split_map[prefix + "ffn.intermediate.dense.bias"] = VSPLIT
                self.split_map[prefix + "ffn.output.dense.weight"] = HSPLIT
                self.split_map[prefix + "ffn.output.dense.bias"] = HSPLIT
        if self.do_dynamic_quantize_convert:
            if self.quant_config != None:
                if self.quant_config.quantize_mode in [
                        QuantizeConfig.QuantMode.A16W8,
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
                    raise ValueError(
                        "not support quantize_mode",
                        (str(self.quant_config.quantize_mode)),
                    )
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
        embedding = Embedding("embedding",
                              [self.model.inputs[0], mask.outputs[1]])()
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
            if self.multigpu_mode == 0:
                attn_self_gemm = Gemm(prefix + "attention.self",
                                      first_ln.outputs[0])()
                mha = MultiQueryAttention(
                    prefix + "attention",
                    [attn_self_gemm.outputs[0], mask.outputs[0]],
                    {
                        "num_heads": cfg.num_heads,
                        "hidden_size": hidden_size_,
                        "kv_channels": cfg.kv_channels,
                        "multi_query_group_num": 1,
                    },
                )()
                attn_out_gemm = Gemm(prefix + "attention.output.dense",
                                     mha.outputs[0])()
                attn_add = Binary(
                    prefix + "attention_add",
                    [attn_out_gemm.outputs[0], graph.ops[-1].outputs[0]],
                    {"binary_type": ADD},
                )()
                attn_op_list = [
                    first_ln,
                    attn_self_gemm,
                    mha,
                    attn_out_gemm,
                    attn_add,
                ]
            else:
                attn_self_gemm = Gemm(prefix + "attention.self",
                                      first_ln.outputs[0])()
                mha = MultiQueryAttention(
                    prefix + "attention",
                    [attn_self_gemm.outputs[0], mask.outputs[0]],
                    {
                        "num_heads": cfg.num_heads,
                        "hidden_size": hidden_size_,
                        "kv_channels": cfg.kv_channels,
                        "multi_query_group_num": 1,
                        "multigpu": 1,
                    },
                )()

                attn_out_gemm = Gemm(prefix + "attention.output.dense",
                                     mha.outputs[0])()
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
                    first_ln,
                    attn_self_gemm,
                    mha,
                    attn_out_gemm,
                    all_reduce_attention,
                    attn_add,
                ]
            # ffn
            ffn_ln = LayerNorm(prefix + "ffn.layernorm", attn_add.outputs[0],
                               {"eps": cfg.ln_eps})()
            ffn_intermediate = Gemm(
                prefix + "ffn.intermediate.dense",
                ffn_ln.outputs[0],
                {"activation": GELU_TANH},
            )()

            if self.multigpu_mode == 0:
                ffn_out = Gemm(prefix + "ffn.output.dense",
                               ffn_intermediate.outputs[0])()
                final_add = Binary(
                    prefix + "final_add",
                    [ffn_out.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [ffn_ln, ffn_intermediate, ffn_out, final_add]
            else:
                ffn_out = Gemm(prefix + "ffn.output.dense",
                               ffn_intermediate.outputs[0])()
                all_reduce_ffn_out = AllReduce(prefix + "all_reduce_ffn_out",
                                               ffn_out.outputs[0])()
                final_add = Binary(
                    prefix + "final_add",
                    [all_reduce_ffn_out.outputs[0], attn_add.outputs[0]],
                    {"binary_type": ADD},
                )()
                ffn_op_list = [
                    ffn_ln,
                    ffn_intermediate,
                    ffn_out,
                    all_reduce_ffn_out,
                    final_add,
                ]
            # final
            graph.ops.extend(attn_op_list + ffn_op_list)
        # decoder over
        final_layernorm = LayerNorm("final.layernorm",
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
                self.name_adapter.fullname("lm_head.weight")
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
                if (op.op_type == "DecOptMHA" or op.op_type == "DecOptMQA"):
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
            print("trans_tensor: {}, {}".format(key, torch_name))
            tensor = torch_weight[torch_name].cpu()
            start_time = time.time()
            print(self.model.model_conf.num_heads, tensor.shape[0])
            if len(list(tensor.shape)) > 1:
                print(tensor.shape[1])
            if key.find("weight") != -1:
                tensor = torch.permute(tensor, (1, 0)).contiguous()
            # if torch_name.find("lm_head.weight") != -1:
            #    tensor = torch.permute(tensor, (1, 0)).contiguous()
            print(torch_name, tensor.shape)
            if str(tensor.dtype) != "torch." + self_dtype_str:
                raise ValueError(
                    "DataType not match, [weight dtype: {}] vs [model dtype:{}]"
                    .format(str(tensor.dtype), "torch." + self_dtype_str))
            mode = DENSE if key not in self.sparse_map else self.sparse_map[key]
            split_mode = NOSPLIT if key not in self.split_map else self.split_map[
                key]
            print("split mode: ", split_mode, key)
            if split_mode != MQA_VSPLIT:
                group_list = []
            else:
                print("=======================")
                print(
                    self.model.model_conf.num_heads,
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num,
                )
                group_list = [
                    self.model.model_conf.num_heads *
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.kv_channels,
                    self.model.model_conf.multi_query_group_num *
                    self.model.model_conf.kv_channels,
                ]
            quantize_mode = (False if key not in self.quantize_map else
                             self.quantize_map[key])
            if quantize_mode == False:
                save_torch_to_allsparky(self.weights_path, key, tensor, mode,
                                        split_mode, group_list)
            else:
                if self.quant_config.quantize_mode == QuantizeConfig.QuantMode.A16W8:
                    qdata, scale, zero_point = quantize_gemm_weight_a16w8_torch(
                        tensor, self.quant_config)
                    # TODO: MultiGPU, and Check the number of GPUs.
                    save_torch_to_allsparky(self.weights_path, key, qdata,
                                            mode, split_mode, group_list)
                    if (self.quant_config.extra_option["SubChannel"] == True
                            or split_mode != HSPLIT):
                        save_torch_to_allsparky(
                            self.weights_path,
                            key + ".scale",
                            scale,
                            mode,
                            split_mode,
                            group_list,
                        )
                        save_torch_to_allsparky(
                            self.weights_path,
                            key + ".zero_point",
                            zero_point,
                            mode,
                            split_mode,
                            group_list,
                        )
                    else:
                        save_torch_to_allsparky(
                            self.weights_path,
                            key + ".scale",
                            scale,
                            mode,
                            NOSPLIT,
                            group_list,
                        )
                        save_torch_to_allsparky(
                            self.weights_path,
                            key + ".zero_point",
                            zero_point,
                            mode,
                            NOSPLIT,
                            group_list,
                        )
            if torch_name != "word_embeddings":
                torch_weight[torch_name] = torch.Tensor(0)
        set_global_header(self.weights_path)
