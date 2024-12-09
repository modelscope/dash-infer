/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dec_opt_mha_i8cache_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <core/kernel/cuda/mha_quant_cache/attn_kv_cache.hpp>

namespace allspark {
using cuda::mha_quant_cache::QuantType;

/* @brief:
 *      if CONTEXT:     first = xseql(CONTEXT)
 *          qkv:                [batch,     3, xseql, nhead, phead]
 *          mask:               [batch, xseql, xseql]
 *          position_embedding  [batch, nhead, xseql, xseql]
 *          context             [batch, xseql, nhead, phead]
 *      if DECODER:
 *          qkv:                [batch,     3,     1, nhead, phead]
 *          mask:               [batch, first, first]
 *          position_embedding  [batch, nhead, xseql]
 *          context             [batch,     1, nhead, phead]
 *      in this operator, we assume this operator has two stages.
 *          1. CONTEXT stage. is_decoder_() = false. after this stage, this flag
 * will flop to true.
 *          2. DECODER stage. is_decoder_() = true.  in this stage, xseql is
 * current sequence length, and offset is xseql-1 method Reshape() will call
 * twice, and at the begining of every stages, workspace usage will be
 * recalculated. methods calling pipeline is showing below: Init() ->
 * CONTEXT.Reshape() -> CONTEXT.Forward()
 *             -> DECODER.Reshape() -> DECODER.Forward() -> DECODER.Forward() ->
 * ...
 */

class DecOptMHAI8CacheOp : public AsOperator {
 public:
  explicit DecOptMHAI8CacheOp(const std::string& op_type = "")
      : AsOperator(op_type),
        dtype_(DATATYPE_UNDEFINED),
        batch_(1),
        nhead_(8),
        phead_(64),
        cache_(128),
        xseql_(32),
        mask_exist_(true),
        pos_embedding_exist_(false),
        multigpu_(true),
        alpha_exist_(false),
        alpha_(0.f) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  // AsStatus RunOneBatch(GenerateContext* gen_ctx,int current_batch);

 private:
  // io info
  DataType dtype_ = DATATYPE_UNDEFINED;
  int32_t batch_, /*nhead_,*/ phead_, cache_,
      xseql_;          // xseql is current seqence length.
  int32_t first_ = 1;  // first time / context seqence length.

  int layer_num_ = 0;
  // kv cache
  std::unique_ptr<AsTensor> kc_, vc_;            // cache
  std::unique_ptr<AsTensor> kz_, vz_, ks_, vs_;  // param
  void* kc_ptr_ = nullptr;
  void* vc_ptr_ = nullptr;
  float* kz_ptr_ = nullptr;
  float* vz_ptr_ = nullptr;
  float* ks_ptr_ = nullptr;
  float* vs_ptr_ = nullptr;

  // utils
  bool mask_exist_, pos_embedding_exist_, multigpu_;

  // nhead from attributes
  int32_t nhead_ = 1;
  AsStatus nhead_from_attributes_(const OperatorProto& op) {
    auto& attr = op.attr();
    if (attr.find("num_heads") != attr.end()) {
      nhead_ = *(int32_t*)(attr.at("num_heads").c_str());
      return AsStatus::ALLSPARK_SUCCESS;
    } else {
      LOG(ERROR) << "DecOptMHAI8CacheOp : can't find num_heads attribute."
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }

  // alpha / scale params
  bool alpha_exist_ = false;
  float alpha_ = 1.f;
  AsStatus alpha_from_attributes_(const OperatorProto& op) {
    auto& attr = op.attr();
    if (attr.find("alpha") != attr.end()) {
      alpha_exist_ = true;
      alpha_ = *(float*)(attr.at("alpha").c_str());
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

  // kv cache quant type
  QuantType quant_type_;
  AsStatus quant_type_from_attributes_(const OperatorProto& op) {
    auto& attr = op.attr();
    if (attr.find("quant_type") == attr.end()) {
      // LOG(ERROR) << "DecOptMHAI8CacheOp : can't find quant_type attribute."
      // << std::endl; return AsStatus::ALLSPARK_PARAM_ERROR;
      quant_type_ = QuantType::INT8;
    } else {
      quant_type_ =
          static_cast<QuantType>(*(int*)(attr.at("quant_type").c_str()));
    }
    if (quant_type_ != QuantType::INT8 && quant_type_ != QuantType::UINT4) {
      LOG(ERROR) << "DecOptMHAI8CacheOp : not support for quant_type "
                 << quant_type_ << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

  // logn support
  int xlogn_ = -1;  // model base sequence embedding length.
  bool enable_logn_ = false;
  AsStatus logn_from_attributes_(const OperatorProto& op) {
    auto& attr = op.attr();
    if (attr.find("logn_model_embedding") != attr.end()) {
      enable_logn_ = true;
      xlogn_ = *(int*)(attr.at("logn_model_embedding").c_str());
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }
};

}  // namespace allspark
