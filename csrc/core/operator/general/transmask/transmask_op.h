/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    transmask_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/*!
 * @brief mask的处理op
 * inputs:
 *    > input_mask : (int64) [batch_size, seq_length]
 * attr:
 *    > sequence_mask:
 *    解码需要，生成一个上三角矩阵的additional_mask[seq_length, seq_length]
 *    > blank:
 *    chatglm_v1模型使用，对于seq-1的部分全赋值为1，最后一个设置为只能看到自己
 * outputs:
 *    > out_mask : (fp32) [batch_size, seq_length, seq_length]
 */
class TransMaskOp : public AsOperator {
 public:
  explicit TransMaskOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_length_(1),
        seq_mask_(false),
        blank_(false) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 private:
  std::pair<bool, AsMHAPrefill> GetPrefillMode() {
    AsMHAPrefill prefill_mode = AsMHAPrefill(ctx_->GetPrefillMode());

    std::string mha_dtype_indicator =
        "decoder.layer.0.attention.output.dense.out";
    auto tensor_map_iter = tensor_map_->find(mha_dtype_indicator);
    bool dtype_indicator_exist = false;      // if false, cannot use flash.
    DataType mha_dtype = DataType::FLOAT32;  // flash only support bf16 / half
    if (tensor_map_iter != tensor_map_->end()) {
      dtype_indicator_exist = true;
      mha_dtype = tensor_map_->at(mha_dtype_indicator).get()->GetDataType();
    }

    return std::make_pair(true, prefill_mode);
  }

  AsStatus (*kernel_launcher)(DataType dtype, void* out, const int64_t* in,
                              int batch, int seq_len, bool seq_mask, bool blank,
                              const DeviceContext* ctx) = nullptr;
  AsStatus (*batch_offset_kernel_launcher)(int* out, const int64_t* in,
                                           int batch, int seq_len,
                                           const DeviceContext* ctx) = nullptr;
  int batch_size_;
  int seq_length_;
  bool seq_mask_;
  bool blank_;
};

}  // namespace allspark
