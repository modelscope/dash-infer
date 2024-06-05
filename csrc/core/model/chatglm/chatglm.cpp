/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chatglm.cpp
 */

#include "chatglm.h"  // NOLINT

namespace allspark {
AsStatus ChatGLMModel::Init(const TransformerProto& model_proto,
                            const DeviceContext& ctx) {
  DLOG(INFO) << "ChatGLMModel::Init()" << std::endl;
  AS_CHECK_STATUS(AsModel::Init(model_proto, ctx));
  topo_ops_.clear();
  // parse graph
  for (auto& op : graph_ops_["decoder"]) {
    topo_ops_.emplace_back(op.get());
  }
  for (auto& op : graph_ops_["gen_graph"]) {
    topo_ops_.emplace_back(op.get());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ChatGLMModel::Forward(const TensorMap& inputs, TensorMap* outputs) {
  const Shape& in_shape = inputs.at("input_ids")->GetShape();
  int batch_size = in_shape[0];
  int in_length = in_shape[1];
  gen_ctx_->batch_size = batch_size;
  gen_ctx_->max_length = in_length;
  gen_ctx_->only_decoder = true;

  bool need_reshape = false;
  for (auto& t : inputs) {
    const std::string& name = t.first;
    if (tensors_.find(name) == tensors_.end()) {
      LOG(ERROR) << "Invalid input tensor name: " << name << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    std::shared_ptr<AsTensor> tensor = t.second;
    AsTensor* old_in_tensor = tensors_[name].get();
    // check device_type and data_type
    if (old_in_tensor->GetDeviceType() != tensor->GetDeviceType() ||
        old_in_tensor->GetDataType() != tensor->GetDataType() ||
        old_in_tensor->GetShape().Size() != tensor->GetShape().Size()) {
      LOG(ERROR) << "Invalid input tensor : " << tensor->ToString() << " vs "
                 << old_in_tensor->ToString()
                 << ", please check input_shape, data_type or device_type."
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    // check shape
    if (old_in_tensor->GetShape() != tensor->GetShape()) {
      need_reshape = true;
    }
    tensors_[name] = tensor;
  }

  for (auto& op : graph_ops_["pre_graph"]) {
    op->SetGenerateContext(*gen_ctx_);
  }
  for (auto& op : graph_ops_["decoder"]) {
    op->SetGenerateContext(*gen_ctx_);
  }

  for (auto& op : graph_ops_["pre_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }
  for (auto& op : graph_ops_["pre_graph"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }
  ctx_->Synchronize();
  // copy tensor to out tensor_map
  if (rank_ == 0) {
    std::string output_name = "last_hidden_state";
    (*outputs)[output_name] = tensors_[output_name];
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_MODEL("ChatGLM_v2", ChatGLM_v2Model)
REGISTER_MODEL("ChatGLM_v3", ChatGLM_v3Model)
REGISTER_MODEL("ChatGLM_v4", ChatGLM_v4Model)
}  // namespace allspark
