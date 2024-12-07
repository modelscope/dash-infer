/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    test_operator_utils.h
 */

#pragma once
#include <allspark.h>
#include <common/device_context.h>
#include <core/operator/operator.h>
#include <core/tensor/tensor.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <test_common.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>

namespace AS_UTEST {

/**
 * @brief Test Op Utils
 *
 */
class TestOpUtil {
 public:
  TestOpUtil() = delete;

  TestOpUtil(const allspark::DeviceType device_type) {
    device_context_ =
        allspark::DeviceContextFactory::CreateDeviceContext(device_type);
    if (device_context_ == nullptr) {
      LOG(ERROR) << "Create device context fail.";
    }
    // Init device context
    device_context_->SetDeviceId(0);
    // Init workspace
    tensor_map_["workspace"] = std::make_unique<allspark::AsTensor>(
        "workspace", device_type, allspark::DataType::INT8);
    tensor_map_["cublas_workspace"] = std::make_unique<allspark::AsTensor>(
        "cublas_workspace", device_type, allspark::DataType::INT8);
    // according to https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
    // 64 MiB workspark is enough
    size_t cu_ws_bytes = 64 * 1024 * 1024;
    tensor_map_["cublas_workspace"]->SetShape(
        Shape{static_cast<dim_t>(cu_ws_bytes)});
  }

  void SetOpType(const std::string op_type) { op_proto_.set_op_type(op_type); }

  template <typename T>
  void SetOpAttribute(const std::string key, const T val) {
    auto& proto_map = *(op_proto_.mutable_attr());
    proto_map[key] =
        std::string(reinterpret_cast<const char*>(&val), sizeof(T));
  }

  void SetOpAttribute(const std::string key, const std::string val) {
    auto& proto_map = *(op_proto_.mutable_attr());
    proto_map[key] = val;
  }

  template <typename T>
  void AddInput(const std::string tensor_name, const std::vector<int64_t> shape,
                const allspark::DeviceType device_type,
                const allspark::DataType data_type,
                const allspark::DataMode data_mode,
                const std::vector<T>& src_data, const bool is_weight) {
    auto tensor = std::make_shared<allspark::AsTensor>(
        tensor_name, device_type, data_type, data_mode, allspark::Shape(shape));
    tensor->CopyDataFrom(src_data.data(), src_data.size() * sizeof(T),
                         allspark::DeviceType::CPU,
                         reinterpret_cast<const allspark::DeviceContext*>(
                             device_context_.get()));
    device_context_->Synchronize();

    if (is_weight == true) {
      op_proto_.add_weights()->set_name(tensor_name);
      weight_map_[tensor_name] = tensor;
    } else {
      op_proto_.add_inputs()->set_name(tensor_name);
      tensor_map_[tensor_name] = tensor;
    }
  }

  void AddOutput(const std::string tensor_name,
                 const std::vector<int64_t> shape,
                 const allspark::DeviceType device_type,
                 const allspark::DataType data_type,
                 const allspark::DataMode data_mode) {
    op_proto_.add_outputs()->set_name(tensor_name);
    auto tensor = std::make_shared<allspark::AsTensor>(
        tensor_name, device_type, data_type, data_mode, allspark::Shape(shape));
    tensor_map_[tensor_name] = tensor;
  }

  allspark::OperatorProto& GetOpProto() { return op_proto_; }

  allspark::TensorMap& GetWeightMap() { return weight_map_; }

  allspark::TensorMap& GetTensorMap() { return tensor_map_; }

  std::shared_ptr<allspark::DeviceContext>& GetDeviceContext() {
    return device_context_;
  }

 public:
  allspark::OperatorProto op_proto_;
  std::shared_ptr<allspark::DeviceContext> device_context_ = nullptr;

  allspark::TensorMap weight_map_;
  allspark::TensorMap tensor_map_;
};

}  // namespace AS_UTEST