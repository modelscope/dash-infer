/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator.h
 */

#pragma once

#include <common/device_context.h>
#include <common/generate_context.h>
#include <core/tensor/tensor.h>
#include <utility/model_profiler.h>

#include <dnnl.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/weight/weight_manager.h"

namespace allspark {

/*
class ModelWeightHandler;
class WeightManager;
*/
struct DNNLOpContext {
  std::vector<std::unique_ptr<dnnl::primitive>> pr_fwd_;
  std::vector<std::unique_ptr<dnnl::memory>> ins_;
  std::vector<std::unique_ptr<dnnl::memory>> outs_;
  dnnl::algorithm algo_;
  std::unique_ptr<dnnl::primitive_attr> attr_;
  static std::map<UnaryType, dnnl::algorithm> unary_algo_map_;
  static std::map<BinaryType, dnnl::algorithm> binary_algo_map_;
};

/*!
 * @brief Operator base class
 */
class AsOperator {
 public:
  explicit AsOperator(const std::string& op_type = "");
  virtual ~AsOperator() = default;
  // the only Init interface to call for model module
  AsStatus CallInit(const OperatorProto& op_proto, const DeviceContext& ctx,
                    std::shared_ptr<WeightManager> weight_manager,
                    std::shared_ptr<ModelWeightHandler> model_weight_handler,
                    RankInfo& rank_info, TensorMap* tensor_map,
                    ModelProfiler* profiler);

  // model to call OP forward/reshape/alloc
  AsStatus CallForward();
  AsStatus CallForward(RuntimeContext* runtime_ctx);
  AsStatus CallReshape();
  AsStatus CallReshape(RuntimeContext* runtime_ctx);
  AsStatus CallAlloc(RuntimeContext* runtime_ctx);

  AsStatus SetGenerateContext(GenerateContext& gen_ctx);
  AsStatus SetGenerateContextList(
      std::vector<std::unique_ptr<GenerateContext>>& gen_ctx_list);
  void PrintInformation();
  void SaveInformation();
  void SaveTensorToBinary();
  void SetEmbeddingMap(std::vector<TensorListMap>* embedding_map);
  std::vector<std::string> GetInNames();
  std::vector<std::string> GetOutNames();
  TensorMap GetInTensors();
  TensorMap GetOutTensors();
  TensorMap GetWeights();
  std::string GetOpType();
  virtual AsStatus ResetCache();
  std::string GetOpName() { return op_name_; }

  static int64_t round32(int64_t x) { return ((x + 32 - 1) / 32) * 32; }
  virtual AsStatus InitV2(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map,
                          TensorMap& weights_buffer, TensorMap* tensor_map);

 protected:
  std::string op_type_;
  std::string op_name_;
  std::vector<std::string> in_names_;
  std::vector<std::string> out_names_;
  std::vector<AsTensor*> weights_;  // TODO: use smart weak reference for this.

  TensorMap* tensor_map_;  // TODO: this tensor map only store input and
                           // output tensor.
  std::vector<TensorListMap>* embedding_map_;
  const DeviceContext* ctx_;
  GenerateContext* gen_ctx_;
  std::unique_ptr<DNNLOpContext> dnnl_op_ctx_;
  ModelProfiler* profiler_ = nullptr;
  std::shared_ptr<ModelWeightHandler> weight_handler_;
  std::shared_ptr<WeightManager> weight_manager_;
  RankInfo rank_info_;

  virtual AsStatus Forward();
  virtual AsStatus Forward(RuntimeContext* runtime_ctx);
  virtual AsStatus Reshape();
  virtual AsStatus Reshape(RuntimeContext* runtime_ctx);
  virtual AsStatus Alloc(RuntimeContext* runtime_ctx);

  virtual AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                        const TensorMap& weights_map, TensorMap* tensor_map);
};

/*!
 * @brief Operator regist type class
 */
class OpRegistType {
 public:
  std::string op_type_str;
  DeviceType device_type;
  OpRegistType(std::string op_type_str, DeviceType device_type)
      : op_type_str(op_type_str), device_type(device_type) {}

  bool operator==(const OpRegistType& p) const {
    return op_type_str == p.op_type_str && device_type == p.device_type;
  }
};

class OpRegistTypeHashFunction {
 public:
  size_t operator()(const OpRegistType& p) const {
    size_t seed = 0;
    seed = std::hash<std::string>{}(p.op_type_str) + 0x9e3779b9 + (seed << 6) +
           (seed >> 2);
    seed = std::hash<int>{}(p.device_type) + 0x9e3779b9 + (seed << 6) +
           (seed >> 2);
    return seed;
  }
};

using OpConstructor = std::function<std::unique_ptr<AsOperator>()>;
using OpMap =
    std::unordered_map<OpRegistType, OpConstructor, OpRegistTypeHashFunction>;

/*!
 * @brief Operator factory class
 */
class OpFactory {
 public:
  static OpFactory& getInstance();
  OpConstructor GetOperator(const OpRegistType& op_reg_type);
  void Register(const OpRegistType& op_reg_type, OpConstructor op_constructor);

 private:
  OpFactory() = default;
  OpFactory(const OpFactory&) = delete;
  OpFactory(OpFactory&&) = delete;
  OpMap op_set_;
};

/*!
 * @brief Operator reflector class
 */
class OpRegisterHelper {
 public:
  OpRegisterHelper(const OpRegistType& op_reg_type,
                   OpConstructor op_constructor) {
    OpFactory::getInstance().Register(op_reg_type, op_constructor);
  }
};

#define REGISTER_OP(op_name, device_type, typed_class)           \
  static OpRegisterHelper typed_class##Register##_##device_type( \
      OpRegistType(op_name, DeviceType::device_type),            \
      []() -> std::unique_ptr<AsOperator> {                      \
        return std::make_unique<typed_class>(op_name);           \
      });

}  // namespace allspark
