/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator.cpp
 */

#include "operator.h"  // NOLINT
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#include <cuda/cuda_util.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#endif
#include <common/engine_runtime.h>
#include <cpu/cpu_context.h>
#include <cpu/cpu_info.h>
#include <weight/weight_manager.h>

#include <fstream>
namespace allspark {

#ifdef ENABLE_CUDA
void ValidateNumeric(const AsTensor* tensor_ptr) {
  void* p = tensor_ptr->GetDataPtr();
  size_t size = tensor_ptr->GetSizeInByte();
  if (tensor_ptr->GetDeviceType() == DeviceType::CUDA) {
    if (tensor_ptr->GetDataType() == DataType::FLOAT16)
      cuda_util::CheckInfNan<__half>((half*)p, size);
#ifdef ENABLE_BF16
    else if (tensor_ptr->GetDataType() == DataType::BFLOAT16)
      cuda_util::CheckInfNan<__nv_bfloat16>((__nv_bfloat16*)p, size);
#endif
  }
}
#endif

std::map<UnaryType, dnnl::algorithm> DNNLOpContext::unary_algo_map_ = {
    {UnaryType::TANH, dnnl::algorithm::eltwise_tanh},
    {UnaryType::GELU_ERF, dnnl::algorithm::eltwise_gelu_erf},
    {UnaryType::GELU_TANH, dnnl::algorithm::eltwise_gelu_tanh},
    {UnaryType::RELU, dnnl::algorithm::eltwise_relu},
    {UnaryType::SILU, dnnl::algorithm::eltwise_swish},
};
std::map<BinaryType, dnnl::algorithm> DNNLOpContext::binary_algo_map_ = {
    {BinaryType::ADD, dnnl::algorithm::binary_add},
    {BinaryType::MUL, dnnl::algorithm::binary_mul},
    {BinaryType::GEGLU,
     dnnl::algorithm::binary_mul},  // geglu is not defined in dnnl, use mul
                                    // instead
    {BinaryType::SWIGLU,
     dnnl::algorithm::binary_mul}  // swiglu is not defined in dnnl, use mul
                                   // instead
};
AsOperator::AsOperator(const std::string& op_type)
    : op_type_(op_type),
      tensor_map_(nullptr),
      ctx_(nullptr),
      gen_ctx_(nullptr) {}

void AsOperator::Synchronize() { ctx_->Synchronize(); }

void AsOperator::PrintInformation() {
  ctx_->Synchronize();
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      if (gpu_ctx->GetRank() != 0) {
        return;
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      if (cpu_ctx->GetRank() != 0) {
        return;
      }
      break;
    }
    default:
      LOG(ERROR) << "Unsupported device type: " << int(backend);
      break;
  }

  std::cout << string_format("{ op_name: %s, op_type: %s}", op_name_.c_str(),
                             op_type_.c_str())
            << std::endl;
  std::cout << "op_inputs:" << std::endl;
  for (auto& v : in_names_) {
    std::cout << tensor_map_->at(v)->ToString() << std::endl;
#ifdef ENABLE_CUDA
    if (backend == DeviceType::CUDA) {
      ValidateNumeric(tensor_map_->at(v).get());
    }
#endif
  }
  std::cout << "op_weights:" << std::endl;
  for (auto& v : weights_) {
    std::cout << v->ToString() << std::endl;
#ifdef ENABLE_CUDA
    if (backend == DeviceType::CUDA) {
      ValidateNumeric(v);
    }
#endif
  }
  std::cout << "op_outputs:" << std::endl;
  for (auto& v : out_names_) {
    if (v == "generated_ids") continue;
    std::cout << tensor_map_->at(v)->ToString() << std::endl;
#ifdef ENABLE_CUDA
    if (backend == DeviceType::CUDA) {
      ValidateNumeric(tensor_map_->at(v).get());
    }
#endif
  }
  std::cout << std::endl;
}

// NOTE: File will be overrided in next step.
void AsOperator::SaveInformation() {
  ctx_->Synchronize();
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      if (gpu_ctx->GetRank() != 0) {
        return;
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      if (cpu_ctx->GetRank() != 0) {
        return;
      }
      break;
    }
    default:
      LOG(ERROR) << "Unsupported device type: " << int(backend);
      break;
  }

  std::ofstream OutFile(op_name_.c_str());

  OutFile << "op_inputs:" << std::endl;
  for (auto& v : in_names_) {
    OutFile << tensor_map_->at(v)->ToStringAll() << std::endl;
  }
  OutFile << "op_weights:" << std::endl;
  for (auto& v : weights_) {
    OutFile << v->ToStringAll() << std::endl;
  }
  OutFile << "op_outputs:" << std::endl;
  for (auto& v : out_names_) {
    if (v == "generated_ids") continue;
    OutFile << tensor_map_->at(v)->ToStringAll() << std::endl;
  }
  OutFile << std::endl;
}

#include <sys/stat.h>
void AsOperator::SaveTensorToBinary() {
  ctx_->Synchronize();

  std::string path = "./tmp_data/";

  struct stat st;
  if (stat(path.c_str(), &st) == -1) {
    // if the folder doesn't exist, create a folder
    if (mkdir(path.c_str(), 0777) == -1) {
      std::cerr << "Error: Failed to create folder!" << std::endl;
      return;
    } else {
      std::cout << "Success: Folder created successfully!" << std::endl;
    }
  } else {
    // if the folder exists, do nothing
    // cout << "Success: Folder already exists!" << endl;
  }

  for (auto& v : in_names_) {
    std::string filename = path + tensor_map_->at(v)->GetName();

    void* data_ptr = tensor_map_->at(v)->GetDataPtr();
    int shape_size = tensor_map_->at(v)->GetShape().Count();
    int elem_size = SizeofType(tensor_map_->at(v)->GetDataType());
    LOG(INFO) << "saving: " << filename.c_str()
              << ", shape count: " << shape_size << ", elem size: " << elem_size
              << std::endl;

    FILE* fp = fopen(filename.c_str(), "wb");
    fwrite(data_ptr, 1, shape_size * elem_size, fp);
    fclose(fp);
  }

  for (auto& v : weights_) {
    std::string filename = path + v->GetName();

    void* data_ptr = v->GetDataPtr();
    int shape_size = v->GetShape().Count();
    int elem_size = SizeofType(v->GetDataType());
    LOG(INFO) << "saving: " << filename.c_str()
              << ", shape count: " << shape_size << ", elem size: " << elem_size
              << std::endl;
    FILE* fp = fopen(filename.c_str(), "wb");
    fwrite(data_ptr, 1, shape_size * elem_size, fp);
    fclose(fp);
  }

  for (auto& v : out_names_) {
    if (v == "generated_ids") continue;

    std::string filename = path + tensor_map_->at(v)->GetName();

    void* data_ptr = tensor_map_->at(v)->GetDataPtr();
    int shape_size = tensor_map_->at(v)->GetShape().Count();
    int elem_size = SizeofType(tensor_map_->at(v)->GetDataType());
    LOG(INFO) << "saving: " << filename.c_str()
              << ", shape count: " << shape_size << ", elem size: " << elem_size
              << std::endl;

    FILE* fp = fopen(filename.c_str(), "wb");
    fwrite(data_ptr, 1, shape_size * elem_size, fp);
    fclose(fp);
  }
}

/* to save intermediate variable like quantized weight, quantized input.
* * only support 2 dims tensor
* # to load from binary file and return torch tensor
* def load_data_f16(filename):
    with open(filename, 'rb') as file:
        rows = int.from_bytes(file.read(4), byteorder='little')
        cols = int.from_bytes(file.read(4), byteorder='little')
        print(f"rows:{rows} cols:{cols}")
        data = np.fromfile(file, dtype=np.float16)

    # Convert NumPy array to torch.tensor
    tensor = torch.tensor(data, dtype=torch.float16)
    tensor = tensor.reshape((rows, cols))
    return tensor
*/
void AsOperator::SaveTmpData(const char* data, int rows, int cols,
                             int type_size, const char* filename) {
  ctx_->Synchronize();
  std::vector<char> h_data(rows * cols * type_size);
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      if (gpu_ctx->GetRank() != 0) {
        return;
      }
      cudaError_t err = cudaMemcpy(h_data.data(), data, rows * cols * type_size,
                                   cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return;
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      if (cpu_ctx->GetRank() != 0) {
        return;
      }
      memccpy(h_data.data(), data, rows * cols * type_size, 1);
      break;
    }
    default:
      LOG(ERROR) << "Unsupported device type: " << int(backend);
      break;
  }

  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char*>(h_data.data()),
             rows * cols * type_size);
  file.close();
}

AsStatus AsOperator::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  tensor_map_ = tensor_map;
  op_name_ = op_proto.op_name();
  in_names_.clear();
  for (auto& t : op_proto.inputs()) {
    const std::string& t_name = t.name();
    if (tensor_map_->count(t_name) == 0) {
      tensor_map_->insert(std::make_pair(
          t_name, std::make_unique<AsTensor>(t, ctx.GetDeviceType())));
    }
    in_names_.emplace_back(t_name);
  }
  for (auto& t : op_proto.outputs()) {
    const std::string& t_name = t.name();
    if (tensor_map_->count(t_name) == 0) {
      tensor_map_->insert(std::make_pair(
          t_name, std::make_unique<AsTensor>(t, ctx.GetDeviceType())));
    }
    if (is_lora_op_ && out_names_.size() > 0) continue;
    out_names_.emplace_back(t_name);
  }

  for (auto& t : op_proto.weights()) {
    const std::string& t_name = t.name();
    if (is_lora_op_) {
      // only in order to make it can be run by GemmOpBase::Init && Reshape,
      // which is called by GemmLora
      weight_names_.emplace_back(t_name);
      weights_.emplace_back(new AsTensor(  // 这里不能使用智能指针
          t_name, DeviceType::CUDA, DataType::INT8, DataMode::DENSE,
          Shape({1, 3})));  // 3 for qkv shape check
    } else if (weight_manager_) {
      auto weight_tensor_p =
          weight_manager_->GetWeightTensor(weight_handler_, rank_info_, t_name);
      weights_.emplace_back(weight_tensor_p.get());
    } else {
      // no manager, fallback to original weight fetch method, usually
      // it's from the unit test.
      if (weights_map.count(t_name) > 0) {
        AsTensor* weight = weights_map.at(t_name).get();
        weights_.emplace_back(weight);
      }
    }
  }
  ctx_ = &ctx;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsOperator::CallInit(
    const OperatorProto& op_proto, const DeviceContext& ctx,
    std::shared_ptr<WeightManager> weight_manager,
    std::shared_ptr<ModelWeightHandler> model_weight_handler,
    std::shared_ptr<LoraManager> lora_manager, RankInfo& rankInfo,
    TensorMap* tensor_map, ModelProfiler* profiler) {
  profiler_ = profiler;
  weight_handler_ = model_weight_handler;
  weight_manager_ = weight_manager;
  lora_manager_ = lora_manager;
  std::string op_type = op_proto.op_type();
  auto& attr_map = op_proto.attr();
  rank_info_ = rankInfo;
  TensorMap stub_weight;  // weight already handle by weight manager, create a
                          // fake one make interface compile pass.
  return InitV2(op_proto, ctx, stub_weight, stub_weight, tensor_map);
}

AsStatus AsOperator::InitV2(const OperatorProto& op_proto,
                            const DeviceContext& ctx,
                            const TensorMap& weights_map,
                            TensorMap& weights_buffer, TensorMap* tensor_map) {
  return Init(op_proto, ctx, weights_map, tensor_map);
}

AsStatus AsOperator::SetGenerateContext(
    std::shared_ptr<GenerateContext>& gen_ctx) {
  gen_ctx_ = gen_ctx;
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsOperator::SetEmbeddingMap(std::vector<TensorListMap>* embedding_map) {
  embedding_map_ = embedding_map;
}
OpFactory& OpFactory::getInstance() {
  static OpFactory op_factory;
  return op_factory;
}

OpConstructor OpFactory::GetOperator(const OpRegistType& op_reg_type) {
  if (op_set_.find(op_reg_type) == op_set_.end()) {
    LOG(ERROR) << "Unsupported op type: " << op_reg_type.op_type_str
               << std::endl;
    throw AsException("Unsupported op type.");
  }
  return op_set_[op_reg_type];
}

void OpFactory::Register(const OpRegistType& op_reg_type,
                         OpConstructor op_constructor) {
  op_set_[op_reg_type] = op_constructor;
}

std::vector<std::string> AsOperator::GetInNames() { return in_names_; }

std::vector<std::string> AsOperator::GetOutNames() { return out_names_; }
// for debug only
TensorMap AsOperator::GetInTensors() {
  TensorMap ret;
  ctx_->Synchronize();
  for (auto& name : in_names_) {
    ret[name] = tensor_map_->at(name);
  }
  return ret;
}

// for debug only
TensorMap AsOperator::GetOutTensors() {
  TensorMap ret;
  ctx_->Synchronize();
  for (auto& name : out_names_) {
    ret[name] = tensor_map_->at(name);
  }
  return ret;
}

// for debug only
TensorMap AsOperator::GetWeights() {
  TensorMap ret;
  ctx_->Synchronize();
  for (auto& p : weights_) {
    ret[p->GetName()] = std::make_shared<AsTensor>(*p);
  }
  return ret;
}

std::string AsOperator::GetOpType() { return op_type_; }

std::pair<bool, AsMHAPrefill> AsOperator::GetPrefillMode() {
  if (cached_prefill_mode_) {
    return *cached_prefill_mode_;
  }

  AsMHAPrefill prefill_mode = AsMHAPrefill(ctx_->GetPrefillMode());

  // A. dtype. check
  // see:     m6_v3.py
  // prefix = "decoder.layer.{}.".format(i)
  // MultiHeadAttention(prefix + "attention",
  //          [rotary_embedding.outputs[0], mask.outputs[0]],
  //          mha_attribtues,)()
  // see:     model_base.py
  // class MultiHeadAttention(Operator):
  //     def __init__(self, op_name, inputs, op_attr={}):
  //         super().__init__("MultiHeadAttention", op_name, inputs, op_attr)
  //         self.op.outputs.append(make_tensor(op_name + ".out"))
  std::string mha_dtype_indicator =
      "decoder.layer.0.attention.output.dense.out";
  auto tensor_map_iter = tensor_map_->find(mha_dtype_indicator);
  bool dtype_indicator_exist = false;      // if false, cannot use flash.
  DataType mha_dtype = DataType::FLOAT32;  // flash only support bf16 / half
  if (tensor_map_iter != tensor_map_->end()) {
    dtype_indicator_exist = true;
    mha_dtype = tensor_map_->at(mha_dtype_indicator).get()->GetDataType();
  }

  // C. cuda and sm-version
  // use cpu flags by default
  bool enable_cuda = false;
  int sm_version = 0;
  int cuda_version = 0;
  bool not_support_flashv2 = !(CPUInfo::SupportAVX512F());

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    int device_id;
    cudaDeviceProp dprop;
    AS_CHECK_CUDA(cudaGetDevice(&device_id));
    AS_CHECK_CUDA(cudaGetDeviceProperties(&dprop, device_id));

    enable_cuda = true;
    sm_version = dprop.major << 8 | dprop.minor;
    cuda_version = CUDA_VERSION;
    bool not_supported_dtype =
        (dtype_indicator_exist && mha_dtype != DataType::BFLOAT16 &&
         mha_dtype != DataType::FLOAT16);
    not_support_flashv2 =
        not_supported_dtype || (sm_version < 0x0800) || (cuda_version < 11080);
  }
#endif

  // flashv2 check
  if (prefill_mode == allspark::AsMHAPrefill::AsPrefillFlashV2 &&
      ((!dtype_indicator_exist) || not_support_flashv2)) {
    LOG(ERROR) << "GetPrefillMode() return false. "
               << "incoming prefill mode = AsMHAPrefill::AsPrefillFlashV2. "
               << std::endl;

    cached_prefill_mode_ = std::make_shared<std::pair<bool, AsMHAPrefill>>(
        std::make_pair(false, prefill_mode));
    return *cached_prefill_mode_;
  }

  // xformer check
  if (prefill_mode == allspark::AsMHAPrefill::AsPrefillXformer &&
      ((!enable_cuda) ||
       (enable_cuda && ctx_->GetDeviceType() != DeviceType::CUDA))) {
    LOG(ERROR) << "GetPrefillMode() return false. "
               << "incoming prefill mode = AsMHAPrefill::AsPrefillXformer. "
               << std::endl;
    cached_prefill_mode_ = std::make_shared<std::pair<bool, AsMHAPrefill>>(
        std::make_pair(false, prefill_mode));
    return *cached_prefill_mode_;
  }

  cached_prefill_mode_ = std::make_shared<std::pair<bool, AsMHAPrefill>>(
      std::make_pair(true, prefill_mode));

  return *cached_prefill_mode_;
}

AsStatus AsOperator::Alloc(RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsOperator::ResetCache() { return AsStatus::ALLSPARK_SUCCESS; }

AsStatus AsOperator::Reshape() { return AsStatus::ALLSPARK_SUCCESS; }

AsStatus AsOperator::Forward() { return AsStatus::ALLSPARK_SUCCESS; }

AsStatus AsOperator::Reshape(RuntimeContext* runtime_ctx) {
  return this->Reshape();
}

AsStatus AsOperator::Forward(RuntimeContext* runtime_ctx) {
  return this->Forward();
}

AsStatus AsOperator::CallForward(RuntimeContext* runtime_ctx) {
#ifdef CONFIG_OP_DEBUG
  LOG(INFO) << "AsOperator::CallForward(ctx) " << GetOpType() << " Start.";
#endif
  AsStatus ret;
  if (profiler_) {
    ProfilerAdder adder(*profiler_, "forward", GetOpType(), ctx_);
    ret = Forward(runtime_ctx);
  } else {
    ret = Forward(runtime_ctx);
  }

#ifdef CONFIG_OP_DEBUG
  ctx_->Synchronize();
  LOG(INFO) << "AsOperator::CallForward(ctx) " << GetOpType() << " Finish.";
#endif
  return ret;
}

AsStatus AsOperator::CallReshape(RuntimeContext* runtime_ctx) {
#ifdef CONFIG_OP_DEBUG
#endif
  if (profiler_) {
    ProfilerAdder adder(*profiler_, "reshape", GetOpType(), ctx_);
    return Reshape(runtime_ctx);
  } else {
    return Reshape(runtime_ctx);
  }
}

AsStatus AsOperator::CallAlloc(RuntimeContext* runtime_ctx) {
#ifdef CONFIG_CONCURRENT_SPAN
  std::string name = "Alloc:" + this->op_name_;
  TracerLog trace(ctx_->GetDeviceType(), name.c_str(), 3);

  return Alloc(runtime_ctx);
#else
  if (profiler_) {
    ProfilerAdder adder(*profiler_, "alloc", GetOpType(), ctx_);
    return Alloc(runtime_ctx);
  } else {
    return Alloc(runtime_ctx);
  }
#endif
}

}  // namespace allspark
