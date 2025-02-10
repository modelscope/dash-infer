/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model.cpp
 */

#include "model.h"  // NOLINT

#include <common/engine_runtime.h>
#include <common/env_config.h>
#include <common/memory_reuser.h>
#include <core/operator/generate_opt/postprocess_id/postprocess_id_op.h>
#include <core/operator/generate_opt/span_attn/span_attn_op.h>
#include <utility/arbiter.h>
#include <utility/file_util.h>
#include <utility/mem_registry.h>
#include <utility/timer.h>
#include <weight/weight_manager.h>

#include <common/extra_embedding.hpp>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#ifdef ENABLE_CUDA
#include <core/kernel/cuda/sample.h>
#include <cuda/cuda_context.h>
#include <curand_kernel.h>
#include <device/cuda/cuda_cache_allocator.h>
#endif
#include <device/memory_func.h>

#include <random>

#include "runtime/weight/weight_manager_lora.h"

#define DEBUG_GEN_LAYER 0
#define DEBUG_GEN_LAYER_SAVE_NPY 0
#define DEBUG_GEN_LAYER_SYNC 0

#ifdef ENABLE_CUDA
// profile the fixed batch size gneration layer.
// start nsys with: `nsys  profile -c cudaProfilerApi xxx`
// better disable wramup for this profiling.
#define PROFILE_CONTEXT_TIME_GPU 0
#define PROFILE_GENERATION_TIME_GPU 0
// this requires ENABLE_NSYS_PROFILE in cuda_context.cpp.
#define PROFILE_GENERATION_TIME_BS 100
#endif

static bool isWarmupRequest(const std::string& str) {
  const std::string start_with = "warmup_request_";
  return str.find(start_with) == 0;
}

#ifdef DEBUG_GEN_LAYER
bool debugCurrentRequest(const std::string& str) {
  if (isWarmupRequest(str)) return false;

  std::string target_request = "";
  // std::string target_request = "0000000000000000000000000000001";

  if (target_request == "") {
    // debug all requests
    return true;
  } else if (target_request == str) {
    return true;
  } else {
    return false;
  }
}
#endif

namespace allspark {
using std::string;
using std::vector;

AsModel::AsModel(const std::string& model_type)
    : model_type_(model_type), ctx_(nullptr), current_unfinished_request_(0) {
  gen_ctx_model_ = std::make_unique<GenerateContext>();
  runtime_ctx_ = std::make_unique<RuntimeContext>();

  // pre-alloc enough request space.
  all_request_map_.reserve(1000);
}

AsTensor AsModel::GetOutputTensor(std::string tensor_name) {
  DLOG(INFO) << "AsModel::GetOutputTensor()" << std::endl;
  return *tensors_[tensor_name];
}

void AsModel::GetInformation(std::string* model_info) {
  DLOG(INFO) << "AsModel::GetInformation()" << std::endl;
  std::stringstream ss;
  ss << "Model Type : " << model_type_ << std::endl;
  ss << "Model Inputs : " << std::endl;
  for (const std::string& t_name : input_names_) {
    ss << "    " << tensors_[t_name]->ToString() << std::endl;
  }
  ss << "Model Outputs:" << std::endl;
  for (const std::string& t_name : output_names_) {
    ss << "    " << tensors_[t_name]->ToString() << std::endl;
  }
  *model_info = ss.str();
}

AsStatus AsModel::SaveWeights(std::string* out_allsparkz) {
  DLOG(INFO) << "AsModel::SaveWeights()" << std::endl;

  try {
    weight_manager_->SaveWeights(weight_handler_, out_allsparkz);
  } catch (AsException& e) {
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsModel::ChangeGemmOpType(OpRegistType& op_type) {
  if (op_type.op_type_str == "GemmA8W8" &&
      ctx_->GetDeviceType() == DeviceType::CUDA) {
    op_type.op_type_str = "GemmSparseA8W8";
  }
}

AsStatus AsModel::Init(const TransformerProto& model_proto,
                       const DeviceContext& ctx) {
  DLOG(INFO) << "AsModel::Init()" << std::endl;

  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  ctx_ = &ctx;
  DeviceType device_type = ctx.GetDeviceType();
  // model_profiler_
  auto do_profile = EnvVarConfig::GetString("AS_PROFILE", "OFF");
  if (do_profile == "ON") {
    model_profiler_ = std::make_shared<ModelProfiler>(this);
  } else {
    model_profiler_ = nullptr;
  }

  auto rankInfo = this->GetRankInfo();
  AS_CHECK_STATUS(
      weight_manager_->LoadWeightForModel(ctx, weight_handler_, rankInfo));

  // load LoRA
  auto& model_cfg = weight_handler_->GetModelConfig();
  lora_manager_ = LoraManager::Create(
      model_cfg.lora_max_num, rankInfo);  // 每个卡上的AsModel都拥有一批LoRA
  if (model_cfg.lora_names.size() > 0) {
    LOG(WARNING) << "Config item 'lora_names' is not any longer supported "
                    "and ignored!";
  }

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_CONCURRENT_SPAN
    int num_pool_threads =
        std::max(std::min(std::thread::hardware_concurrency() / ctx.GetNranks(),
                          static_cast<uint32_t>(ctx.GetDecoderLayer())),
                 1U);
    LOG(INFO) << "Model init: size of context thread pool: "
              << num_pool_threads;
    layer_threadpool_ = std::make_unique<ThreadPool>(num_pool_threads);
#endif

    // cache managers
    tokens_per_cache_span_ = ctx.GetCacheSpanSize();
    if (tokens_per_cache_span_ > 0) {
      int num_cache_heads = ctx.GetNumberGroups() > 0 ? ctx.GetNumberGroups()
                                                      : ctx.GetNumberHeads();

#ifdef ENABLE_CUDA
      if (device_type == DeviceType::CUDA) {
        cache_allocator_ = std::make_shared<CudaCacheAllocator>(ctx_);
      }
#endif

#ifdef CONFIG_CONCURRENT_SPAN
      if (ctx.GetCacheSpanNumGrow() != 0) {
        LOG(WARNING) << "WARNING: using ConcurrentCacheFrameManager, "
                        "cache_span_num_grow is ignored";
      }
      cache_frame_manager_ = std::make_shared<ConcurrentCacheFrameManager>(
          device_type, ctx.GetCacheSpanNumInit());
      cache_span_manager_ =
          std::make_shared<ConcurrentCacheSpanManager>(cache_frame_manager_);
#else
      cache_frame_manager_ = std::make_shared<DefaultCacheFrameManager>(
          device_type, ctx.GetCacheSpanNumInit(), ctx.GetCacheSpanNumGrow());
      cache_span_manager_ =
          std::make_shared<DefaultCacheSpanManager>(cache_frame_manager_);
#endif

      if (num_cache_heads % rankInfo.rank_size != 0) {
        LOG(ERROR) << "AsModel::Init: head number should be a multiple of "
                   << "nranks, head number: " << num_cache_heads
                   << ", nranks: " << rankInfo.rank_size;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }

      size_t span_bytes_z = CacheUtils::GetSpanSizeInBytes(
          *(ctx.GetCacheConfig()), ctx.GetDtype(),
          num_cache_heads / rankInfo.rank_size, ctx.GetSizePerHead());
      if (span_bytes_z > std::numeric_limits<int64_t>::max()) {
        LOG(ERROR) << "AsModel::Init: span size in bytes exceeds int64_t, got "
                   << span_bytes_z;
        return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
      }

      cache_span_manager_->Init(static_cast<int64_t>(span_bytes_z));
      LOG(INFO) << "AsModel: tokens per cache span: " << tokens_per_cache_span_
                << ", init spans: " << ctx.GetCacheSpanNumInit()
                << ", grow spans: " << ctx.GetCacheSpanNumGrow();
    }
  }
#endif

  // parse io tensor
  for (auto& t : model_proto.inputs()) {
    tensors_.insert(
        std::make_pair(t.name(), std::make_unique<AsTensor>(t, device_type)));
    input_names_.emplace_back(t.name());
  }

  for (auto& t : model_proto.outputs()) {
    tensors_.insert(
        std::make_pair(t.name(), std::make_unique<AsTensor>(t, device_type)));
    output_names_.emplace_back(t.name());
  }

  // shared tmp workspace for operator
  tensors_.insert(std::make_pair(
      "workspace", std::make_unique<AsTensor>("workspace", ctx.GetDeviceType(),
                                              DataType::INT8)));
  gen_ctx_model_ = std::make_unique<GenerateContext>();
  runtime_ctx_ = std::make_unique<RuntimeContext>();

  std::unique_ptr<AsTensor> rotary_step = std::make_unique<AsTensor>(
      "rotary_step", ctx_->GetDeviceType(), DataType::INT32);
  rotary_step->SetShape(Shape{ctx_->GetModelMaxBatch()});
  std::unique_ptr<AsTensor> rotary_inv_freq = std::make_unique<AsTensor>(
      "rotary_inv_freq", ctx_->GetDeviceType(), DataType::FLOAT32);
  rotary_inv_freq->SetShape(
      Shape{ctx_->GetModelMaxBatch() * ctx_->GetSizePerHead() / 2});

  runtime_ctx_->CreateLayerCacheManager();
  runtime_ctx_->GetLayerCacheManager()->CreateCache("rotary_step",
                                                    std::move(rotary_step));
  runtime_ctx_->GetLayerCacheManager()->CreateCache("rotary_inv_freq",
                                                    std::move(rotary_inv_freq));

  auto& graph = model_proto.graphs();
  int nodes = 0;

  DLOG(INFO) << "Start process model graph.";
  for (auto& g_name : model_proto.graph_names()) {
    std::vector<std::unique_ptr<AsOperator>> ops;

    for (auto& op_proto : graph.at(g_name).ops()) {
      OpRegistType op_type(op_proto.op_type(), ctx.GetDeviceType());
      if (ctx_->GetSparsityMatmulMode()) {
        ChangeGemmOpType(op_type);
      }

      std::unique_ptr<AsOperator> op =
          OpFactory::getInstance().GetOperator(op_type)();

      AS_CHECK_STATUS(op->CallInit(op_proto, ctx, weight_manager_,
                                   weight_handler_, lora_manager_, rankInfo,
                                   &tensors_, model_profiler_.get()));

      op->SetEmbeddingMap(&embedding_);
      ops.emplace_back(std::move(op));

      nodes += 1;
    }
    graph_ops_.insert(std::make_pair(g_name, std::move(ops)));
  }

  vector<vector<AsTensor*>> topo_tensors;
  topo_tensors.resize(nodes + 2);
  int topo_size = 0;
  for (auto& g_name : model_proto.graph_names()) {
    for (auto& op : graph_ops_[g_name]) {
      for (const std::string& name : op->GetInNames()) {
        topo_tensors[topo_size].emplace_back(tensors_[name].get());
      }
      for (const std::string& name : op->GetOutNames()) {
        topo_tensors[topo_size].emplace_back(tensors_[name].get());
      }
      topo_size += 1;
    }
  }
  MemoryReuser memory_reuser_;
  memory_reuser_.binding_with_algo_0(topo_tensors,
                                     const_cast<DeviceContext*>(ctx_));
  tensors_["attention_mask"]->SetShape(Shape{1, ctx_->GetModelMaxLength()});
  tensors_.insert(std::make_pair(
      "context_k_workspace",
      std::make_unique<AsTensor>("context_k_workspace", ctx.GetDeviceType(),
                                 DataType::INT8)));
  tensors_.insert(std::make_pair(
      "context_v_workspace",
      std::make_unique<AsTensor>("context_v_workspace", ctx.GetDeviceType(),
                                 DataType::INT8)));

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    tensors_.insert(std::make_pair(
        "cublas_workspace",
        std::make_unique<AsTensor>("cublas_workspace", ctx.GetDeviceType(),
                                   DataType::INT8)));
  }
#endif

  const size_t kv_ws_bytes = ctx.GetModelMaxLength() * ctx.GetNumberHeads() *
                             ctx.GetSizePerHead() * SizeofType(ctx.GetDtype()) /
                             rankInfo.rank_size;
  if (kv_ws_bytes > std::numeric_limits<dim_t>::max()) {
    LOG(ERROR) << "AsModel::Init: context KV workspace size exceeds dim_t";
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  tensors_["context_k_workspace"]->SetShape(
      Shape{static_cast<dim_t>(kv_ws_bytes)});
  tensors_["context_v_workspace"]->SetShape(
      Shape{static_cast<dim_t>(kv_ws_bytes)});

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    model_cfg = weight_handler_->GetModelConfig();
    if (model_cfg.enable_prefix_cache) {
      prefix_cache_manager_ = std::make_shared<PrefixCacheManager>(
          cache_span_manager_, cache_frame_manager_, prefix_cache_coordinator_,
          &tensors_, ctx_, model_cfg.prefix_cache_ttl);
    }
  }
#endif

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, device_id);
    // according to https://docs.nvidia.com/cuda/cublas/#cublassetstream
    // cublasSetWorkspace supports user-defined workspace for cublas,
    // suggest workspace size 32 MiB for Hopper Architecture, 4 MiB for others.
    size_t cu_ws_bytes = 8 * 1024 * 1024;
    if (dprop.major >= 9) {
      cu_ws_bytes = 64 * 1024 * 1024;
    }
    tensors_["cublas_workspace"]->SetShape(
        Shape{static_cast<dim_t>(cu_ws_bytes)});
  }
#endif  // ENABLE_CUDA

  DLOG(INFO) << "load model success.";

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::ReloadModelToDeviceMemory() {
  DLOG(INFO) << "AsModel::LoadWeightsFromBuffer()" << std::endl;
  weight_manager_->SwapInWeight(weight_handler_, this->GetRankInfo());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::UnloadModelFromDeviceMemory() {
  DLOG(INFO) << "AsModel::UnloadModelFromDeviceMemory()" << std::endl;
  ctx_->Synchronize();

  // free all containers
  graph_ops_.clear();
  tensors_.clear();
  embedding_.clear();
  input_names_.clear();
  output_names_.clear();
  topo_ops_.clear();

  weight_manager_->SwapOutWeight(weight_handler_, this->GetRankInfo());
  // 释放blocks
  const_cast<DeviceContext*>(ctx_)->ResetBlockPools();

  DLOG(INFO) << "AsModel::UnloadModelFromDeviceMemory() END" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsModel::PrintWeights() {}

#define CHECK_CUDA_ERROR(op)

/*
#define CHECK_CUDA_ERROR(op) do { \
                ctx_->Synchronize(); \
                cudaError_t r = cudaGetLastError(); \
                if (cudaSuccess != r) { \
                    LOG(ERROR) << "OP ERROR! " << cudaGetErrorString(r) <<
std::endl; \
                    //op->PrintInformation(); \
                } \
            } while (false);
*/

AsStatus AsModel::buildGenContext(
    std::shared_ptr<GenerateContext>& gen_ctx,
    const std::shared_ptr<Request>& request) const {
  gen_ctx->gen_cfg = request->gen_cfg;
  gen_ctx->request = request;
  gen_ctx->k_cache_list = std::vector<std::unique_ptr<CacheMemory>>();
  gen_ctx->v_cache_list = std::vector<std::unique_ptr<CacheMemory>>();

  gen_ctx->engine_max_length = ctx_->GetModelMaxLength();
  gen_ctx->input_len = request->inputs.at("input_ids")->GetShape()[1];
  gen_ctx->real_input_len = gen_ctx->input_len;
  gen_ctx->gen_cfg.input_len = gen_ctx->input_len;
  gen_ctx->max_length = ctx_->GetModelMaxLength();

#ifdef ENABLE_JSON_MODE
  if (request->gen_cfg.response_format.count("type") &&
      request->gen_cfg.response_format["type"] == "json_object") {
    gen_ctx->format_enforcer = request->format_enforcer;
    DLOG(INFO)
        << "Request:" << request->request_id
        << " FormatEnforcer pointer successfully passed to GenerateContext\n";
  }
#endif

  std::stringstream k_cache_tag;
  std::stringstream v_cache_tag;
  k_cache_tag << request->request_id << "_k";
  v_cache_tag << request->request_id << "_v";

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    switch (ctx_->GetCacheSpanSize()) {
      case 0:
        /* span cache disabled */
        break;
      default:
        gen_ctx->virtual_k_cache = std::make_unique<SpannedVirtualCache>(
            cache_span_manager_, ctx_->GetCacheConfig(), k_cache_tag.str(),
            ctx_->GetDecoderLayer());
        gen_ctx->virtual_v_cache = std::make_unique<SpannedVirtualCache>(
            cache_span_manager_, ctx_->GetCacheConfig(), v_cache_tag.str(),
            ctx_->GetDecoderLayer());
        break;
    }
  }
#endif

  int state_size = sizeof(std::mt19937);
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (ctx_->GetUseTorchSample()) {
      state_size = (int)sizeof(cuda::PhiloxCudaState);
    } else {
      state_size = (int)sizeof(curandState_t);
      gen_ctx->sample_state = std::make_unique<AsTensor>(
          "sample_state:" + request->request_id, DeviceType::CUDA,
          DataType::INT8, DataMode::DENSE, Shape({state_size}));
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      AS_CHECK_CUDA(cudaMemsetAsync(gen_ctx->sample_state->GetDataPtr(), 0,
                                    gen_ctx->sample_state->GetSizeInByte(),
                                    cu_stream));
    }
  }
#endif

  if (gen_ctx->sample_state == nullptr) {
    gen_ctx->sample_state = std::make_unique<AsTensor>(
        "sample_state:" + request->request_id, DeviceType::CPU, DataType::INT8,
        DataMode::DENSE, Shape({state_size}));
    memset(gen_ctx->sample_state->GetDataPtr(), 0,
           gen_ctx->sample_state->GetSizeInByte());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::runDecoderContext() {
  util::Timer t_begin;

#if PROFILE_CONTEXT_TIME_GPU
  // should skip the warm up request.
  {
    if (runtime_ctx_->GetGenCtxListSize() >= 10) {
      auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
      if (cuda_ctx) {
        cuda_ctx->Synchronize();
        LOG(INFO) << "NSys Profiler start.";
        cuda_ctx->NsysProfilerStart();
      }
    }
  }
#endif

  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");

  std::shared_ptr<GenerateContext> gen_ctx =
      (runtime_ctx_->GetGenCtx(runtime_ctx_->current_batch));
  GenerateConfig gen_cfg = gen_ctx->gen_cfg;

  DLOG(INFO) << "start run context ,uuid = " << gen_ctx->request->request_id
             << " lora_name=" << gen_cfg.lora_name << std::endl;
  int batch_size = 1;
  size_t in_length =
      gen_ctx->request->interim.at("new_input_ids")->GetShape()[1];
  gen_ctx->batch_size = batch_size;
  if (gen_cfg.do_sample && gen_cfg.num_beams == 1) {
    gen_ctx->generate_method = 0;  // sample
  } else {
    gen_ctx->generate_method = 1;  // beam_search
  }
  gen_ctx->num_beams = gen_cfg.num_beams;
  if (in_length > gen_cfg.max_length) {
    LOG(ERROR) << "Error: input length: " << in_length
               << " exceeds generation config's max length: "
               << gen_cfg.max_length << std::endl;
    return ErrorProcess(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  for (auto& graph : graph_ops_) {
    for (auto& op : graph_ops_[graph.first]) {
      op->SetGenerateContext(gen_ctx);
    }
  }
  gen_ctx->only_decoder = true;
  gen_ctx->num_beams = 1;
  gen_ctx->step = gen_ctx->prefix_len;

  // finish pre-graph first to avoid possibly troublesome set shape
  for (auto& op : graph_ops_["pre_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  for (auto& op : graph_ops_["pre_graph"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());

#if DEBUG_GEN_LAYER_SYNC
    op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
    if (debugCurrentRequest(runtime_ctx_->GetGenCtx(0)->request->request_id)) {
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
    }
#endif

    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_pre_graph;
  // other op
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_alloc;
  {
    TracerLog trace(ctx_->GetDeviceType(), "ContextAlloc", 1);
#if ENABLE_SPAN_ATTENTION
    if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_CONCURRENT_SPAN
      auto num_dec_layers = ctx_->GetDecoderLayer();
      std::vector<std::future<AsStatus>> result(num_dec_layers);
      int layer_idx = 0;
      for (auto& op : graph_ops_["decoder"]) {
        if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr) {
          result[layer_idx++] = layer_threadpool_->enqueue(
              [this, &op]() { return op->CallAlloc(runtime_ctx_.get()); });
        }
      }
      // sanity check
      if (layer_idx != num_dec_layers) {
        LOG(ERROR) << "ContextAlloc: decoder layer number mismatch";
        return ErrorProcess(AsStatus::ALLSPARK_RUNTIME_ERROR);
      }

      for (int i = 0; i < num_dec_layers; ++i) {
        auto status = result[i].get();
        if (status != AsStatus::ALLSPARK_SUCCESS) {
          LOG(ERROR) << "ContextAlloc: alloc failed in loop::decoder";
          return ErrorProcess(status);
        }
      }
#else
      for (auto& op : graph_ops_["decoder"]) {
        if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr) {
          AsStatus status = op->CallAlloc(runtime_ctx_.get());
          CHECK_CUDA_ERROR(op)
          if (status != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << "ContextAlloc: alloc failed in loop::decoder";
            return ErrorProcess(status);
          }
        }
      }
#endif  // CONFIG_CONCURRENT_SPAN
    } else
#endif  // ENABLE_SPAN_ATTENTION
    {
      for (auto& op : graph_ops_["decoder"]) {
        AsStatus status = op->CallAlloc(runtime_ctx_.get());
        CHECK_CUDA_ERROR(op)
        if (status != AsStatus::ALLSPARK_SUCCESS) {
          LOG(ERROR) << "ContextAlloc: alloc failed in loop::decoder";
          return ErrorProcess(status);
        }
      }
    }
  }

  util::Timer t_forward;
  // pre_forward
  // first decoder for input_ids
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
    op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
    if (debugCurrentRequest(runtime_ctx_->GetGenCtx(0)->request->request_id)) {
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
    }
#endif
    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_reshape;
  gen_ctx->in_length_bias =
      gen_ctx->prefix_len == 0 ? gen_ctx->input_len : in_length;

  gen_ctx->num_beams = gen_cfg.num_beams;
  for (auto& op : graph_ops_["gen_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in gen_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_gen_graph;
  for (auto& op : graph_ops_["gen_graph"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
    op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
    if (debugCurrentRequest(runtime_ctx_->GetGenCtx(0)->request->request_id)) {
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
    }
#endif
    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in gen_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  util::Timer t_post_graph;
  for (auto& op : graph_ops_["post_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in post" << std::endl;
      return ErrorProcess(status);
    }
    status = op->CallForward(runtime_ctx_.get());
    CHECK_CUDA_ERROR(op)
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in post" << std::endl;
      return ErrorProcess(status);
    }
  }
  gen_ctx->in_length_bias = 0;
  gen_ctx->step = in_length + gen_ctx->prefix_len;

  DLOG(INFO) << "end run context ,uuid = " << gen_ctx->request->request_id
             << std::endl;
  util::Timer t_end;

  auto do_time_profile = EnvVarConfig::GetInt("ALLSPARK_TIME_LOG", 0);
  if (do_time_profile) {
    using util::Timer;
    auto pre_graph_time = Timer::duration_ms(t_begin, t_pre_graph);
    auto pre_reshape_time = Timer::duration_ms(t_pre_graph, t_alloc);
    auto alloc_time = Timer::duration_ms(t_alloc, t_forward);
    auto forward_time = Timer::duration_ms(t_forward, t_reshape);
    auto gen_reshape_time = Timer::duration_ms(t_reshape, t_gen_graph);
    auto gen_time = Timer::duration_ms(t_gen_graph, t_post_graph);
    auto post_graph_time = Timer::duration_ms(t_post_graph, t_end);

    auto context_time_ms = t_begin.elapsed();

    LOG(INFO) << "Context Time [TTFT](ms) " << context_time_ms
              << " pre_graph: " << pre_graph_time
              << " pre_reshape: " << pre_reshape_time
              << " allocate: " << alloc_time << " forward: " << forward_time
              << " gen_reshape: " << gen_reshape_time
              << " gen_forward: " << gen_time
              << " post_graph: " << post_graph_time;
  }

#if PROFILE_CONTEXT_TIME_GPU
  {
    if (runtime_ctx_->GetGenCtxListSize() >= 10) {
      auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
      if (cuda_ctx) {
        cuda_ctx->Synchronize();
        LOG(INFO) << "NSys Profiler Stop.";
        cuda_ctx->NsysProfilerStop();
      }
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::StartRequest(std::shared_ptr<Request> request) {
  DLOG(INFO) << "AsModel:StartRequest()" << std::endl;

  // check lora
  auto& lora_name = request->gen_cfg.lora_name;
  DLOG(INFO) << "Incoming request: " << request->request_id
             << " lora_name=" << lora_name << std::endl;
  if (!lora_name.empty()) {
    assert(lora_manager_ != nullptr);
    if (!lora_manager_->IsLoraExists(lora_name)) {
      LOG(ERROR) << "check lora in AsModel::StartRequest failed, LoRA "
                 << lora_name << " not loaded!";
      StopRequest(request->request_id);
      request->status = AsEngine::GenerateRequestStatus::GenerateFinished;
      request->finish = true;
      return AsStatus::ALLSPARK_LORA_NOT_FOUND;
    }
  }

  int batch = runtime_ctx_->GetGenCtxListSize();

  std::shared_ptr<GenerateContext> gen_ctx =
      std::make_shared<GenerateContext>();
  AS_CHECK_STATUS(buildGenContext(gen_ctx, request));
  runtime_ctx_->PushBackGenCtx(gen_ctx);

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    const int max_seq_len = ctx_->GetModelMaxLength();
    const int span_len = ctx_->GetCacheSpanSize();
    const int max_num_spans = (max_seq_len + span_len - 1) / span_len;
    for (int layer_id = 0; layer_id < ctx_->GetDecoderLayer(); layer_id++) {
      AS_CHECK_STATUS(gen_ctx->virtual_k_cache->InitLayer(
          layer_id, ctx_->GetNumberHeads(), ctx_->GetSizePerHead(), 0,
          max_num_spans));
      AS_CHECK_STATUS(gen_ctx->virtual_v_cache->InitLayer(
          layer_id, ctx_->GetNumberHeads(), ctx_->GetSizePerHead(), 0,
          max_num_spans));
    }
  }

  if (prefix_cache_manager_ != nullptr &&
      ctx_->GetDeviceType() == DeviceType::CUDA) {
    std::shared_ptr<AsTensor> new_input_ids_tensor;
    prefix_cache_manager_->RefFill(
        gen_ctx->request->inputs.at("input_ids"),
        gen_ctx->request->interim.at("input_ids_for_hash"),
        new_input_ids_tensor, gen_ctx->request->start_ts, gen_ctx->prefix_len,
        gen_ctx->virtual_k_cache, gen_ctx->virtual_v_cache,
        gen_ctx->prefix_cache_node_list);
    gen_ctx->request->interim.insert({"new_input_ids", new_input_ids_tensor});

    if (ctx_->GetRank() == 0) {
      LOG(INFO) << "[" << __FUNCTION__ << "] "
                << "request id: " << gen_ctx->request->request_id << ", "
                << "cached prefix_len: " << gen_ctx->prefix_len << ", "
                << "total tokens: "
                << gen_ctx->request->inputs.at("input_ids")->GetShape()[1];
    }
  } else {
#endif

    {
      int batch_now = request->inputs.at("input_ids")->GetShape()[0];
      int seq_now = request->inputs.at("input_ids")->GetShape()[1];
      gen_ctx->request->interim.insert(
          {"new_input_ids", request->inputs.at("input_ids")});
      tensors_["attention_mask"]->SetShape(Shape({batch_now, seq_now}));
    }

#if ENABLE_SPAN_ATTENTION
  }
#endif

  runtime_ctx_->is_context = true;
  runtime_ctx_->current_batch = batch;
  try {
    DLOG(INFO) << "before enter runcontext for request: " << request->request_id
               << " lora_name " << lora_name
               << " exist: " << lora_manager_->IsLoraExists(lora_name)
               << std::endl;
    runDecoderContext();
  } catch (std::exception& e) {
    LOG(ERROR) << "runDecoderContext() Failed: " << std::string(e.what())
               << ", "
               << "request_id = " << request->request_id;
    StopRequest(request->request_id);
    request->status = AsEngine::GenerateRequestStatus::GenerateInterrupted;
    throw e;
  }

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefix_cache_manager_ != nullptr) {
      prefix_cache_manager_->Insert(
          gen_ctx->request->interim.at("input_ids_for_hash"),
          gen_ctx->prefix_len, gen_ctx->request->start_ts,
          gen_ctx->virtual_k_cache->GetLayerCache(),
          gen_ctx->virtual_v_cache->GetLayerCache(),
          gen_ctx->prefix_cache_node_list);
    }
  }
#endif

  // context阶段成功结束
  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
  DLOG(INFO)
      << "AsModel::StartRequest: context finish, restore ops with Reshape";
  for (AsOperator* op : topo_ops_) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in topo_ops" << std::endl;
      return ErrorProcess(status);
    }
  }

  request->status = AsEngine::GenerateRequestStatus::ContextFinished;

  request->context_ts = std::chrono::steady_clock::now();
  auto duration = request->context_ts - request->start_ts;
  auto duration_in_milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  if (ctx_->GetRank() == 0) {
    LOG(INFO) << "[" << __FUNCTION__ << "] "
              << "Context Success, request id: " << request->request_id << ", "
              << "context phase time(ms): " << duration_in_milliseconds;
  }

  // clean up the finished request
  for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
    if (runtime_ctx_->GetGenCtx(i)->finish) {
      auto ret = StopRequest(runtime_ctx_->GetGenCtx(i)->request->request_id);
      if (ret != AsStatus::ALLSPARK_SUCCESS) return ret;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
bool StopPenddingReuqest(const std::string& request_id,
                         std::queue<std::shared_ptr<Request>>& pendding_queue) {
  std::queue<std::shared_ptr<Request>> newQueue;
  bool find_request = false;
  while (!pendding_queue.empty()) {
    std::shared_ptr<Request> request_ptr = pendding_queue.front();
    if (request_ptr->request_id == request_id) {
      find_request = true;
      request_ptr->status = AsEngine::GenerateRequestStatus::GenerateFinished;
      LOG(INFO) << "Request " << request_id << " stop before running";
      // delete reqeust in pendding queue
    } else {
      newQueue.push(request_ptr);
    }
    pendding_queue.pop();
  }
  pendding_queue = newQueue;
  return find_request;
}
AsStatus AsModel::StopRequest(const std::string& request_id) {
  if (StopPenddingReuqest(request_id, pending_request_queue_)) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  int request_idx = -1;
  for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
    if (runtime_ctx_->GetGenCtx(i)->request->request_id == request_id) {
      request_idx = i;
      break;
    }
  }
  if (request_idx == -1) {
    DLOG(ERROR) << "not find running request id:" << request_id
                << ",maybe already stop." << std::endl;
    return AsStatus::ALLSPARK_SUCCESS;
  }
  std::shared_ptr<GenerateContext> gen_ctx =
      runtime_ctx_->GetGenCtx(request_idx);
  std::shared_ptr<Request> request = gen_ctx->request;

  // free cache
  for (int i = 0; i < gen_ctx->k_cache_list.size(); i++) {
    gen_ctx->k_cache_list[i]->Free();
  }
  for (int i = 0; i < gen_ctx->v_cache_list.size(); i++) {
    gen_ctx->v_cache_list[i]->Free();
  }
  DLOG(INFO) << "AsModel::StopRequest: [" << request_id << "] cache released";

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefix_cache_manager_ != nullptr) {
      AS_CHECK_STATUS(ExtraEmbeddingUtils::CreateTensorForHash(
          request, request->interim, request->interim, "generated_ids"));
      prefix_cache_manager_->Insert(
          request->interim.at("generated_ids_for_hash"),
          gen_ctx->input_len / ctx_->GetCacheSpanSize() *
              ctx_->GetCacheSpanSize(),
          request->start_ts, gen_ctx->virtual_k_cache->GetLayerCache(),
          gen_ctx->virtual_v_cache->GetLayerCache(),
          gen_ctx->prefix_cache_node_list);

      prefix_cache_manager_->UnRef(gen_ctx->prefix_cache_node_list);

      std::stringstream ss;
      ss << "UnRef request_id: " << request->request_id << ", "
         << "rank: " << ctx_->GetRank();
      prefix_cache_manager_->PrintPrefixCacheInfo(ss.str());
      // prefix_cache_manager_->PrintAllPrefixCache();
    }

    gen_ctx->virtual_k_cache.reset();
    gen_ctx->virtual_v_cache.reset();

    if (prefix_cache_manager_ != nullptr) {
      if (isWarmupRequest(request->request_id)) {
        prefix_cache_manager_->EvictAllUnrefered();
      }
    }
  }
#endif
  request->extra_embedding.clear();
  int last_batch = runtime_ctx_->GetGenCtxListSize() - 1;

  // don't remove this sync, make sure finish with correct state.
  ctx_->Synchronize();
  runtime_ctx_->FinishRequest(request_idx);
  current_unfinished_request_--;

  if (ctx_->GetRank() == 0) {
    LOG(INFO) << "Stop request with request id: " << request_id;
  }
  if (runtime_ctx_->GetGenCtxListSize() > 0) {
    for (AsOperator* op : topo_ops_) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "reshape failed in topo_ops" << std::endl;
        return ErrorProcess(status);
      }
    }
  }
  using namespace std::chrono;
  request->generate_ts = std::chrono::steady_clock::now();
  auto gen_duration = request->generate_ts - request->context_ts;
  auto ctx_duration = request->context_ts - request->start_ts;
  auto gen_duration_in_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(gen_duration)
          .count();
  float gen_tps =
      (request->generated_len * 1000.0f) / ((gen_duration_in_ms) + 0.1f);

  auto ctx_dur_ms = duration_cast<milliseconds>(ctx_duration).count();
  float ctx_tps = (request->input_len * 1000.0f) / (ctx_dur_ms + 0.1f);

  if (ctx_->GetRank() == 0) {
    LOG(INFO) << "[" << __FUNCTION__ << "] "
              << "Request ID: " << request->request_id << ", "
              << "Context time(ms): " << ctx_dur_ms << ", "
              << "Generate time(ms): " << gen_duration_in_ms << ", "
              << "Context Length: " << request->input_len << ", "
              << "Generated Length: " << request->generated_len << ", "
              << "Context TPS: " << ctx_tps << ", "
              << "Generate TPS: " << gen_tps << ", "
              << "Prefix Cache Len: " << request->prefix_len;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::ReleaseRequest(const std::string& request_id) {
  std::unique_lock<std::mutex> lock(request_map_lock_);
  if (all_request_map_.find(request_id) != all_request_map_.end()) {
    all_request_map_.erase(request_id);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

Request* AsModel::GetRequestById(const std::string& request_id) {
  std::unique_lock<std::mutex> lock(request_map_lock_);
  if (all_request_map_.find(request_id) == all_request_map_.end()) {
    return nullptr;
  }
  return all_request_map_.at(request_id).get();
}

void AsModel::PrefillChunkRequest(std::shared_ptr<Request> request) {
  if (request->request_id.find("warmup") != std::string::npos) {
    pending_request_queue_.pop();
    StartRequest(request);
    request->prefill_chunk_len = request->input_len;
    return;
  }

  int new_len = request->prefill_chunk_len + ctx_->GetModelMaxPrefillLength();
  new_len += 1;  // prefix cache only cache input_len - 1, so new_len += 1

  if (new_len >= request->input_len) {
    request->inputs.at("input_ids")->SetShape(Shape{1, request->origin_len});
    pending_request_queue_.pop();
    StartRequest(request);
    request->prefill_chunk_len = request->input_len;
  } else {
    std::shared_ptr<Request> request_ptr = std::make_shared<Request>(
        request->request_id, request->inputs, request->outputs,
        request->gen_cfg, request->interim);
    request_ptr->request_id.append("_chunk_prefill");
    request_ptr->input_len = new_len;
    request_ptr->inputs.at("input_ids")->SetShape(Shape{1, new_len});
    request_ptr->gen_cfg.max_length = new_len + 1;
    StartRequest(request_ptr);
    request->prefill_chunk_len = new_len - 1;
  }
}
AsStatus AsModel::GenerateContinueContext(
    bool is_new_context)  // if is_new_context=true, set
                          // already_context_length_=0
{
  // maybe use if,each turn only run one context phase
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  if (!runtime_ctx_) return AsStatus::ALLSPARK_EMPTY_REQUEST;

  DLOG(INFO) << " gen ctx list " << runtime_ctx_->GetGenCtxListSize()
             << " pending init " << pending_request_queue_.size();
  if (is_new_context) {
    already_context_length_ = 0;
  }
  if (!pending_request_queue_.empty() &&
      runtime_ctx_->GetGenCtxListSize() < ctx_->GetModelMaxBatch()) {
    std::shared_ptr<Request> request = pending_request_queue_.front();
    if (already_context_length_ != 0 &&
        already_context_length_ + request->input_len >
            ctx_->GetModelMaxPrefillLength()) {
      return AsStatus::ALLSPARK_EMPTY_REQUEST;
    }
#if ENABLE_SPAN_ATTENTION
    if (ctx_->GetDeviceType() == DeviceType::CUDA) {
      if (!cache_frame_manager_) {
        // not use span
        pending_request_queue_.pop();
        StartRequest(request);
        DLOG(INFO) << "RunContext SUCCESS ,request id = "
                   << request->request_id;

        current_unfinished_request_.store(pending_request_queue_.size() +
                                          runtime_ctx_->GetGenCtxListSize());
        return AsStatus::ALLSPARK_SUCCESS;
      } else {
        // use_span
        int model_layer = ctx_->GetDecoderLayer();
        int min_gen_length = 10;
        int span_size = ctx_->GetCacheSpanSize();
        size_t context_frame = 0;
        if (prefix_cache_manager_ != nullptr) {
          AS_CHECK_STATUS(ExtraEmbeddingUtils::CreateTensorForHash(
              request, request->interim, request->inputs, "input_ids"));

          std::vector<PrefixCacheManager::PrefixNodePtr> prefix_cache_node_list;
          int prefix_len = 0;
          int gpu_cached_len = 0;
          prefix_cache_manager_->RefOnly(
              request->interim.at("input_ids_for_hash"), request->start_ts,
              prefix_len, gpu_cached_len, prefix_cache_node_list);

          int real_input = request->input_len - prefix_len;
          if (request->prefill_chunk_len == 0) {
            // first running context, calc prefix cnt
            prefix_cache_manager_->UpdateCnt(prefix_len, real_input);
            request->prefill_chunk_len = prefix_len;
            request->prefix_len = prefix_len;
            request->prefix_len_gpu = gpu_cached_len;
            if (prefix_len != 0) {
              LOG(INFO) << "request: " << request->request_id << ", "
                        << "find prefix cache, len: " << prefix_len << ", "
                        << "total tokens: " << request->input_len;
            }
          }
          context_frame =
              (std::min((request->input_len - gpu_cached_len) + min_gen_length,
                        ctx_->GetModelMaxLength()) /
                   span_size +
               1) *
              2 * model_layer;
          if (context_frame > cache_frame_manager_->CountFreeFrame()) {
            LOG(INFO) << "Not enough frame for new request, "
                      << "need frame vs free frame: " << context_frame << "/ "
                      << cache_frame_manager_->CountFreeFrame()
                      << ", swap unrefered prefix cache to cpu memory";
            prefix_cache_manager_->EvictUnrefered(context_frame);
          }
          prefix_cache_manager_->UnRef(prefix_cache_node_list);
        } else {
          context_frame = (std::min(request->input_len + min_gen_length,
                                    ctx_->GetModelMaxLength()) /
                               span_size +
                           1) *
                          2 * model_layer;
        }
        // int context_frame = request->input_len * model_layer *
        // model_nhead / ctx_->GetNrank();

        if (context_frame <= cache_frame_manager_->CountFreeFrame()) {
          PrefillChunkRequest(request);
          DLOG(INFO) << "RunContext SUCCESS ,request id = "
                     << request->request_id;
          current_unfinished_request_.store(pending_request_queue_.size() +
                                            runtime_ctx_->GetGenCtxListSize());
          if (request->prefill_chunk_len != request->input_len) {
            return AsStatus::ALLSPARK_CHUNK_PREFILL;
          } else {
            already_context_length_ += request->input_len;
            return AsStatus::ALLSPARK_SUCCESS;
          }
        } else {
          LOG(ERROR) << "Try RunContext " << request->request_id
                     << ", but not enough frame, so RunContext failed";
          pending_request_queue_.pop();
          request->finish = true;
          request->status =
              AsEngine::GenerateRequestStatus::GenerateInterrupted;
          LOG(INFO) << request->request_id << "GenerateInterrupted";
          current_unfinished_request_.store(pending_request_queue_.size() +
                                            runtime_ctx_->GetGenCtxListSize());
          return AsStatus::ALLSPARK_EMPTY_REQUEST;
        }
      }
    } else
#endif
    {
      pending_request_queue_.pop();
      StartRequest(request);
      DLOG(INFO) << "RunContext SUCCESS ,request id = " << request->request_id;
    }

    current_unfinished_request_.store(pending_request_queue_.size() +
                                      runtime_ctx_->GetGenCtxListSize());
  } else {
    // not free batch or no pending request
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::GenerateContinueDecoder() {
  DLOG(INFO) << "AsModel::GenerateContinueDecoder()" << std::endl;
  // maybe use if,each turn only run one context phase
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  util::Timer t1;
  // DLOG(INFO) << "pthread_self: " << (unsigned long)pthread_self();

  DLOG(INFO) << "Decoder: gen ctx list " << runtime_ctx_->GetGenCtxListSize()
             << " pending init " << pending_request_queue_.size()
             << " t1(ms): " << t0.elapsed();
#if PROFILE_GENERATION_TIME_GPU
  if (runtime_ctx_->GetGenCtxListSize() >= PROFILE_GENERATION_TIME_BS) {
    auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
    if (cuda_ctx) {
      cuda_ctx->Synchronize();
      cuda_ctx->NsysProfilerStart();
    }
  }
#endif

  // while (pending_request_queue_.size() > 0 &&
  //        runtime_ctx_->GetGenCtxListSize() < ctx_->GetModelMaxBatch()) {
  //     StartRequest(pending_request_queue_.front());
  //     pending_request_queue_.pop();
  // }

  current_unfinished_request_.store(pending_request_queue_.size() +
                                    runtime_ctx_->GetGenCtxListSize());
  const int async_token_num = 1;  // TODO gen_cfg.async_token_num
  for (int now_step = 0; now_step < async_token_num; now_step++) {
    int batch_size = runtime_ctx_->GetGenCtxListSize();
    if (batch_size == 0) {
      // LOG(INFO) << "Continue: empty batch size";
      return AsStatus::ALLSPARK_EMPTY_REQUEST;
    }

    util::Timer t2;
    gen_ctx_model_->step++;  // 利用这个废弃的字段，给校对工具使用
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");
    {
      TracerLog trace(ctx_->GetDeviceType(), "DecoderAlloc", 0);
      // do NOT run this concurrently, it reduces performance
#if ENABLE_SPAN_ATTENTION
      if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#ifdef CONFIG_CONCURRENT_SPAN
        auto num_dec_layers = ctx_->GetDecoderLayer();
        std::vector<std::future<AsStatus>> result(num_dec_layers);
        int layer_idx = 0;
        for (auto& op : graph_ops_["decoder"]) {
          if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr) {
            result[layer_idx++] = layer_threadpool_->enqueue(
                [this, &op]() { return op->CallAlloc(runtime_ctx_.get()); });
          }
        }
        // sanity check
        if (layer_idx != num_dec_layers) {
          LOG(ERROR) << "DecoderAlloc: decoder layer number mismatch";
          return ErrorProcess(AsStatus::ALLSPARK_RUNTIME_ERROR);
        }

        for (int i = 0; i < num_dec_layers; ++i) {
          auto status = result[i].get();
          if (status != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << "DecoderAlloc: alloc failed in loop::decoder";
            return ErrorProcess(status);
          }
        }
#else
        for (auto& op : graph_ops_["decoder"]) {
          if (dynamic_cast<SpanAttnOp*>(op.get()) != nullptr) {
            AsStatus status = op->CallAlloc(runtime_ctx_.get());
            CHECK_CUDA_ERROR(op)
            if (status != AsStatus::ALLSPARK_SUCCESS) {
              LOG(ERROR) << "DecoderAlloc: alloc failed in loop::decoder";
              return ErrorProcess(status);
            }
          }
        }
#endif  // CONFIG_CONCURRENT_SPAN
      } else
#endif  // ENABLE_SPAN_ATTENTION
      {
        for (auto& op : graph_ops_["decoder"]) {
          AsStatus status = op->CallAlloc(runtime_ctx_.get());
          CHECK_CUDA_ERROR(op)
          if (status != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << "DecoderAlloc: alloc failed in loop::decoder";
            return ErrorProcess(status);
          }
        }
      }
    }

    util::Timer t3;
    {
      TracerLog trace(ctx_->GetDeviceType(), "DecoderForward", 1);
      for (auto& op : graph_ops_["decoder"]) {
        AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
        op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
        if (debugCurrentRequest(
                runtime_ctx_->GetGenCtx(0)->request->request_id)) {
          op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
          DO_ARBITRATE(rank_, nranks_, gen_ctx_model_->step, op);
#endif
        }
#endif
        CHECK_CUDA_ERROR(op)
        if (status != AsStatus::ALLSPARK_SUCCESS) {
          LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
          return ErrorProcess(status);
        }
      }
    }

    util::Timer t4;
    for (int i = 0; i < batch_size; i++) {
      runtime_ctx_->GetGenCtx(i)->step += 1;
    }

    DLOG(INFO) << " decoder(ms): " << t0.elapsed();
    for (auto& op : graph_ops_["gen_graph"]) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      CHECK_CUDA_ERROR(op)
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
        return ErrorProcess(status);
      }
    }

    util::Timer t5;
    for (auto& op : graph_ops_["gen_graph"]) {
      AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
      op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
      if (debugCurrentRequest(
              runtime_ctx_->GetGenCtx(0)->request->request_id)) {
        op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
        DO_ARBITRATE(rank_, nranks_, gen_ctx_model_->step, op);
#endif
      }
#endif
      CHECK_CUDA_ERROR(op)
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::gen_graph" << std::endl;
        return ErrorProcess(status);
      }
    }

    util::Timer t6;
    for (auto& op : graph_ops_["post_graph"]) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "reshape failed in post" << std::endl;
        return ErrorProcess(status);
      }
      status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER_SYNC
      op->Synchronize();
#endif
#if DEBUG_GEN_LAYER
      if (debugCurrentRequest(
              runtime_ctx_->GetGenCtx(0)->request->request_id)) {
        op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
        DO_ARBITRATE(rank_, nranks_, gen_ctx_model_->step, op);
#endif
      }
#endif
      CHECK_CUDA_ERROR(op)
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in post" << std::endl;
        return ErrorProcess(status);
      }
    }

    // clean up the finished request
    for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
      if (runtime_ctx_->GetGenCtx(i)->finish) {
        auto ret = StopRequest(runtime_ctx_->GetGenCtx(i)->request->request_id);
        if (ret != AsStatus::ALLSPARK_SUCCESS) return ret;
      }
    }
    util::Timer t7;

    long tpot_ms = t0.elapsed();
    DLOG(INFO) << " Decoder Time(TPOT) (ms): " << tpot_ms;
    auto do_time_profile = EnvVarConfig::GetInt("ALLSPARK_TIME_LOG", 0);
    if (do_time_profile) {
      using util::Timer;
      auto lock_time = Timer::duration_ms(t0, t1);
      auto alloc_time = Timer::duration_ms(t1, t3);
      auto forward_time = Timer::duration_ms(t3, t4);
      auto reshape_time = Timer::duration_ms(t4, t5);
      auto gen_forward_time = Timer::duration_ms(t5, t6);
      auto post_forward_time = Timer::duration_ms(t6, t7);

      LOG(INFO) << "Decoder Loop Time [TPOT] (ms): " << tpot_ms
                << " running: " << runtime_ctx_->GetGenCtxListSize()
                << " lock time(ms):" << lock_time << " alloc: " << alloc_time
                << " forward_time: " << forward_time
                << " reshape: " << reshape_time
                << " gen_frd: " << gen_forward_time
                << " post_gen: " << post_forward_time;
    }

#if PROFILE_GENERATION_TIME_GPU
    if (runtime_ctx_->GetGenCtxListSize() >= PROFILE_GENERATION_TIME_BS) {
      auto cuda_ctx = dynamic_cast<const CUDAContext*>(ctx_);
      if (cuda_ctx) {
        cuda_ctx->Synchronize();
        cuda_ctx->NsysProfilerStop();
      }
    }
#endif
  }

  return AsStatus::ALLSPARK_STREAMING;
}

// XXX: This function will be called by *all* worker thread
// the handle and gen cfg passed from main loop thread,
// so *any* write operation (none const operation is not allowed!)
AsStatus AsModel::StartRequestImpl(
    const std::shared_ptr<RequestHandle> request_handle,
    const std::string request_id, TensorMap* outputs,
    const GenerateConfig& gen_cfg) {
  DLOG(INFO) << "AsModel::StartRequestImpl()" << std::endl;
  std::shared_ptr<Request> request_ptr = std::make_shared<Request>(
      request_id, *request_handle->inputs_internal, *outputs, gen_cfg);
  request_ptr->input_len = request_ptr->inputs.at("input_ids")->GetShape()[1];
  request_ptr->origin_len = request_ptr->input_len;
  request_ptr->extra_embedding = request_handle->mm_embedding_internal;
  request_ptr->enqueue_ts = request_handle->create_ts;
#ifdef ENABLE_JSON_MODE
  if (gen_cfg.response_format.count("type")) {
    try {
      if (gen_cfg.response_format.at("type") == "json_object") {
        request_ptr->format_enforcer = request_handle->format_enforcer;
      }
    } catch (const std::out_of_range& ex) {
      // not found response format, ignore.
    }
  }
#endif
  DLOG(INFO) << "AsModel::StartRequestImpl(): input length:"
             << request_ptr->input_len;

  std::unique_lock<std::mutex> lock(request_map_lock_);
  pending_request_queue_.push(request_ptr);

  all_request_map_[request_id] = request_ptr;
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::GenerateContinue() {
  AsStatus ret = GenerateContinueDecoder();
  AS_CHECK_STATUS(ret);
  return ret;
  // return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::AllocDecoderMemory() {
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);
  const int async_token_num = 1;  // TODO gen_cfg.async_token_num
  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (cache_frame_manager_) {
      int model_layer = ctx_->GetDecoderLayer();
      int span_size = ctx_->GetCacheSpanSize();
      int new_span_batch = 0;
      for (int i = 0; i < runtime_ctx_->GetGenCtxListSize(); i++) {
        std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx_->GetGenCtx(i);
        size_t length_now = gen_ctx->virtual_k_cache->GetSeqLength(0);
        if (length_now % span_size == 0) {
          new_span_batch += 1;
        }
      }
      int decoder_frame = model_layer * 2 * new_span_batch;
      if (decoder_frame > cache_frame_manager_->CountFreeFrame()) {
        if (cache_frame_manager_ && prefix_cache_manager_ != nullptr) {
          LOG(INFO) << "Not enough frame for decoder, "
                    << "need frame vs free frame: " << decoder_frame << " / "
                    << cache_frame_manager_->CountFreeFrame()
                    << ", swap unrefered prefix cache to cpu memory";
          prefix_cache_manager_->EvictUnrefered(decoder_frame);
        }
      }
      // after release all preifx cache still not enough frame
      if (decoder_frame > cache_frame_manager_->CountFreeFrame()) {
        LOG(ERROR) << "free span frame not enough for decoder: "
                   << decoder_frame << " vs "
                   << cache_frame_manager_->CountFreeFrame();
        throw AsException("ALLSPARK_MEMORY_ERROR");
      }
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::Warmup(int64_t bytes_available, int64_t bytes_runtime) {
  DLOG(INFO) << "AsModel::Warmup()";
  if (bytes_available < 0) {
    LOG(ERROR) << "AsModel::Warmup: bytes_available must be non-negative, got "
               << bytes_available;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (bytes_runtime < 0) {
    LOG(ERROR) << "AsModel::Warmup: bytes_runtime must be non-negative, got "
               << bytes_runtime;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  float runtime_mem_ratio = 1.1;
  // sgmv op在load、unload时仍有cuda mem小幅波动，多留些余量以免波动出OOM
  // 波动的原因与BFC释放和重新回收mem有关
  if (ctx_->GetLoraEnabled()) {
    runtime_mem_ratio = 1.5;
  }
  LOG(INFO) << "warm-up: runtime memory reservation ratio: "
            << runtime_mem_ratio;

  const int64_t bytes_cache = std::max(
      0LL, bytes_available - static_cast<int64_t>(
                                std::ceil(bytes_runtime * runtime_mem_ratio)));

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    size_t num_to_grow = bytes_cache / cache_frame_manager_->GetFrameSize();
    LOG(INFO) << "warm-up: trying to grow " << num_to_grow
              << " frames, current count of frames: "
              << cache_frame_manager_->CountFrame();
    if (cache_frame_manager_->GrowBy(num_to_grow)) {
      if (prefix_cache_manager_ != nullptr) {
        prefix_cache_manager_->UpdateCapacity();
      }
      LOG(INFO)
          << "warm-up: grow successfully, total number of claimed span frames: "
          << cache_frame_manager_->CountFrame();
    } else {
      LOG(ERROR) << "AsModel::Warmup: failed to grow all " << num_to_grow
                 << " frames, total number of claimed span frames: "
                 << cache_frame_manager_->CountFrame();
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
  }
#endif

// recommended batch size is never used in practice, so just disable it
#if 0
  // compute best batch size
  int maxlen = ctx_->GetModelMaxLength();
  const char* env_outlen = std::getenv("ALLSPARK_EXPECT_OUTLEN");
  if (env_outlen != nullptr) {
    int outlen = std::atoi(env_outlen);
    if (outlen > ctx_->GetModelMaxLength() || outlen < 0) {
      LOG(ERROR) << "AsModel::Warmup: invalid ALLSPARK_EXPECT_OUTLEN=" << outlen
                 << ", should be a non-negative integer no larger than "
                    "model max length "
                 << ctx_->GetModelMaxLength();
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }

    // assert: 0 <= outlen <= maxlen
    if (outlen > 0) {
      LOG(INFO) << "warm-up: using ALLSPARK_EXPECT_OUTLEN=" << outlen;
      maxlen = outlen;
    } else {
      LOG(INFO) << "warm-up: ALLSPARK_EXPECT_OUTLEN=0, use model max length "
                << ctx_->GetModelMaxLength();
    }
  } else {
    LOG(INFO) << "warm-up: envariable ALLSPARK_EXPECT_OUTLEN not found, "
                 "use model max length "
              << ctx_->GetModelMaxLength();
  }

#if ENABLE_SPAN_ATTENTION
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
  const size_t best_batch_size_z =
      cache_frame_manager_->CountFrame() /
      (size_t(2) * ctx_->GetDecoderLayer() *
       (maxlen + ctx_->GetCacheSpanSize() - 1) / ctx_->GetCacheSpanSize());
  if (best_batch_size_z > std::numeric_limits<int>::max()) {
    LOG(ERROR) << "AsModel::Warmup: best_batch_size exceeds int max, got "
               << best_batch_size_z;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  const int best_batch_size = static_cast<int>(best_batch_size_z);
  LOG(INFO) << "warm-up: recommended batch size is " << best_batch_size
            << ", current model max batch is " << ctx_->GetModelMaxBatch();
  }
#endif  // ENABLE_SPAN_ATTENTION
#endif

  return AsStatus::ALLSPARK_SUCCESS;
}

int64_t AsModel::GetAvailableMemoryBytes() {
  int64_t ret{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    ret = cache_allocator_->GetDeviceFreeMemory();
    LOG(INFO) << "AsModel: device available memory (MB): " << (ret >> 20);
#else
    LOG(WARNING)
        << "AsModel::GetAvailableMemoryBytes: span attention disabled, "
           "this function will always return 0";
#endif  // ENABLE_SPAN_ATTENTION
  }
  return ret;
}

int64_t AsModel::GetOccupiedMemoryBytes() {
  int64_t ret{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    ret = cache_allocator_->GetDeviceUsedMemory();
    LOG(INFO) << "AsModel: device occupied memory (MB): " << (ret >> 20);
#else
    LOG(WARNING) << "AsModel::GetOccupiedMemoryBytes: span attention disabled, "
                    "this function will always return 0";
#endif  // ENABLE_SPAN_ATTENTION
  }
  return ret;
}

int64_t AsModel::GetTotalMemoryBytes() {
  int64_t ret{0};
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
#if ENABLE_SPAN_ATTENTION
    ret = cache_allocator_->GetDeviceTotalMemory();
    LOG(INFO) << "AsModel: device total memory (MB): " << (ret >> 20);
#else
    LOG(WARNING) << "AsModel::GetTotalMemoryBytes: span attention disabled, "
                    "this function will always return 0";
#endif  // ENABLE_SPAN_ATTENTION
  }
  return ret;
}

#if ENABLE_SPAN_ATTENTION
int64_t AsModel::GetFreeFrame() {
  return cache_frame_manager_->CountFreeFrame();
}
#endif

void AsModel::UpdateAsEngineStat(AsEngineStat* as_stat) {
#if ENABLE_SPAN_ATTENTION
  if (cache_span_manager_ && cache_frame_manager_) {
    as_stat->span_size = cache_span_manager_->GetSpanSize();
    as_stat->total_span = cache_frame_manager_->CountFrame();
    as_stat->free_span = cache_frame_manager_->CountFreeFrame();
    as_stat->total_token = as_stat->total_span / (2 * ctx_->GetDecoderLayer()) *
                           ctx_->GetCacheSpanSize();
    as_stat->free_token = as_stat->free_span / (2 * ctx_->GetDecoderLayer()) *
                          ctx_->GetCacheSpanSize();
    as_stat->used_span = as_stat->total_span - as_stat->free_span;
    as_stat->token_usage_percentage =
        static_cast<float>(((as_stat->total_token - as_stat->free_token))) /
        (float)as_stat->total_token;
    if (prefix_cache_manager_ != nullptr) {
      prefix_cache_manager_->UpdateEngineStat(as_stat);
    }
  } else
#endif
  {
    as_stat->total_token = 0;
    as_stat->free_token = 0;
  }
  as_stat->pendding_request = (int)pending_request_queue_.size();
  as_stat->running_request = (int)runtime_ctx_->GetGenCtxListSize();
}

AsStatus AsModel::LoadLoraByName(const std::string& lora_name_or_path) {
  DLOG(INFO) << "AsModel::LoadLoraByName() " << lora_name_or_path << std::endl;
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  // check if lora already exists
  assert(lora_manager_ !=
         nullptr);  // 必须AsModel::Init()之后才能调用LoadLoraByName
  AsModelConfig lora_cfg =
      weight_handler_->GetModelConfig();  // copy cfg from base-model

  if (lora_manager_->GetNumLoras() >= lora_cfg.lora_max_num) {
    LOG(ERROR) << "lora number exceeds limit: " << lora_cfg.lora_max_num;
    return AsStatus::ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR;
  }
  auto lora_path_obj = util::Path(lora_cfg.weights_path);
  auto lora_dir = lora_path_obj.parent_path();
  auto lora_name = lora_name_or_path;
  auto lora_path = lora_dir + '/' + lora_name + ".aslora";
  if (lora_name_or_path.front() == '/') {  // only absolute path or name is
                                           // allowed, relative-path not allowed
    lora_path_obj = util::Path(lora_name_or_path);
    lora_path = lora_name_or_path;
    assert(lora_path_obj.extension() == ".aslora");
    lora_name = lora_path_obj.filename().substr(
        0, lora_path_obj.filename().find(".aslora"));
  }
  // lora_path, lora_name OK

  if (lora_manager_->IsLoraExists(lora_name)) {
    LOG(WARNING) << "lora " << lora_name << " already exists!";
    return ret;
  }
  // load lora
  lora_cfg.model_name = lora_name;
  lora_cfg.weights_path = lora_path;
  lora_cfg.model_path = "";  // lora不使用该字段, (no graph for lora)
  lora_cfg.is_lora_cfg = true;
  lora_cfg.lora_names.clear();  // lora should NOT have any sub-loras...
  auto& lora_weight_handle = lora_manager_->RegisterLora(lora_cfg);
  WeightSwapConfig swap_config;
  swap_config.enable =
      false;  // 由调用方来显式load_lora/unload_lora，所以对于lora禁用swap，来提升加载速度
  lora_manager_->SetSwapConfig(lora_weight_handle, swap_config);
  RankInfo rank_info = GetRankInfo();
  ret = lora_manager_->LoadWeightForModel(*ctx_, lora_weight_handle, rank_info);
  if (ret != AsStatus::ALLSPARK_SUCCESS)
    lora_manager_->UnRegisterLora(lora_name);  // rollback
  return ret;
}

AsStatus AsModel::UnloadLoraByName(const std::string& lora_name) {
  DLOG(INFO) << "AsModel::UnloadLoraByName()" << std::endl;
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  // check if lora already exists
  assert(lora_manager_ !=
         nullptr);  // 必须AsModel::Init()之后才能调用LoadLoraByName
  if (!lora_manager_->IsLoraExists(lora_name)) {
    LOG(WARNING) << "lora " << lora_name << " not exists!";
    return ret;
  }

  lora_manager_->UnRegisterLora(lora_name);
  // set lora tainted, currently only useful for GemmLora op
  for (auto& op : topo_ops_) {
    op->AddTaintedStatus(lora_name);
  }

  return ret;
}

std::string AsModel::GetOpProfilingInfo() {
  if (model_profiler_ == nullptr) {
    LOG(WARNING) << "AS_PROFILE env variable should be set to do profile, "
                    "export AS_PROFILE=ON";
    return {""};
  }
  std::stringstream ss;
  constexpr const char* tags[] = {"forward", "reshape", "alloc"};
  for (auto& tag : tags) {
    ss << "*** " << tag << " ***" << std::endl;
    auto res_stat = model_profiler_->ReportOpStat(tag);
    DLOG(INFO) << "res_stat size: " << res_stat.size() << std::endl;
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    ss << std::setfill(' ') << std::left << std::setw(10) << "rank" << std::left
       << std::setw(20) << "opname" << std::left << std::setw(10) << "count"
       << std::left << std::setw(10) << "min_ms" << std::left << std::setw(10)
       << "max_ms" << std::left << std::setw(10) << "ave_ms" << std::left
       << std::setw(15) << "total_ms" << std::left << std::setw(10)
       << "percentage" << std::endl;
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    for (auto& stat : res_stat) {
      DLOG(INFO) << std::fixed << " rank id: " << rank_
                 << " op name: " << stat.first
                 << " count: " << (long)stat.second[3] << std::setprecision(2)
                 << " min_ms: " << stat.second[0]
                 << " max_ms: " << stat.second[1]
                 << " ave_ms: " << stat.second[2]
                 << " total_ms: " << stat.second[4]
                 << " percentage(%): " << stat.second[5] << std::endl;
      ss << std::setfill(' ') << std::fixed << std::setprecision(2) << std::left
         << std::setw(10) << rank_ << std::left << std::setw(20) << stat.first
         << std::left << std::setw(10) << (long)stat.second[3] << std::left
         << std::setw(10) << stat.second[0] << std::left << std::setw(10)
         << stat.second[1] << std::left << std::setw(10) << stat.second[2]
         << std::left << std::setw(15) << stat.second[4] << std::left
         << std::setw(10) << stat.second[5] << std::endl;
    }
    ss << std::setfill('-') << std::setw(95) << "-" << std::endl;
    ss << std::endl;
  }
  return ss.str();
}
AsModel::~AsModel() {}

// --------------------------------------------------------------------------
// //

ModelFactory& ModelFactory::getInstance() {
  static ModelFactory model_factory;
  return model_factory;
}

ModelConstructor ModelFactory::GetModel(const std::string& model_type_str) {
  if (model_set_.find(model_type_str) == model_set_.end()) {
    LOG(ERROR) << "Unsupported model type : " << model_type_str << std::endl;
    throw AsException("Unsupported model type");
  }
  return model_set_[model_type_str];
}

void ModelFactory::Register(const std::string& model_type_str,
                            ModelConstructor model_constructor) {
  model_set_[model_type_str] = model_constructor;
}

}  // namespace allspark
