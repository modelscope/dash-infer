/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model.cpp
 */

#include "model.h"  // NOLINT

#include <common/engine_runtime.h>
#include <common/memory_reuser.h>
#include <core/operator/generate_opt/postprocess_id/postprocess_id_op.h>
#include <utility/arbiter.h>
#include <utility/file_util.h>
#include <utility/timer.h>
#include <weight/weight_manager.h>

#include <exception>
#include <iomanip>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#define DEBUG_GEN_LAYER 0
#define DEBUG_GEN_LAYER_SAVE_NPY 0
namespace allspark {
using std::string;
using std::vector;

AsModel::AsModel(const std::string& model_type)
    : model_type_(model_type), ctx_(nullptr) {}

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

AsStatus AsModel::Init(const TransformerProto& model_proto,
                       const DeviceContext& ctx) {
  DLOG(INFO) << "AsModel::Init()" << std::endl;

  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  ctx_ = &ctx;
  DeviceType device_type = ctx.GetDeviceType();
  const char* do_profile = std::getenv("AS_PROFILE");
  if (do_profile && std::string(do_profile) == "ON") {
    model_profiler_ = std::make_unique<ModelProfiler>(this);
  } else {
    model_profiler_ = nullptr;
  }

  auto rankInfo = this->GetRankInfo();
  AS_CHECK_STATUS(
      weight_manager_->LoadWeightForModel(ctx, weight_handler_, rankInfo));

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

  gen_ctx_ = std::make_unique<GenerateContext>();
  runtime_ctx_ = std::make_unique<RuntimeContext>();
  std::unique_ptr<AsTensor> rotary_step = std::make_unique<AsTensor>(
      "rotary_step", ctx_->GetDeviceType(), DataType::INT32);
  rotary_step->SetShape(Shape{ctx_->GetModelMaxBatch()});
  std::unique_ptr<AsTensor> rotary_inv_freq = std::make_unique<AsTensor>(
      "rotary_inv_freq", ctx_->GetDeviceType(), DataType::FLOAT32);
  rotary_inv_freq->SetShape(
      Shape{ctx_->GetModelMaxBatch() * ctx_->GetSizePerHead() / 2});

  std::shared_ptr<LayerCacheManager> layer_cache_manager =
      runtime_ctx_->CreateLayerCacheManager();
  layer_cache_manager->CreateCache("rotary_step", std::move(rotary_step));
  layer_cache_manager->CreateCache("rotary_inv_freq",
                                   std::move(rotary_inv_freq));
  auto& graph = model_proto.graphs();
  int nodes = 0;

  DLOG(INFO) << "Start process model graph.";
  for (auto& g_name : model_proto.graph_names()) {
    std::vector<std::unique_ptr<AsOperator>> ops;

    for (auto& op_proto : graph.at(g_name).ops()) {
      OpRegistType op_type(op_proto.op_type(), ctx.GetDeviceType());

      std::unique_ptr<AsOperator> op =
          OpFactory::getInstance().GetOperator(op_type)();

      AS_CHECK_STATUS(op->CallInit(op_proto, ctx, weight_manager_,
                                   weight_handler_, rankInfo, &tensors_,
                                   model_profiler_.get()));

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
  tensors_["input_ids"]->SetShape(Shape{1, ctx_->GetModelMaxLength()});
  tensors_["attention_mask"]->SetShape(Shape{1, ctx_->GetModelMaxLength()});
  tensors_["dec_ids"]->SetShape(
      Shape{ctx_->GetModelMaxBatch(), ctx_->GetModelMaxLength()});
  // shared tmp workspace for operator
  tensors_.insert(std::make_pair(
      "workspace", std::make_unique<AsTensor>("workspace", ctx.GetDeviceType(),
                                              DataType::INT8)));
  tensors_.insert(std::make_pair(
      "context_k_workspace",
      std::make_unique<AsTensor>("context_k_workspace", ctx.GetDeviceType(),
                                 DataType::INT8)));
  tensors_.insert(std::make_pair(
      "context_v_workspace",
      std::make_unique<AsTensor>("context_v_workspace", ctx.GetDeviceType(),
                                 DataType::INT8)));
  tensors_.insert(std::make_pair(
      "tmp_dec_ids", std::make_unique<AsTensor>(
                         "tmp_dec_ids", ctx.GetDeviceType(), DataType::INT64)));
  ctx_->Synchronize();
  tensors_["max_dec_ids"]->SetShape(
      Shape{ctx_->GetModelMaxBatch(), ctx_->GetModelMaxLength()});
  tensors_["tmp_dec_ids"]->SetShape(
      Shape{ctx_->GetModelMaxBatch(), ctx_->GetModelMaxLength()});
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

  DLOG(INFO) << "load model success.";

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::ReloadModelToDeviceMemory() {
  DLOG(INFO) << "AsModel::LoadWeightsFromBuffer()" << std::endl;
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

  const_cast<DeviceContext*>(ctx_)->ResetBlockPools();

  DLOG(INFO) << "AsModel::UnloadModelFromDeviceMemory() END" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsModel::PrintWeights() {}

AsStatus AsModel::buildGenContext(
    GenerateContext* gen_ctx, const std::shared_ptr<Request>& request) const {
  gen_ctx->gen_cfg = request->gen_cfg;
  gen_ctx->request = request;
  gen_ctx->k_cache_list = std::vector<std::unique_ptr<CacheMemory>>();
  gen_ctx->v_cache_list = std::vector<std::unique_ptr<CacheMemory>>();

  gen_ctx->engine_max_length = ctx_->GetModelMaxLength();
  gen_ctx->input_len = tensors_.at("input_ids")->GetShape()[1];
  gen_ctx->gen_cfg.input_len = gen_ctx->input_len;
  gen_ctx->max_length = ctx_->GetModelMaxLength();

  std::stringstream k_cache_tag;
  std::stringstream v_cache_tag;
  k_cache_tag << request->request_id << "_k";
  v_cache_tag << request->request_id << "_v";

  int state_size = sizeof(std::mt19937);
  gen_ctx->sample_state = std::make_unique<AsTensor>(
      "sample_state:" + request->request_id, DeviceType::CPU, DataType::INT8,
      DataMode::DENSE, Shape({state_size}));
  memset(gen_ctx->sample_state->GetDataPtr(), 0,
         gen_ctx->sample_state->GetSizeInByte());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::runDecoderContext() {
  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
  runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");
  GenerateContext* gen_ctx =
      (runtime_ctx_->GetGenCtx(runtime_ctx_->current_batch));
  GenerateConfig gen_cfg = gen_ctx->gen_cfg;

  DLOG(INFO) << "start run context ,uuid = " << gen_ctx->request->request_id
             << std::endl;
  const Shape& in_shape = tensors_["input_ids"]->GetShape();
  int batch_size = in_shape[0];
  int in_length = in_shape[1];
  gen_ctx->batch_size = batch_size;
  if (gen_cfg.do_sample && gen_cfg.num_beams == 1) {
    gen_ctx->generate_method = 0;  // sample
  } else {
    gen_ctx->generate_method = 1;  // beam_search
  }
  gen_ctx->num_beams = gen_cfg.num_beams;
  if (gen_cfg.max_length <= in_length) {
    LOG(ERROR) << "Invalid param: max_length <= input_length" << std::endl;
    return ErrorProcess(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  for (auto& graph : graph_ops_) {
    for (auto& op : graph_ops_[graph.first]) {
      op->SetGenerateContext(*gen_ctx);
    }
  }
  gen_ctx->only_decoder = true;
  gen_ctx->num_beams = 1;
  gen_ctx->step = 0;

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
#if DEBUG_GEN_LAYER
    op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
    DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
#endif
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in pre_graph" << std::endl;
      return ErrorProcess(status);
    }
  }

  // other op
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }

  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallAlloc(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "alloc failed in loop::decoder" << std::endl;
      return ErrorProcess(status);
    }
  }

  // pre_forward
  // first decoder for input_ids
  for (auto& op : graph_ops_["decoder"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER
    op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
    DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
#endif
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in decoder" << std::endl;
      return ErrorProcess(status);
    }
  }
  gen_ctx->in_length_bias =
      gen_ctx->input_len != 0 ? gen_ctx->input_len : in_length;
  gen_ctx->num_beams = gen_cfg.num_beams;
  tensors_["dec_ids"]->SetShape(Shape{batch_size * gen_ctx->num_beams, 1});
  tensors_["max_dec_ids"]->SetShape(
      Shape{batch_size * gen_ctx->num_beams, ctx_->GetModelMaxLength()});
  // LOG(INFO)<<"before gen_graph"<<std::endl;
  for (auto& op : graph_ops_["gen_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in gen_graph" << std::endl;
      return ErrorProcess(status);
    }
  }
  for (auto& op : graph_ops_["gen_graph"]) {
    AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER
    op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
    DO_ARBITRATE(rank_, nranks_, 0, op);
#endif
#endif
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in gen_graph" << std::endl;
      return ErrorProcess(status);
    }
  }
  for (auto& op : graph_ops_["post_graph"]) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in post" << std::endl;
      return ErrorProcess(status);
    }
    status = op->CallForward(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "forward failed in post" << std::endl;
      return ErrorProcess(status);
    }
  }
  gen_ctx->in_length_bias = 0;
  gen_ctx->step = in_length;
  DLOG(INFO) << "end run context ,uuid = " << gen_ctx->request->request_id
             << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::StartRequest(std::shared_ptr<Request> request) {
  DLOG(INFO) << "AsModel:StartRequest()" << std::endl;

  int batch = runtime_ctx_->GetGenCtxListSize();
  runtime_ctx_->PushBackGenCtx(std::make_unique<GenerateContext>());
  int batch_now = request->inputs["input_ids"]->GetShape()[0];
  int seq_now = request->inputs["input_ids"]->GetShape()[1];
  tensors_["input_ids"]->SetShape(Shape({batch_now, seq_now}));
  TensorUtils::DeepCopyWholeAsync(*tensors_["input_ids"],
                                  *request->inputs["input_ids"], ctx_);
  tensors_["attention_mask"]->SetShape(Shape({batch_now, seq_now}));
  // not use input["attention_mask"]
  // TensorUtils::DeepCopyWholeAsync(*tensors_["attention_mask"],
  //                                 *request->inputs["attention_mask"], ctx_);
  DeviceType backend = ctx_->GetDeviceType();
  AsTensor tmp_dec_ids = *tensors_["tmp_dec_ids"];
  tmp_dec_ids.SetShape(Shape({batch + 1, 1}));
  CopyData(tmp_dec_ids.GetDataPtr(), backend, tensors_["dec_ids"]->GetDataPtr(),
           backend, (batch) * sizeof(int64_t), ctx_);

  // gen_ctx must be built after setting tensors_
  GenerateContext* gen_ctx = runtime_ctx_->GetGenCtx(batch);
  AS_CHECK_STATUS(buildGenContext(gen_ctx, request));

  runtime_ctx_->is_context = true;
  runtime_ctx_->current_batch = batch;
  try {
    runDecoderContext();
  } catch (std::exception& e) {
    LOG(ERROR) << "runDecoderContext() Failed" << std::string(e.what())
               << "request_id = " << request->request_id;
    StopRequest(request->request_id);
    request->status = AsEngine::GenerateRequestStatus::GenerateInterrupted;
    throw e;
  }
  // context阶段成功结束
  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
  CopyData((int64_t*)tmp_dec_ids.GetDataPtr() + (batch), backend,
           tensors_["dec_ids"]->GetDataPtr(), backend, (1) * sizeof(int64_t),
           ctx_);
  tensors_["dec_ids"]->SetShape(Shape{batch + 1, 1});
  CopyData(tensors_["dec_ids"]->GetDataPtr(), backend, tmp_dec_ids.GetDataPtr(),
           backend, (batch + 1) * sizeof(int64_t), ctx_);
  tensors_["max_dec_ids"]->SetShape(
      Shape{batch + 1, ctx_->GetModelMaxLength()});
  DLOG(INFO)
      << "AsModel::StartRequest: context finish, restore ops with Reshape";
  for (AsOperator* op : topo_ops_) {
    AsStatus status = op->CallReshape(runtime_ctx_.get());
    if (status != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "reshape failed in topo_ops" << std::endl;
      return ErrorProcess(status);
    }
  }
  LOG(INFO) << "RunDecoderContext() Success ID: " << request->request_id;
  request->status = AsEngine::GenerateRequestStatus::ContextFinished;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::StopRequest(std::string request_id) {
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
  GenerateContext* gen_ctx = runtime_ctx_->GetGenCtx(request_idx);
  Request* request = gen_ctx->request.get();

  // free cache
  for (int i = 0; i < gen_ctx->k_cache_list.size(); i++) {
    gen_ctx->k_cache_list[i]->Free();
  }
  for (int i = 0; i < gen_ctx->v_cache_list.size(); i++) {
    gen_ctx->v_cache_list[i]->Free();
  }
  DLOG(INFO) << "AsModel::StopRequest: [" << request_id << "] cache released";
  request->extra_embedding.clear();
  int last_batch = runtime_ctx_->GetGenCtxListSize() - 1;
  ctx_->Synchronize();
  DeviceType backend = ctx_->GetDeviceType();
  if (request_idx != last_batch) {
    CopyData((int64_t*)tensors_["dec_ids"]->GetDataPtr() + request_idx, backend,
             (int64_t*)tensors_["dec_ids"]->GetDataPtr() + last_batch, backend,
             (1) * sizeof(int64_t), ctx_);
    CopyData((int64_t*)tensors_["max_dec_ids"]->GetDataPtr() +
                 request_idx * ctx_->GetModelMaxLength(),
             backend,
             (int64_t*)tensors_["max_dec_ids"]->GetDataPtr() +
                 last_batch * ctx_->GetModelMaxLength(),
             backend, (ctx_->GetModelMaxLength()) * sizeof(int64_t), ctx_);
  }
  tensors_["dec_ids"]->SetShape(
      Shape{(int64_t)runtime_ctx_->GetGenCtxListSize() - 1, 1});
  tensors_["max_dec_ids"]->SetShape(
      Shape{(int64_t)runtime_ctx_->GetGenCtxListSize() - 1,
            ctx_->GetModelMaxLength()});
  ctx_->Synchronize();
  runtime_ctx_->FinishRequest(request_idx);
  current_unfinished_request_--;
  LOG(INFO) << "Stop request with request id: " << request_id;
  if (runtime_ctx_->GetGenCtxListSize() > 0) {
    for (AsOperator* op : topo_ops_) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "reshape failed in topo_ops" << std::endl;
        return ErrorProcess(status);
      }
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::ReleaseRequest(std::string request_id) {
  if (all_request_map_.find(request_id) != all_request_map_.end()) {
    all_request_map_.erase(request_id);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

Request* AsModel::GetRequestById(std::string request_id) {
  if (all_request_map_.find(request_id) == all_request_map_.end()) {
    return nullptr;
  }
  return all_request_map_.at(request_id).get();
}

AsStatus AsModel::GenerateContinueContext() {
  // maybe use if,each turn only run one context phase
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  DLOG(INFO) << " gen ctx list " << runtime_ctx_->GetGenCtxListSize()
             << " pending init " << pending_request_queue_.size();

  if (!pending_request_queue_.empty() &&
      runtime_ctx_->GetGenCtxListSize() < ctx_->GetModelMaxBatch()) {
    std::shared_ptr<Request> request = pending_request_queue_.front();
    pending_request_queue_.pop();
    StartRequest(request);
    DLOG(INFO) << "RunContext SUCCESS ,request id = " << request->request_id;
    current_unfinished_request_.store(pending_request_queue_.size() +
                                      runtime_ctx_->GetGenCtxListSize());
  } else {
    // not free batch or not waitting reqeust
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::GenerateContinueDecoder() {
  DLOG(INFO) << "AsModel::GenerateContinueDecoder()" << std::endl;
  util::Timer t0;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);

  DLOG(INFO) << "pthread_self: " << (unsigned long)pthread_self();

  DLOG(INFO) << " gen ctx list " << runtime_ctx_->GetGenCtxListSize()
             << " pending init " << pending_request_queue_.size()
             << " t1(ms): " << t0.elapsed();

  current_unfinished_request_.store(pending_request_queue_.size() +
                                    runtime_ctx_->GetGenCtxListSize());
  DLOG(INFO) << " running task " << current_unfinished_request_.load()
             << " t2(ms):" << t0.elapsed();
  const int async_token_num = 1;
  for (int now_step = 0; now_step < async_token_num; now_step++) {
    int batch_size = runtime_ctx_->GetGenCtxListSize();
    if (batch_size == 0) {
      return AsStatus::ALLSPARK_EMPTY_REQUEST;
    }

    gen_ctx_->step++;
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_step");
    runtime_ctx_->GetLayerCacheManager()->ResetCache("rotary_inv_freq");
    for (auto& op : graph_ops_["decoder"]) {
      AsStatus status = op->CallAlloc(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
        return ErrorProcess(status);
      }
    }
    for (auto& op : graph_ops_["decoder"]) {
      AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, gen_ctx_->step, op);
#endif
#endif
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
        return ErrorProcess(status);
      }
    }

    for (int i = 0; i < batch_size; i++) {
      runtime_ctx_->GetGenCtx(i)->step += 1;
    }

    DLOG(INFO) << " decoder(ms): " << t0.elapsed();
    for (auto& op : graph_ops_["gen_graph"]) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::decoder" << std::endl;
        return ErrorProcess(status);
      }
    }
    for (auto& op : graph_ops_["gen_graph"]) {
      AsStatus status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, gen_ctx_->step, op);
#endif
#endif
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in loop::gen_graph" << std::endl;
        return ErrorProcess(status);
      }
    }

    DLOG(INFO) << " Generate Continue Decoder: time(ms): " << t0.elapsed();
    tensors_["max_dec_ids"]->SetShape(
        Shape{batch_size, ctx_->GetModelMaxLength()});
    for (auto& op : graph_ops_["post_graph"]) {
      AsStatus status = op->CallReshape(runtime_ctx_.get());
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "reshape failed in post" << std::endl;
        return ErrorProcess(status);
      }
      status = op->CallForward(runtime_ctx_.get());
#if DEBUG_GEN_LAYER
      op->PrintInformation();
#if DEBUG_GEN_LAYER_SAVE_NPY
      DO_ARBITRATE(rank_, nranks_, gen_ctx_->step, op);
#endif
#endif
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "forward failed in post" << std::endl;
        return ErrorProcess(status);
      }
    }

    DLOG(INFO) << " post(ms): " << t0.elapsed();
    // clean up the finished request
    for (int i = runtime_ctx_->GetGenCtxListSize() - 1; i >= 0; i--) {
      if (runtime_ctx_->GetGenCtx(i)->finish) {
        auto ret = StopRequest(runtime_ctx_->GetGenCtx(i)->request->request_id);
        if (ret != AsStatus::ALLSPARK_SUCCESS) return ret;
      }
    }

    DLOG(INFO) << " close(ms): " << t0.elapsed();
  }

  return AsStatus::ALLSPARK_STREAMING;
}

AsStatus AsModel::StartRequestImpl(
    const std::shared_ptr<RequestHandle> request_handle, TensorMap* outputs,
    GenerateConfig& gen_cfg) {
  DLOG(INFO) << "AsModel::StartRequestImpl()" << std::endl;
  std::shared_ptr<Request> request_ptr = std::make_shared<Request>(
      gen_cfg.uuid, *request_handle->inputs_internal, *outputs, gen_cfg);
  request_ptr->input_len = request_ptr->inputs["input_ids"]->GetShape()[1];
  DLOG(INFO) << "request_ptr->input_len=" << request_ptr->input_len;

  pending_request_queue_.push(request_ptr);

  all_request_map_[request_ptr->request_id] = request_ptr;
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsModel::GenerateContinue() {
  DLOG(INFO) << "AsModel::GenerateContinue()" << std::endl;
  AsStatus ret = GenerateContinueDecoder();
  AS_CHECK_STATUS(ret);
  return ret;
}
AsStatus AsModel::AllocDecoderMemory() {
  DLOG(INFO) << "AsModel::AllocDecoderMemory()" << std::endl;
  std::unique_lock<std::mutex> lock(gen_ctx_lock_);
  const int async_token_num = 1;
  runtime_ctx_->is_context = false;
  runtime_ctx_->current_batch = 0;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsModel::Warmup(int64_t bytes_available, int64_t bytes_per_req) {
  DLOG(INFO) << "AsModel::Warmup()";
  if (bytes_available < 0) {
    LOG(ERROR) << "AsModel::Warmup: bytes_available must be non-negative"
                  ", got "
               << bytes_available;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (bytes_per_req < 0) {
    LOG(ERROR) << "AsModel::Warmup: bytes_per_req must be non-negative"
                  ", got "
               << bytes_per_req;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  constexpr float runtime_mem_ratio = 1.1;
  LOG(INFO) << "warm-up: runtime memory reservation ratio: "
            << runtime_mem_ratio;
  const int64_t bytes_cache = std::max(
      0L, bytes_available - static_cast<int64_t>(
                                std::ceil(bytes_per_req * runtime_mem_ratio)));

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

  return AsStatus::ALLSPARK_SUCCESS;
}

int64_t AsModel::GetAvailableMemoryBytes() {
  int64_t bytes_pre_warmup{0};
  int64_t bytes_limit{0};
  return bytes_limit - bytes_pre_warmup;
}

void AsModel::UpdateAsEngineStat(AsEngineStat* as_stat) {
  as_stat->total_token = 0;
  as_stat->free_token = 0;
  as_stat->pendding_request = (int)pending_request_queue_.size();
  as_stat->running_request = (int)runtime_ctx_->GetGenCtxListSize();
}

void AsModel::ResetProfiler() {
  if (model_profiler_ == nullptr) {
    return;
  }

  model_profiler_->Reset();
}

std::string AsModel::GetOpProfilingInfo() {
  if (model_profiler_ == nullptr) {
    LOG(WARNING) << "AS_PROFILE env variable should be set to do profile, "
                    "export AS_PROFILE=ON";
    return std::string("");
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
