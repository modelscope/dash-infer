/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_worker.cpp
 */

#include "engine_worker.h"
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

#include <cpu/cpu_context.h>
#include <utility/file_util.h>

#include <fstream>

namespace allspark {

#ifdef ENABLE_CUDA

thread_local int CudaWorker::last_device_id_of_this_thread_ = -1;

CudaWorker::CudaWorker(int rank, int nranks, const ncclUniqueId& id,
                       int device_id)
    : Worker(rank, nranks, device_id), nccl_id_(id) {
  // don't call virtual function in constructor.
  device_ctx_ = std::make_unique<CUDAContext>();
  device_ctx_->SetDeviceId(device_id_);
}

void CudaWorker::Init() { SetWorkerDeviceId(device_id_); }

AsStatus CudaWorker::InitCCL(int rank, int nranks) {
  CUDAContext* cu_ctx_ = (CUDAContext*)(device_ctx_.get());
  SetWorkerDeviceId(device_id_);
  cu_ctx_->InitNCCL(rank, nccl_id_, nranks_);
  return AsStatus::ALLSPARK_SUCCESS;
}
void CudaWorker::SetWorkerDeviceId(int device_id) {
  DLOG(INFO) << "set worker device id: " << device_id
             << " local value: " << last_device_id_of_this_thread_;
  if (last_device_id_of_this_thread_ == device_id) {
    return;
  } else {
    cudaSetDevice(device_id);
    last_device_id_of_this_thread_ = device_id;
  }
}

CudaWorker::~CudaWorker() { last_device_id_of_this_thread_ = -1; }
#endif

CpuWorker::CpuWorker(int rank, int nranks, int device_id)
    : Worker(rank, nranks, device_id) {
  device_ctx_ = std::make_unique<CPUContext>();
}

void CpuWorker::Init() {}

AsStatus CpuWorker::InitCCL(int rank, int nranks) {
#ifdef ENABLE_MULTINUMA
  CPUContext* cpu_ctx = (CPUContext*)(device_ctx_.get());
  cpu_ctx->InitMCCL(rank, nranks_);
  rank_ = cpu_ctx->GetRank();
  nranks_ = cpu_ctx->GetNranks();
  return AsStatus::ALLSPARK_SUCCESS;
#else
  LOG(ERROR) << "Multi-NUMA codes are not compiled" << std::endl;
  return AsStatus::ALLSPARK_RUNTIME_ERROR;
#endif
}

AsStatus CpuWorker::SetNumThreads(int nums) {
  device_ctx_->SetNumThreads(nums);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus Worker::BuildModel(
    const TransformerProto& model_proto,
    std::shared_ptr<WeightManager> weight_manager,
    std::shared_ptr<ModelWeightHandler> weight_handler,
    const DeviceContext* main_ctx,
    PrefixCacheCoordinator::Ptr prefix_cache_coordinator) {
  DLOG(INFO) << "Worker::BuildModel()" << std::endl;
  SetWorkerDeviceId(device_id_);
  if (main_ctx != nullptr) {
    device_ctx_->CopyFromOther(main_ctx);
  }
  model_ = ModelFactory::getInstance().GetModel(model_proto.model_type())();
  model_->SetRank(rank_, nranks_);
  model_->SetWeightHandler(weight_manager, weight_handler);
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefix_cache_coordinator != nullptr) {
      model_->SetPrefixCacheCoordinator(prefix_cache_coordinator);
    }
  }
#endif
  AS_CHECK_STATUS(model_->Init(model_proto, *device_ctx_));

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus Worker::StartRequestImpl(
    const std::shared_ptr<RequestHandle> request_handle, std::string request_id,
    TensorMap* outputs, const GenerateConfig& gen_cfg) {
  DLOG(INFO) << "Worker::StartRequestImpl" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret =
      model_->StartRequestImpl(request_handle, request_id, outputs, gen_cfg);
  return ret;
}

AsStatus Worker::UnloadModelFromDeviceMemory() {
  SetWorkerDeviceId(device_id_);
  AS_CHECK_STATUS(model_->UnloadModelFromDeviceMemory());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus Worker::RebuildModelFromBuffer(
    const std::unique_ptr<TransformerProto>& model_ir) {
  DLOG(INFO) << "Worker::RebuildModelFromBuffer()" << std::endl;
  SetWorkerDeviceId(device_id_);
  model_->SetRank(rank_, nranks_);
  // let weight manager restore the weights
  model_->ReloadModelToDeviceMemory();
  AS_CHECK_STATUS(model_->Init(*model_ir, *device_ctx_));
  return AsStatus::ALLSPARK_SUCCESS;
}
Request* Worker::GetRequestById(std::string request_id) {
  SetWorkerDeviceId(device_id_);
  return model_->GetRequestById(request_id);
}
AsStatus Worker::StopRequest(std::string request_id) {
  DLOG(INFO) << "Worker::StopRequest" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->StopRequest(request_id);
  return ret;
}
AsStatus Worker::ReleaseRequest(std::string request_id) {
  DLOG(INFO) << "Worker::ReleaseRequest" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->ReleaseRequest(request_id);
  return ret;
}
AsStatus Worker::RunTextGenerationContinue() {
  DLOG(INFO) << "Worker::RunTextGenerationContinue" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->GenerateContinue();
  return ret;
}
AsStatus Worker::RunTextGenerationContext(bool is_new_context) {
  DLOG(INFO) << "Worker::RunTextGenerationContext" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->GenerateContinueContext(is_new_context);
  return ret;
}

AsStatus Worker::AllocDecoderMemory() {
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->AllocDecoderMemory();
  return ret;
}

AsStatus Worker::Warmup(int64_t bytes_available, int64_t bytes_runtime) {
  DLOG(INFO) << "Worker::Warmup" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->Warmup(bytes_available, bytes_runtime);
  return ret;
}

int64_t Worker::GetAvailableMemoryBytes() {
  DLOG(INFO) << "Worker::GetAvailableMemoryBytes" << std::endl;
  SetWorkerDeviceId(device_id_);
  return model_->GetAvailableMemoryBytes();
}

int64_t Worker::GetOccupiedMemoryBytes() {
  DLOG(INFO) << "Worker::GetOccupiedMemoryBytes" << std::endl;
  SetWorkerDeviceId(device_id_);
  return model_->GetOccupiedMemoryBytes();
}

int64_t Worker::GetTotalMemoryBytes() {
  DLOG(INFO) << "Worker::GetTotalMemoryBytes" << std::endl;
  SetWorkerDeviceId(device_id_);
  return model_->GetTotalMemoryBytes();
}

int Worker::GetUnFinishedRequest() { return model_->GetUnFinishedRequest(); }

AsStatus Worker::GetInformation(std::string* model_info) {
  DLOG(INFO) << "Worker::GetInformation()" << std::endl;
  SetWorkerDeviceId(device_id_);
  model_->GetInformation(model_info);
  return AsStatus::ALLSPARK_SUCCESS;
}

std::string Worker::GetOpProfilingInfo() {
  return model_->GetOpProfilingInfo();
}

AsStatus Worker::LoadLoraByName(const std::string lora_name) {
  return model_->LoadLoraByName(lora_name);
}

AsStatus Worker::UnloadLoraByName(const std::string lora_name) {
  return model_->UnloadLoraByName(lora_name);
}

void Worker::UpdateAsEngineStat(AsEngineStat* as_stat) {
  return model_->UpdateAsEngineStat(as_stat);
}
#if ENABLE_SPAN_ATTENTION
int64_t Worker::GetFreeFrame() { return model_->GetFreeFrame(); }
#endif
};  // namespace allspark
