/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_worker.cpp
 */

#include "engine_worker.h"

#include <cpu/cpu_context.h>
#include <utility/file_util.h>

#include <fstream>

namespace allspark {
CpuWorker::CpuWorker(int rank, int nranks, int device_id)
    : Worker(rank, nranks, device_id) {
  device_ctx_ = std::make_unique<CPUContext>();
}
AsStatus CpuWorker::InitCCL(int rank, int nranks) {
  CPUContext* cpu_ctx = (CPUContext*)(device_ctx_.get());
  cpu_ctx->InitMCCL(rank, nranks_);
  rank_ = cpu_ctx->GetRank();
  nranks_ = cpu_ctx->GetNranks();
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus CpuWorker::SetNumThreads(int nums) {
  device_ctx_->SetNumThreads(nums);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus Worker::BuildModel(const TransformerProto& model_proto,
                            std::shared_ptr<WeightManager> weight_manager,
                            std::shared_ptr<ModelWeightHandler> weight_handler,
                            const DeviceContext* main_ctx) {
  DLOG(INFO) << "Worker::BuildModel()" << std::endl;
  SetWorkerDeviceId(device_id_);
  if (main_ctx != nullptr) {
    device_ctx_->CopyFromOther(main_ctx);
  }
  model_ = ModelFactory::getInstance().GetModel(model_proto.model_type())();
  model_->SetRank(rank_, nranks_);
  model_->SetWeightHandler(weight_manager, weight_handler);

  AS_CHECK_STATUS(model_->Init(model_proto, *device_ctx_));

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus Worker::StartRequestImpl(
    const std::shared_ptr<RequestHandle> request_handle, TensorMap* outputs,
    GenerateConfig& gen_cfg) {
  DLOG(INFO) << "Worker::StartRequestImpl" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->StartRequestImpl(request_handle, outputs, gen_cfg);
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
  DLOG(INFO) << "Worker::GetRequestById" << std::endl;
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
AsStatus Worker::RunTextGenerationContext() {
  DLOG(INFO) << "Worker::RunTextGenerationContext" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->GenerateContinueContext();
  return ret;
}

AsStatus Worker::AllocDecoderMemory() {
  DLOG(INFO) << "Worker::AllocDecoderMemory" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->AllocDecoderMemory();
  return ret;
}

AsStatus Worker::Warmup(int64_t bytes_available, int64_t bytes_per_req) {
  DLOG(INFO) << "Worker::Warmup" << std::endl;
  SetWorkerDeviceId(device_id_);
  AsStatus ret = model_->Warmup(bytes_available, bytes_per_req);
  return ret;
}

int64_t Worker::GetAvailableMemoryBytes() {
  DLOG(INFO) << "Worker::GetAvailableMemoryBytes" << std::endl;
  SetWorkerDeviceId(device_id_);
  return model_->GetAvailableMemoryBytes();
}

int Worker::GetUnFinishedRequest() { return model_->GetUnFinishedRequest(); }

AsStatus Worker::GetInformation(std::string* model_info) {
  DLOG(INFO) << "Worker::GetInformation()" << std::endl;
  SetWorkerDeviceId(device_id_);
  model_->GetInformation(model_info);
  return AsStatus::ALLSPARK_SUCCESS;
}

void Worker::ResetProfiler() { return model_->ResetProfiler(); }

std::string Worker::GetOpProfilingInfo() {
  return model_->GetOpProfilingInfo();
}

void Worker::UpdateAsEngineStat(AsEngineStat* as_stat) {
  return model_->UpdateAsEngineStat(as_stat);
}

};  // namespace allspark
