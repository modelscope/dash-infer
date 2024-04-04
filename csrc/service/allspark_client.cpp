/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_client.cpp
 */

#include "allspark_client.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "allspark_client_impl.h"

namespace allspark {
// tmp impl to make as client only in dependency of allspark_client
// all AsModelConfig in as client will use builder class
AsModelConfig::AsModelConfig() {}
AsModelConfig::AsModelConfig(
    std::string in_model_name, std::string in_model_path,
    std::string in_weights_path, std::string in_compute_unit,
    int in_engine_max_length, int in_engine_max_batch, bool in_text_graph,
    int in_num_threads, std::string in_matmul_precision,
    AsMHAPrefill in_prefill_mode, AsCacheMode in_cache_mode)
    : model_name(std::move(in_model_name)),
      model_path(std::move(in_model_path)),
      weights_path(std::move(in_weights_path)),
      compute_unit(std::move(in_compute_unit)),
      num_threads(in_num_threads),
      matmul_precision(in_matmul_precision),
      engine_max_length(in_engine_max_length),
      engine_max_batch(in_engine_max_batch),
      cache_mode(in_cache_mode),
      prefill_mode(in_prefill_mode),
      text_graph(in_text_graph) {}

static std::vector<std::string> g_errors;
static std::mutex g_errors_lock;
const std::string AsGetErrorByCode(AsStatus error_code) {
  switch (error_code) {
    case AsStatus::ALLSPARK_SUCCESS:
      return "ALLSPARK_SUCCESS";
    case AsStatus::ALLSPARK_STREAMING:
      return "ALLSPARK_STREAMING";
    case AsStatus::ALLSPARK_UNKNOWN_ERROR:
      return "ALLSPARK_UNKNOWN_ERROR";
    case AsStatus::ALLSPARK_PARAM_ERROR:
      return "ALLSPARK_PARAM_ERROR";
    case AsStatus::ALLSPARK_IO_ERROR:
      return "ALLSPARK_IO_ERROR";
    case AsStatus::ALLSPARK_MEMORY_ERROR:
      return "ALLSPARK_MEMORY_ERROR";
    case AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR:
      return "ALLSPARK_EXCEED_LIMIT_ERROR";
    case AsStatus::ALLSPARK_INVALID_CALL_ERROR:
      return "ALLSPARK_INVALID_CALL_ERROR";
    case AsStatus::ALLSPARK_EMPTY_REQUEST:
      return "ALLSPARK_EMPTY_REQUEST";
    case AsStatus::ALLSPARK_ILLEGAL_REQUEST_ID:
      return "ALLSPARK_ILLEGAL_REQUEST_ID";
    case AsStatus::ALLSPARK_CACHE_MEMORY_OUT:
      return "ALLSPARK_CACHE_MEMORY_OUT";
    case AsStatus::ALLSPARK_RUNTIME_ERROR:
      if (g_errors.size())
        return "ALLSPARK_RUNTIME_ERROR" + AsConcatErrors();
      else
        return "ALLSPARK_RUNTIME_ERROR";

    default:
      return "ALLSPARK_UNDEFINED_ERROR_CODE";
  }
}

void AsSaveError(const std::string& err_str) {
  std::lock_guard<std::mutex> guard(g_errors_lock);
  // one err_str per type is enough
  if (std::find(g_errors.begin(), g_errors.end(), err_str) == g_errors.end()) {
    g_errors.emplace_back(err_str);
  }
}

const std::string AsConcatErrors() {
  std::lock_guard<std::mutex> guard(g_errors_lock);
  std::stringstream ss;
  if (g_errors.size()) ss << "|";
  for (auto& err_str : g_errors) {
    ss << err_str << "#";
  }
  return ss.str();
}

void AsClearErrors() {
  std::lock_guard<std::mutex> guard(g_errors_lock);
  g_errors.clear();
}

AsClientEngine::AsClientEngine()
    : as_client_engine_impl_(std::make_unique<AsClientEngineImpl>()) {}
AsClientEngine::~AsClientEngine() {}
AsStatus AsClientEngine::BuildModelFromConfigStruct(
    AsModelConfig& model_config) {
  return as_client_engine_impl_->BuildModelFromConfigStruct(model_config);
}
AsStatus AsClientEngine::UnloadModelFromDeviceMemory(const char* model_name) {
  return as_client_engine_impl_->UnloadModelFromDeviceMemory(model_name);
}

AsStatus AsClientEngine::ReloadModelToDeviceMemory(const char* model_name) {
  return as_client_engine_impl_->ReloadModelToDeviceMemory(model_name);
}

AsStatus AsClientEngine::GetModelInformation(const char* model_name,
                                             std::string* model_info) {
  return as_client_engine_impl_->GetModelInformation(model_name, model_info);
}

AsFileInfo AsClientEngine::GetFileInformation(const char* as_model_path,
                                              const char* as_param_path) {
  return as_client_engine_impl_->GetFileInformation(as_model_path,
                                                    as_param_path);
}

AsStatus AsClientEngine::StartModel(const char* model_name) {
  return as_client_engine_impl_->StartModel(model_name);
}

AsStatus AsClientEngine::StopModel(const char* model_name) {
  return as_client_engine_impl_->StopModel(model_name);
}

AsStatus AsClientEngine::ReleaseModel(const char* model_name) {
  return as_client_engine_impl_->ReleaseModel(model_name);
}

AsStatus AsClientEngine::StartRequest(
    const char* model_name,
    std::shared_ptr<AsEngine::RequestContent> request_info,
    RequestHandle_t* request_handle, AsEngine::ResultQueue_t* queue) {
  return as_client_engine_impl_->StartRequest(model_name, request_info,
                                              request_handle, queue);
}

AsStatus AsClientEngine::StopRequest(const char* model_name,
                                     RequestHandle_t request_handle) {
  return as_client_engine_impl_->StopRequest(model_name, request_handle);
}

AsStatus AsClientEngine::ReleaseRequest(const char* model_name,
                                        RequestHandle_t request_handle) {
  return as_client_engine_impl_->ReleaseRequest(model_name, request_handle);
}

AsStatus AsClientEngine::SyncRequest(const char* model_name,
                                     RequestHandle_t request_handle) {
  return as_client_engine_impl_->SyncRequest(model_name, request_handle);
}

AsEngineStat AsClientEngine::GetAsEngineStat(const char* model_name) {
  return as_client_engine_impl_->GetAsEngineStat(model_name);
}

std::string AsClientEngine::GetVersionFull() {
  return as_client_engine_impl_->GetVersionFull();
}

std::string AsClientEngine::GetOpProfilingInfo(const char* model_name) {
  return as_client_engine_impl_->GetOpProfilingInfo(model_name);
}

int AsClientEngine::GetRankId() { return as_client_engine_impl_->GetRankId(); }

bool AsClientEngine::IsAllSparkWorkAsService() {
  // it must return true if code runs here
  return true;
}
}  // namespace allspark
