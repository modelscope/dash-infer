/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_client_impl.h
 */

#pragma once
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "allspark_client.h"
#include "allspark_service.grpc.pb.h"
#include "engine_runtime.h"
namespace allspark {

class AsClientContext final {
 public:
  static AsClientContext& GetInstance() {
    static AsClientContext myInstance;
    return myInstance;
  }
  AsClientContext(AsClientContext const&) = delete;  // Copy construct
  AsClientContext(AsClientContext&&) = delete;       // Move construct
  AsClientContext& operator=(AsClientContext const&) = delete;  // Copy assign
  AsClientContext& operator=(AsClientContext&&) = delete;       // Move assign
  int GetServiceNums() { return context_size_; }
  int GetClientId() { return client_pid_; }
  void GetServiceStubs(std::vector<allspark_service::AllSpark::Stub*>& stubs) {
    stubs.clear();
    for (auto& stub_ptr : stub_) {
      stubs.push_back(stub_ptr.get());
    }
  }
  bool CheckServiceLaunched() { return lauch_success_; }

 private:
  AsClientContext();
  ~AsClientContext();
  void RegisterService(const std::vector<std::string>& server_path);
  AsStatus LaunchService();
  void ShutdownService();
  int CheckServiceAlive();
  std::vector<std::unique_ptr<allspark_service::AllSpark::Stub>> stub_;
  int context_size_;
  // address: base_addr_ + client_pid_ + "_rank_" + numa_index
  const std::string base_addr_ = "unix:/tmp/allspark.pid_";
  int client_pid_;
  bool lauch_success_ = false;
};

class ClientResultQueueImpl : public AsEngine::ResultQueue {
 public:
  ClientResultQueueImpl(const std::string& uuid) : uuid_(uuid) {
    context_size_ = AsClientContext::GetInstance().GetServiceNums();
    AsClientContext::GetInstance().GetServiceStubs(stub_);
  }
  virtual AsEngine::GenerateRequestStatus GenerateStatus();
  virtual size_t GeneratedLength();
  virtual std::shared_ptr<AsEngine::GeneratedElements> Get();
  virtual std::shared_ptr<AsEngine::GeneratedElements> GetNoWait();

 private:
  std::vector<allspark_service::AllSpark::Stub*> stub_;
  int context_size_;
  std::string uuid_;
};

class ClientRequestManager {
 public:
  struct ClientRequestData {
    std::shared_ptr<RequestHandle> handle;
    std::shared_ptr<AsEngine::ResultQueue> queue;
  };

  void addRequest(const std::string& key,
                  std::shared_ptr<ClientRequestData> req_data) {
    std::unique_lock<std::mutex> lock(mtx_);
    request_map_[key] = req_data;
  }

  void eraseRequest(const std::string& key) {
    std::unique_lock<std::mutex> lock(mtx_);
    request_map_.erase(key);
  }

  std::shared_ptr<ClientRequestData> getRequest(const std::string& key) {
    auto it = request_map_.find(key);
    if (it != request_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  std::map<std::string, std::shared_ptr<ClientRequestData>> request_map_;
  std::mutex mtx_;
};

class AsClientEngineImpl final {
 public:
  enum class ModelOperation {
    Start,
    Stop,
    Release,
  };

  enum class RequestOperation {
    Start,
    Stop,
    Release,
    Sync,
  };

  AsClientEngineImpl();
  ~AsClientEngineImpl();

  // Context Config API
  AsStatus BuildModelFromConfigStruct(AsModelConfig& model_config);
  AsStatus GetModelInformation(const char* model_name, std::string* model_info);
  AsFileInfo GetFileInformation(const char* as_model_path,
                                const char* as_param_path);
  AsStatus StartModel(const char* model_name);
  AsStatus StopModel(const char* model_name);
  AsStatus ReleaseModel(const char* model_name);

  AsStatus StartRequest(const char* model_name,
                        std::shared_ptr<AsEngine::RequestContent> request_info,
                        RequestHandle_t* request_handle,
                        AsEngine::ResultQueue_t* queue);
  AsStatus StopRequest(const char* model_name, RequestHandle_t request_handle);
  AsStatus ReleaseRequest(const char* model_name,
                          RequestHandle_t request_handle);

  // sync request handle, nullptr for all requests on this model.
  AsStatus SyncRequest(const char* model_name, RequestHandle_t request_handle);
  int64_t GetFreeFrame(const char* model_name);
  AsEngineStat GetAsEngineStat(const char* model_name);

  AsStatus UnloadModelFromDeviceMemory(const char* model_name);
  AsStatus ReloadModelToDeviceMemory(const char* model_name);

  std::string GetVersionFull();
  std::string GetOpProfilingInfo(const char* model_name);
  int GetRankId();

 private:
  AsStatus CallModelOperation(const char* model_name, ModelOperation op);
  AsStatus CallRequestOperation(
      RequestOperation op, const char* model_name,
      allspark::RequestHandle_t* request_handle,
      allspark::AsEngine::ResultQueue_t* queue,
      std::shared_ptr<allspark::AsEngine::RequestContent> request_info);
  std::vector<allspark_service::AllSpark::Stub*> stub_;
  int context_size_;
  std::unique_ptr<ClientRequestManager> req_manager_;
  std::mutex mtx_;
};

}  // namespace allspark