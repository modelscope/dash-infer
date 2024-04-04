/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_service.cpp
 */

#include <allspark.h>
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <chrono>
#include <csignal>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "allspark_service.grpc.pb.h"
#include "allspark_service_helper.h"
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
// using namespace allspark;

namespace allspark {
namespace allspark_service {
void PrintOut(DLManagedTensor* dl_tensor) {
  std::vector<std::vector<int64_t>> out;
  std::shared_ptr<allspark_service::DLTensorManager> dl_out_manager =
      std::make_shared<allspark_service::DLTensorManager>();
  dl_out_manager->ToVectorData(out, dl_tensor);
  std::cout << "generated_ids: " << std::endl;
  for (auto& vec : out) {
    std::cout << "output: ";
    for (auto& val : vec) {
      std::cout << val << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

class RequestManager {
 public:
  struct RequestData {
    allspark::RequestHandle_t handle;
    allspark::AsEngine::ResultQueue_t queue;
    std::shared_ptr<AsEngine::RequestContent> req;
    allspark_service::SharedDLTensorMap shared_inputs;
    std::vector<allspark_service::SharedDLTensorListMap> shared_mm_embedding;
  };

  void addRequest(const std::string& key,
                  const std::shared_ptr<RequestData>& req_data) {
    std::unique_lock<std::mutex> lock(mtx_);
    request_map_[key] = req_data;
  }

  void eraseRequest(const std::string& key) {
    std::unique_lock<std::mutex> lock(mtx_);
    request_map_.erase(key);
  }

  std::shared_ptr<RequestData> getRequest(const std::string& key) {
    auto it = request_map_.find(key);
    if (it != request_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  std::map<std::string, std::shared_ptr<RequestData>> request_map_;
  std::mutex mtx_;
};

class AllSparkServiceImpl final : public allspark_service::AllSpark::Service {
 public:
  AllSparkServiceImpl(int pid)
      : stop_service_(false),
        client_pid_(pid),
        req_manager_(std::make_unique<RequestManager>()),
        shut_down_promise_(std::make_unique<std::promise<void>>()) {
    allspark_service::as_rpc_init_log("as_service");
  };

  Status BuildModelFromConfigStruct(
      grpc::ServerContext* context,
      const allspark::allspark_service::ModelStructConfig* struct_config,
      allspark::allspark_service::AsStatus* response) override {
    DLOG(INFO) << "BuildModelFromConfigStruct";
    auto config =
        allspark_service::makeModelStructConfigAsFromProto(*struct_config);
    auto status = engine_.BuildModelFromConfigStruct(config);
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status ReloadModelToDeviceMemory(
      grpc::ServerContext* context,
      const allspark::allspark_service::ModelName* model_name,
      allspark::allspark_service::AsStatus* response) override {
    auto status =
        engine_.ReloadModelToDeviceMemory(model_name->model_name().c_str());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status UnloadModelFromDeviceMemory(
      grpc::ServerContext* context,
      const allspark::allspark_service::ModelName* model_name,
      allspark::allspark_service::AsStatus* response) override {
    auto status =
        engine_.UnloadModelFromDeviceMemory(model_name->model_name().c_str());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status GetModelInformation(
      grpc::ServerContext* context,
      const allspark::allspark_service::ModelName* model_name,
      allspark::allspark_service::ModelInfo* model_info) override {
    std::string info;
    auto status =
        engine_.GetModelInformation(model_name->model_name().c_str(), &info);
    model_info->set_model_info(info);
    return Status::OK;
  }

  Status GetFileInformation(
      grpc::ServerContext* context,
      const allspark::allspark_service::FileInformationRequest* request,
      allspark::allspark_service::FileInformationResponse* response) override {
    std::string info;
    auto file_info = engine_.GetFileInformation(request->model_path().c_str(),
                                                request->param_path().c_str());
    response->set_create_version_graph(file_info.create_version_graph);
    response->set_create_version_param(file_info.create_version_param);
    response->set_current_version_engine(file_info.current_version_engine);
    return Status::OK;
  }

  Status StartModel(grpc::ServerContext* context,
                    const allspark::allspark_service::ModelName* model_name,
                    allspark::allspark_service::AsStatus* response) override {
    auto status = engine_.StartModel(model_name->model_name().c_str());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }
  Status StopModel(grpc::ServerContext* context,
                   const allspark::allspark_service::ModelName* model_name,
                   allspark::allspark_service::AsStatus* response) override {
    auto status = engine_.StopModel(model_name->model_name().c_str());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }
  Status ReleaseModel(grpc::ServerContext* context,
                      const allspark::allspark_service::ModelName* model_name,
                      allspark::allspark_service::AsStatus* response) override {
    auto status = engine_.ReleaseModel(model_name->model_name().c_str());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status StartRequest(
      grpc::ServerContext* context,
      const allspark::allspark_service::StartRequestRequest* request_proto,
      allspark::allspark_service::StartRequestResponse* response) override {
    auto req_data = std::make_shared<RequestManager::RequestData>();
    req_data->req = std::make_shared<AsEngine::RequestContent>();
    std::string model_name;
    allspark_service::makeRequestParamsAsFromProto(
        model_name, req_data->req.get(), *request_proto,
        req_data->shared_inputs, req_data->shared_mm_embedding);
    auto status = engine_.StartRequest(model_name.c_str(), req_data->req,
                                       &req_data->handle, &req_data->queue);
    if (status == allspark::AsStatus::ALLSPARK_SUCCESS) {
      DLOG(INFO) << "add request data req: " << req_data->req
                 << " uuid: " << req_data->req->config.uuid;
      req_manager_->addRequest(req_data->req->config.uuid, req_data);
    }
    response->set_uuid(req_data->req->config.uuid);
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status StopRequest(
      grpc::ServerContext* context,
      const allspark::allspark_service::StopRequestRequest* request,
      allspark::allspark_service::StartRequestResponse* response) override {
    auto req_data = req_manager_->getRequest(request->uuid());
    if (!req_data) {
      LOG(ERROR) << "StopRequest RequestData was not found, uuid: "
                 << request->uuid();
      return Status::OK;
    }
    auto status =
        engine_.StopRequest(request->model_name().c_str(), req_data->handle);
    response->set_uuid(request->uuid());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status ReleaseRequest(
      grpc::ServerContext* context,
      const allspark::allspark_service::StopRequestRequest* request,
      allspark::allspark_service::StartRequestResponse* response) override {
    auto req_data = req_manager_->getRequest(request->uuid());
    if (!req_data) {
      LOG(ERROR) << "ReleaseRequest RequestData was not found, uuid: "
                 << request->uuid();
      return Status::OK;
    }
    DLOG(INFO) << "service release request uuid: " << request->uuid();
    auto status =
        engine_.ReleaseRequest(request->model_name().c_str(), req_data->handle);
    response->set_uuid(request->uuid());
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    req_manager_->eraseRequest(request->uuid());
    return Status::OK;
  }

  Status SyncRequest(
      grpc::ServerContext* context,
      const allspark::allspark_service::StopRequestRequest* request,
      allspark::allspark_service::StartRequestResponse* response) override {
    std::shared_ptr<RequestManager::RequestData> req_data = nullptr;
    if (!request->uuid().empty()) {
      req_data = req_manager_->getRequest(request->uuid());
    }
    auto status = engine_.SyncRequest(request->model_name().c_str(),
                                      req_data ? req_data->handle : nullptr);
    if (!request->uuid().empty()) {
      response->set_uuid(request->uuid());
    }
    response->set_as_status(static_cast<allspark_service::AS_STATUS>(status));
    return Status::OK;
  }

  Status GetAsEngineStat(
      grpc::ServerContext* context,
      const allspark::allspark_service::ModelName* model_name,
      allspark::allspark_service::AsEngineStat* response) override {
    auto eng_stat = engine_.GetAsEngineStat(model_name->model_name().c_str());
    allspark_service::makeAsEngineStatProtoFromAs(*response, eng_stat);
    return Status::OK;
  }

  Status GetVersionFull(
      grpc::ServerContext* context,
      const allspark::allspark_service::Empty* empty,
      allspark::allspark_service::VersionInfo* version_info) override {
    auto info = engine_.GetVersionFull();
    version_info->set_version_info(info);
    return Status::OK;
  }

  Status GetOpProfilingInfo(
      grpc::ServerContext* context,
      const allspark::allspark_service::ModelName* model_name,
      allspark::allspark_service::OpProfilingInfo* profiling_info) override {
    auto info = engine_.GetOpProfilingInfo(model_name->model_name().c_str());
    profiling_info->set_op_profiling_info(info);
    return Status::OK;
  }

  Status GetRankId(grpc::ServerContext* context,
                   const allspark::allspark_service::Empty* empty,
                   allspark::allspark_service::RankId* rank_id) override {
    auto rank = engine_.GetRankId();
    rank_id->set_rank_id(rank);
    return Status::OK;
  }

  Status GetRankNums(grpc::ServerContext* context,
                     const allspark::allspark_service::Empty* empty,
                     allspark::allspark_service::RankId* rank_id) override {
    auto rank = engine_.GetRankNums();
    rank_id->set_rank_id(rank);
    return Status::OK;
  }

  Status ShutdownService(
      grpc::ServerContext* context,
      const allspark::allspark_service::Empty* empty,
      allspark::allspark_service::AsStatus* response) override {
    LOG(INFO) << "ShutdownService";
    stop_service_ = true;
    shut_down_promise_->set_value();
    return Status::OK;
  }

  Status GenerateStatus(
      grpc::ServerContext* context,
      const allspark::allspark_service::UUID* uuid,
      allspark::allspark_service::GenerateRequestStatus* response) override {
    auto req_data = req_manager_->getRequest(uuid->uuid());
    if (!req_data) {
      LOG(ERROR) << "GenerateStatus RequestData was not found, uuid: "
                 << uuid->uuid();
      return Status::OK;
    }
    auto status = req_data->queue->GenerateStatus();
    response->set_status(
        static_cast<allspark_service::GENERATE_STATUS>(status));
    return Status::OK;
  }

  Status GeneratedLength(
      grpc::ServerContext* context,
      const allspark::allspark_service::UUID* uuid,
      allspark::allspark_service::GenerateLen* response) override {
    auto req_data = req_manager_->getRequest(uuid->uuid());
    if (!req_data) {
      LOG(ERROR) << "GeneratedLength RequestData was not found, uuid: "
                 << uuid->uuid();
      return Status::OK;
    }
    auto size = req_data->queue->GeneratedLength();
    response->set_len(size);
    return Status::OK;
  }

  Status Get(grpc::ServerContext* context,
             const allspark::allspark_service::UUID* uuid,
             allspark::allspark_service::GeneratedElements* response) override {
    auto req_data = req_manager_->getRequest(uuid->uuid());
    if (!req_data) {
      LOG(ERROR) << "Get RequestData was not found, uuid: " << uuid->uuid();
      return Status::OK;
    }
    auto ele = req_data->queue->Get();
    allspark_service::makeGeneratedElementsProtoFromAs(response, ele);
    return Status::OK;
  }

  Status GetNoWait(
      grpc::ServerContext* context,
      const allspark::allspark_service::UUID* uuid,
      allspark::allspark_service::GeneratedElements* response) override {
    auto req_data = req_manager_->getRequest(uuid->uuid());
    if (!req_data) {
      LOG(ERROR) << "Get RequestData was not found, uuid: " << uuid->uuid();
      return Status::OK;
    }
    auto ele = req_data->queue->GetNoWait();
    allspark_service::makeGeneratedElementsProtoFromAs(response, ele);
    return Status::OK;
  }

  std::string GetServerAddress(int rank_id) {
    std::stringstream ss;
    ss << base_addr_ << client_pid_ << "_rank_" << rank_id;
    return ss.str();
  }

  void RunServer(int rank_id) {
    auto server_address = GetServerAddress(rank_id);
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    server_ = std::move(server);
    LOG(INFO) << "Server listening on " << server_address << std::endl;
    std::thread exit([&]() {
      this->shut_down_promise_->get_future().wait();
      const std::chrono::milliseconds wait_duration =
          std::chrono::milliseconds(50);
      const std::chrono::time_point<std::chrono::system_clock> deadline =
          std::chrono::system_clock::now() + wait_duration;
      LOG(INFO) << "shutting down...";
      usleep(100000);
      server_->Shutdown(deadline);
    });
    server_->Wait();
    exit.join();
    LOG(INFO) << "Server return...";
  }

  void ShutdownService() {
    stop_service_ = true;
    shut_down_promise_->set_value();
  }

 private:
  const std::string base_addr_ = "unix:/tmp/allspark.pid_";
  allspark::AsEngine engine_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<std::promise<void>> shut_down_promise_;
  bool stop_service_;
  int client_pid_;
  std::unique_ptr<RequestManager> req_manager_;
};

class SignalManager {
 private:
  static inline std::function<void(int)> signal_handle_func_;
  static inline void SignalHandler(int signal) {
    LOG(INFO) << "Received signal: " << signal << std::endl;
    signal_handle_func_(signal);
  }

 public:
  SignalManager() = default;
  static void Init(std::function<void(int)> f) {
    signal_handle_func_ = f;
    std::signal(SIGTERM, &SignalManager::SignalHandler);
    std::signal(SIGABRT, &SignalManager::SignalHandler);
  }
};

}  // namespace allspark_service
}  // namespace allspark

int main(int argc, char** argv) {
  if (argc != 3) {
    LOG(ERROR) << "invalid params";
    return -1;
  }
  int client_pid = atoi(argv[1]);
  if (client_pid <= 0) {
    LOG(ERROR) << "client pid is invalid";
    return -1;
  }

  int rank_id = atoi(argv[2]);
  if (rank_id < 0) {
    LOG(ERROR) << "rank_id is invalid rank_id: " << rank_id;
    return -1;
  }

  allspark::allspark_service::AllSparkServiceImpl service(client_pid);
  allspark::allspark_service::SignalManager::Init(
      [&service](int signum) { service.ShutdownService(); });
  service.RunServer(rank_id);
  return 0;
}
