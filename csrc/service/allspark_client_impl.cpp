/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_client_impl.cpp
 */

#include "allspark_client_impl.h"

#include <unistd.h>

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "allspark_service_helper.h"
#include "allspark_service_parallel.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

namespace allspark {
#define CALL_RPC_FUNC(STUB, FUNC, CONTEXT, REQ, RESP, OUT_STATUS)       \
  do {                                                                  \
    *(OUT_STATUS) = (STUB)->FUNC((CONTEXT), (REQ), (RESP));             \
    if (!(OUT_STATUS)->ok()) {                                          \
      LOG(ERROR) << "RPC failed RESP: "                                 \
                 << static_cast<int>((RESP)->as_status()) << std::endl; \
      (RESP)->set_as_status(                                            \
          allspark_service::AS_STATUS::ALLSPARK_UNKNOWN_ERROR);         \
    }                                                                   \
  } while (0)

#define AS_CLIENT_CHECK_SERVICE_IS_RUNNING()                       \
  do {                                                             \
    if (!AsClientContext::GetInstance().CheckServiceLaunched()) {  \
      LOG(ERROR) << "service lauch failure, please check if set "  \
                    "AS_NUMA_NUM and AS_DAEMON_PATH env varialbe"; \
      return AsStatus::ALLSPARK_INVALID_CALL_ERROR;                \
    }                                                              \
  } while (0)

#define AS_CLIENT_CHECK_CPU_DEVICE_TYPE(device_tpye)    \
  do {                                                  \
    if (device_tpye.find("CPU") == std::string::npos) { \
      LOG(ERROR) << "check device type is CPU";         \
      return AsStatus::ALLSPARK_INVALID_CALL_ERROR;     \
    }                                                   \
  } while (0)

#define AS_CLIENT_CHECK_STATUS(size, response)                           \
  do {                                                                   \
    for (int i = 0; i < size; i++) {                                     \
      if (response[i].as_status() !=                                     \
          allspark_service::AS_STATUS::ALLSPARK_SUCCESS) {               \
        return static_cast<allspark::AsStatus>(response[i].as_status()); \
      }                                                                  \
    }                                                                    \
  } while (0)

#define AS_CLIENT_GET_SUCCESS_STATUS(size, response, success) \
  do {                                                        \
    for (int i = 0; i < size; i++) {                          \
      if (response[i].as_status() !=                          \
          allspark_service::AS_STATUS::ALLSPARK_SUCCESS) {    \
        success = false;                                      \
      }                                                       \
    }                                                         \
  } while (0)

allspark::AsEngine::GenerateRequestStatus
ClientResultQueueImpl::GenerateStatus() {
  // only call rank 0 version info
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return empty";
    return allspark::AsEngine::GenerateRequestStatus::Init;
  }
  allspark::allspark_service::GenerateRequestStatus status_proto;
  allspark_service::UUID uuid_proto;
  uuid_proto.set_uuid(uuid_);
  grpc::ClientContext context;
  auto status = stub_[0]->GenerateStatus(&context, uuid_proto, &status_proto);
  if (!status.ok()) {
    return allspark::AsEngine::GenerateRequestStatus::GenerateInterrupted;
  }
  return static_cast<allspark::AsEngine::GenerateRequestStatus>(
      status_proto.status());
}
size_t ClientResultQueueImpl::GeneratedLength() {
  // only call rank 0 version info
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return 0";
    return 0;
  }
  allspark::allspark_service::GenerateLen gen_len_proto;
  allspark_service::UUID uuid_proto;
  uuid_proto.set_uuid(uuid_);
  grpc::ClientContext context;
  auto status = stub_[0]->GeneratedLength(&context, uuid_proto, &gen_len_proto);
  if (!status.ok()) {
    return 0;
  }
  return static_cast<size_t>(gen_len_proto.len());
}
std::shared_ptr<AsEngine::GeneratedElements> ClientResultQueueImpl::Get() {
  // only call rank 0 version info
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return nullptr";
    return nullptr;
  }
  allspark::allspark_service::GeneratedElements ele_proto;
  allspark_service::UUID uuid_proto;
  uuid_proto.set_uuid(uuid_);
  grpc::ClientContext context;
  auto status = stub_[0]->Get(&context, uuid_proto, &ele_proto);
  if (!status.ok()) {
    return nullptr;
  }
  auto new_ele = std::make_shared<allspark::AsEngine::GeneratedElements>();
  allspark_service::makeGeneratedElementsAsFromProto(&ele_proto, new_ele);
  return new_ele;
}
std::shared_ptr<AsEngine::GeneratedElements>
ClientResultQueueImpl::GetNoWait() {
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return nullptr";
    return nullptr;
  }
  allspark::allspark_service::GeneratedElements ele_proto;
  allspark_service::UUID uuid_proto;
  uuid_proto.set_uuid(uuid_);
  grpc::ClientContext context;
  auto status = stub_[0]->GetNoWait(&context, uuid_proto, &ele_proto);
  if (!status.ok()) {
    return nullptr;
  }
  auto new_ele = std::make_shared<allspark::AsEngine::GeneratedElements>();
  allspark_service::makeGeneratedElementsAsFromProto(&ele_proto, new_ele);
  return new_ele;
}

AsClientEngineImpl::AsClientEngineImpl()
    : req_manager_(std::make_unique<ClientRequestManager>()) {
  context_size_ = AsClientContext::GetInstance().GetServiceNums();
  AsClientContext::GetInstance().GetServiceStubs(stub_);
  LOG(INFO) << "stub_ size: " << stub_.size();
}
AsClientEngineImpl::~AsClientEngineImpl() {}
AsStatus AsClientEngineImpl::BuildModelFromConfigStruct(
    AsModelConfig& model_config) {
  AS_CLIENT_CHECK_SERVICE_IS_RUNNING();
  AS_CLIENT_CHECK_CPU_DEVICE_TYPE(model_config.compute_unit);
  allspark_service::AsStatus as_status[context_size_];
  grpc::Status rpc_status[context_size_];
  grpc::ClientContext context[context_size_];
  allspark_service::ModelStructConfig model_struct_proto;
  allspark_service::makeModelStructConfigProtoFromAs(model_struct_proto,
                                                     model_config);
  allspark_service::parallel_loop(0, context_size_, [&](int id) {
    CALL_RPC_FUNC(stub_[id], BuildModelFromConfigStruct, &context[id],
                  model_struct_proto, &as_status[id], &rpc_status[id]);
  });
  AS_CLIENT_CHECK_STATUS(context_size_, as_status);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsClientEngineImpl::UnloadModelFromDeviceMemory(
    const char* model_name) {
  // TBD
  AS_CLIENT_CHECK_SERVICE_IS_RUNNING();
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsClientEngineImpl::ReloadModelToDeviceMemory(const char* model_name) {
  // TBD
  AS_CLIENT_CHECK_SERVICE_IS_RUNNING();
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsClientEngineImpl::GetModelInformation(const char* model_name,
                                                 std::string* model_info) {
  // only call rank0
  AS_CLIENT_CHECK_SERVICE_IS_RUNNING();
  allspark_service::ModelName model_name_proto;
  model_name_proto.set_model_name(model_name);
  grpc::ClientContext context;
  allspark_service::ModelInfo model_info_proto;
  auto rpc_status = stub_[0]->GetModelInformation(&context, model_name_proto,
                                                  &model_info_proto);
  *model_info = model_info_proto.model_info();
  return AsStatus::ALLSPARK_SUCCESS;
}

AsFileInfo AsClientEngineImpl::GetFileInformation(const char* as_model_path,
                                                  const char* as_param_path) {
  // only call rank0
  AsFileInfo as_file;
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, please check if set "
                  "AS_NUMA_NUM and AS_DAEMON_PATH env varialbe";
    return as_file;
  }
  allspark_service::FileInformationRequest file_in_proto;
  file_in_proto.set_model_path(as_model_path);
  file_in_proto.set_param_path(as_param_path);
  grpc::ClientContext context;
  allspark_service::FileInformationResponse file_out_proto;
  auto rpc_status =
      stub_[0]->GetFileInformation(&context, file_in_proto, &file_out_proto);
  as_file.create_version_graph = file_out_proto.create_version_graph();
  as_file.create_version_param = file_out_proto.create_version_param();
  as_file.current_version_engine = file_out_proto.current_version_engine();
  return as_file;
}

AsStatus AsClientEngineImpl::CallModelOperation(const char* model_name,
                                                ModelOperation op) {
  AS_CLIENT_CHECK_SERVICE_IS_RUNNING();
  allspark_service::AsStatus as_status[context_size_];
  grpc::Status rpc_status[context_size_];
  grpc::ClientContext context[context_size_];
  allspark_service::ModelName model_name_proto;
  model_name_proto.set_model_name(model_name);
  switch (op) {
    case ModelOperation::Start:
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], StartModel, &context[id], model_name_proto,
                      &as_status[id], &rpc_status[id]);
      });
      break;
    case ModelOperation::Stop:
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], StopModel, &context[id], model_name_proto,
                      &as_status[id], &rpc_status[id]);
      });
      break;
    case ModelOperation::Release:
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], ReleaseModel, &context[id], model_name_proto,
                      &as_status[id], &rpc_status[id]);
      });
      break;
    default:
      break;
  }
  AS_CLIENT_CHECK_STATUS(context_size_, as_status);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsClientEngineImpl::CallRequestOperation(
    RequestOperation op, const char* model_name,
    allspark::RequestHandle_t* request_handle,
    allspark::AsEngine::ResultQueue_t* queue,
    std::shared_ptr<allspark::AsEngine::RequestContent> request_info) {
  AS_CLIENT_CHECK_SERVICE_IS_RUNNING();
  allspark::allspark_service::StartRequestResponse
      response_proto[context_size_];
  grpc::Status rpc_status[context_size_];
  grpc::ClientContext context[context_size_];
  std::unique_lock<std::mutex> lock(mtx_);
  switch (op) {
    case RequestOperation::Start: {
      allspark::allspark_service::StartRequestRequest request_proto;
      std::string model_name_str = std::string(model_name);
      allspark_service::makeRequestParamsProtoFromAs(
          model_name_str, request_info.get(), request_proto);
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], StartRequest, &context[id], request_proto,
                      &response_proto[id], &rpc_status[id]);
      });
      bool success = true;
      AS_CLIENT_GET_SUCCESS_STATUS(context_size_, response_proto, success);
      if (success) {
        // create handle and queues
        auto req_data = std::make_shared<ClientRequestManager::RequestData>();
        auto uuid = response_proto[0].uuid();
        req_data->handle = std::make_shared<RequestHandle>(uuid);
        req_data->queue = std::make_shared<ClientResultQueueImpl>(uuid);
        req_manager_->addRequest(uuid, req_data);
        *request_handle = req_data->handle.get();
        *queue = req_data->queue.get();
      }
      break;
    }
    case RequestOperation::Stop: {
      allspark::allspark_service::StopRequestRequest request_proto;
      std::string model_name_str = std::string(model_name);
      request_proto.set_uuid((*request_handle)->uuid_);
      request_proto.set_model_name(model_name_str);
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], StopRequest, &context[id], request_proto,
                      &response_proto[id], &rpc_status[id]);
      });
      break;
    }
    case RequestOperation::Release: {
      allspark::allspark_service::StopRequestRequest request_proto;
      std::string model_name_str = std::string(model_name);
      request_proto.set_uuid((*request_handle)->uuid_);
      request_proto.set_model_name(model_name_str);
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], ReleaseRequest, &context[id], request_proto,
                      &response_proto[id], &rpc_status[id]);
      });
      bool success = true;
      AS_CLIENT_GET_SUCCESS_STATUS(context_size_, response_proto, success);
      if (success) {
        auto uuid = response_proto[0].uuid();
        req_manager_->eraseRequest(uuid);
      }
      DLOG(INFO) << "Client Release request uuid: " << (*request_handle)->uuid_
                 << " response_proto[0].uuid(): " << response_proto[0].uuid();
      break;
    }
    case RequestOperation::Sync: {
      allspark::allspark_service::StopRequestRequest request_proto;
      std::string model_name_str = std::string(model_name);
      if (*request_handle != nullptr) {
        request_proto.set_uuid((*request_handle)->uuid_);
      }
      request_proto.set_model_name(model_name_str);
      allspark_service::parallel_loop(0, context_size_, [&](int id) {
        CALL_RPC_FUNC(stub_[id], SyncRequest, &context[id], request_proto,
                      &response_proto[id], &rpc_status[id]);
      });
      break;
    }
    default:
      break;
  }

  AS_CLIENT_CHECK_STATUS(context_size_, response_proto);
  return AsStatus::ALLSPARK_SUCCESS;
}

std::string AsClientEngineImpl::GetVersionFull() {
  // only call rank 0 version info
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return empty";
    return std::string("");
  }
  allspark_service::Empty empty_proto;
  allspark_service::VersionInfo version_proto;
  grpc::ClientContext context;
  stub_[0]->GetVersionFull(&context, empty_proto, &version_proto);
  return version_proto.version_info();
}

std::string AsClientEngineImpl::GetOpProfilingInfo(const char* model_name) {
  // only call rank 0 profiling info
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return empty";
    return std::string("");
  }
  allspark_service::ModelName model_name_proto;
  model_name_proto.set_model_name(model_name);
  grpc::ClientContext context;
  allspark_service::OpProfilingInfo profiling_info_proto;
  stub_[0]->GetOpProfilingInfo(&context, model_name_proto,
                               &profiling_info_proto);
  return profiling_info_proto.op_profiling_info();
}

int AsClientEngineImpl::GetRankId() {
  // only return rank 0
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return 0";
    return 0;
  }
  return 0;
}

AsStatus AsClientEngineImpl::StartModel(const char* model_name) {
  return CallModelOperation(model_name, ModelOperation::Start);
}
AsStatus AsClientEngineImpl::StopModel(const char* model_name) {
  return CallModelOperation(model_name, ModelOperation::Stop);
}
AsStatus AsClientEngineImpl::ReleaseModel(const char* model_name) {
  return CallModelOperation(model_name, ModelOperation::Release);
}

AsStatus AsClientEngineImpl::StartRequest(
    const char* model_name,
    std::shared_ptr<AsEngine::RequestContent> request_info,
    RequestHandle_t* request_handle, AsEngine::ResultQueue_t* queue) {
  return CallRequestOperation(RequestOperation::Start, model_name,
                              request_handle, queue, request_info);
}

AsStatus AsClientEngineImpl::StopRequest(const char* model_name,
                                         RequestHandle_t request_handle) {
  return CallRequestOperation(RequestOperation::Stop, model_name,
                              &request_handle, nullptr, nullptr);
}

AsStatus AsClientEngineImpl::ReleaseRequest(const char* model_name,
                                            RequestHandle_t request_handle) {
  return CallRequestOperation(RequestOperation::Release, model_name,
                              &request_handle, nullptr, nullptr);
}

AsStatus AsClientEngineImpl::SyncRequest(const char* model_name,
                                         RequestHandle_t request_handle) {
  return CallRequestOperation(RequestOperation::Sync, model_name,
                              &request_handle, nullptr, nullptr);
}

AsEngineStat AsClientEngineImpl::GetAsEngineStat(const char* model_name) {
  // only call in rank 0
  if (!AsClientContext::GetInstance().CheckServiceLaunched()) {
    LOG(ERROR) << "service lauch failure, return 0";
    return AsEngineStat();
  }
  allspark_service::ModelName model_name_proto;
  model_name_proto.set_model_name(model_name);
  grpc::ClientContext context;
  AsEngineStat as_eng_stat;
  allspark_service::AsEngineStat eng_stat_proto;
  stub_[0]->GetAsEngineStat(&context, model_name_proto, &eng_stat_proto);
  makeAsEngineStatAsFromProto(as_eng_stat, eng_stat_proto);
  return as_eng_stat;
}

void AsClientContext::RegisterService(
    const std::vector<std::string>& server_path) {
  // create stub and context
  stub_.clear();
  for (const auto& address : server_path) {
    LOG(INFO) << "address: " << address;
    std::shared_ptr<Channel> channel =
        grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    std::unique_ptr<allspark_service::AllSpark::Stub> stub =
        allspark_service::AllSpark::NewStub(channel);
    stub_.push_back(std::move(stub));
    auto context = std::make_unique<ClientContext>();
  }
  // set context size
  context_size_ = server_path.size();
}

AsStatus AsClientContext::LaunchService() {
  std::vector<std::string> cmd;
  // get env
  char* env_numa_str = getenv("AS_NUMA_NUM");
  char* env_daemon_str = getenv("AS_DAEMON_PATH");
  if (env_numa_str == nullptr || env_daemon_str == nullptr) {
    LOG(ERROR) << "Failed to LaunchService, need to set AS_NUMA_NUM and "
                  "AS_DAEMON_PATH env"
               << std::endl;
    return AsStatus::ALLSPARK_UNKNOWN_ERROR;
  }
  int numa_offset = 0;
  char* env_numa_offset = getenv("AS_NUMA_OFFSET");
  if (env_numa_offset != nullptr) {
    numa_offset = atoi(env_numa_offset);
  }
  // check if service is alive
  allspark_service::makeLauchServiceCmd(cmd, 1, env_daemon_str, client_pid_,
                                        numa_offset);
  std::vector<std::string> server_path_old;
  for (int i = 0; i < 1; i++) {
    std::stringstream ss;
    ss << base_addr_ << client_pid_ << "_rank_" << i;
    server_path_old.push_back(ss.str());
  }
  RegisterService(server_path_old);
  // sent request
  int old_service_nums = CheckServiceAlive();
  LOG(INFO) << "old_service_nums: " << old_service_nums;
  if (old_service_nums > 0) {
    // ShutdownService
    // register client and shutdown alive service
    allspark_service::makeLauchServiceCmd(cmd, old_service_nums, env_daemon_str,
                                          client_pid_, numa_offset);
    std::vector<std::string> server_path_old;
    for (int i = 0; i < old_service_nums; i++) {
      std::stringstream ss;
      ss << base_addr_ << client_pid_ << "_rank_" << i;
      server_path_old.push_back(ss.str());
    }
    RegisterService(server_path_old);
    ShutdownService();
    // FIX ME... To make sure service has been terminated. But How???
    usleep(200000);
  }

  // launch new service
  int numa_nums = atoi(env_numa_str);
  allspark_service::makeLauchServiceCmd(cmd, numa_nums, env_daemon_str,
                                        client_pid_, numa_offset);
  std::vector<char*> args;
  for (int i = 0; i < cmd.size(); i++) {
    args.push_back(const_cast<char*>(cmd[i].c_str()));
  }
  args.push_back(nullptr);
  pid_t pid = fork();  // 创建新进程
  LOG(INFO) << "pid: " << pid << " numa_nums: " << env_numa_str
            << " size cmd: " << cmd.size() << " size args: " << args.size();
  if (pid == 0) {
    // 设置环境变量
    putenv("OMPI_ALLOW_RUN_AS_ROOT=1");
    putenv("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1");
    // launch mpi daemon process
    LOG(INFO) << "launch service cmd: " << cmd[0];
    int ret = execvp(cmd[0].c_str(), args.data());
    if (ret == -1) {
      LOG(ERROR) << "Failed to execute mpirun command" << std::endl;
      exit(-1);  // 如果启动失败，退出子进程并返回错误码
    }
  } else if (pid == -1) {
    // 创建新进程失败
    LOG(ERROR) << "Failed to LaunchService" << std::endl;
    return AsStatus::ALLSPARK_UNKNOWN_ERROR;
  }
  // register grpc service
  std::vector<std::string> server_path;
  for (int i = 0; i < numa_nums; i++) {
    std::stringstream ss;
    ss << base_addr_ << client_pid_ << "_rank_" << i;
    server_path.push_back(ss.str());
    // server_path.push_back(base_addr_);
    LOG(INFO) << "service path:" << ss.str();
    // LOG(INFO) << "service path:" <<base_addr_ ;
  }
  // to make sure Server has been launched
  usleep(8000000);
  RegisterService(server_path);
  LOG(INFO) << "RegisterService";
  return AsStatus::ALLSPARK_SUCCESS;
}
void AsClientContext::ShutdownService() {
  // only call rank 0 profiling info
  allspark_service::AsStatus as_status[context_size_];
  grpc::Status rpc_status[context_size_];
  grpc::ClientContext context[context_size_];
  allspark_service::Empty empty_proto;
  allspark_service::parallel_loop(0, context_size_, [&](int id) {
    CALL_RPC_FUNC(stub_[id].get(), ShutdownService, &context[id], empty_proto,
                  &as_status[id], &rpc_status[id]);
  });
}

int AsClientContext::CheckServiceAlive() {
  // return
  allspark_service::Empty empty_proto;
  allspark_service::RankId rank_proto;
  grpc::ClientContext context;
  auto ret = stub_[0]->GetRankNums(&context, empty_proto, &rank_proto);
  if (!ret.ok()) {
    LOG(WARNING) << "RPC CheckService not alive" << std::endl;
    return -1;
  }
  return rank_proto.rank_id();
}
AsClientContext::AsClientContext() : context_size_(0) {
  allspark_service::as_rpc_init_log("as_client");
  client_pid_ = getpid();
  lauch_success_ = LaunchService() == AsStatus::ALLSPARK_SUCCESS;
}
AsClientContext::~AsClientContext() { ShutdownService(); }
}  // namespace allspark