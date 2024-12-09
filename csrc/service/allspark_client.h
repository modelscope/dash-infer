/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_client.h
 */

#pragma once
#include <allspark.h>

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace allspark {

class AsClientEngineImpl;
class AsClientEngine final {
 public:
  AsClientEngine();
  ~AsClientEngine();

  /**
   * build a model from config struct
   * param is same with yaml config file
   *
   * @return status code.
   */
  AsStatus BuildModelFromConfigStruct(AsModelConfig& model_config);

  AsStatus ReloadModelToDeviceMemory(const char* model_name);
  AsStatus UnloadModelFromDeviceMemory(const char* model_name);

  /**
   * Get model type, input tensor , output tensor info
   *
   * @param model_name  model name key.
   * @param model_info  return model key.
   *
   * @return status code
   */
  AsStatus GetModelInformation(const char* model_name, std::string* model_info);

  /**
   * Get a model's info by passing model file paths
   * this api can fetch convert version information from user
   * */
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
  AsStatus LoadLoraByName(const char* model_name, const char* lora_name);
  AsStatus UnloadLoraByName(const char* model_name, const char* lora_name);
  AsEngineStat GetAsEngineStat(const char* model_name);

  /**
   * Get SDK version string
   * version string will include version, git sha1, and build time,
   * eg: 0.1.4/(GitSha1:beaca93)/(Build:20230403154806)
   */
  std::string GetVersionFull();

  /**
   * Get op profiling info
   * profiling string includes op name, min_time, max_time, count, sum,
   * percentage
   */
  std::string GetOpProfilingInfo(const char* model_name);

  /**
   * Get rank id (0~rank_num-1)
   * Since openmpi is used to manage CPU inferer task, which may
   * launch multiply process to do the inferer, GetRankId is used
   * to indicate the manager process and get the output in manager
   * process.
   * @note 0 is the manager process, we get output only if GetRandId
   * return 0;
   * GetRankId always return 0 in GPU inferer.
   */
  int GetRankId();

  /**
   * check if allspark work as servie.
   * Normally in CPU mode, it uses mpi to make AllSpark run a larger model fast,
   * in this case, AllSpark serves as MPI daemon service.
   */
  bool IsAllSparkWorkAsService();

 private:
  std::unique_ptr<AsClientEngineImpl> as_client_engine_impl_;
};

}  // namespace allspark
