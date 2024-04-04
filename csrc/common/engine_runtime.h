/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_runtime.h
 */

#pragma once

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "allspark.h"
#include "engine_control_message.h"

namespace allspark {
class ModelControlState final {
  std::unique_ptr<std::thread> loop_thread_;

 public:
  std::string model_name;
  std::queue<std::unique_ptr<EngineControlMessage>> msg_queue;

  std::unique_ptr<std::mutex> lock;
  std::unique_ptr<std::condition_variable> cond_var;

  std::unordered_map<std::string, std::shared_ptr<RequestHandle>>
      request_handle_map;
  std::unordered_map<std::string, std::shared_ptr<AsEngine::ResultQueue>>
      result_queue_map;

  bool model_stopping = false;  // after GracefulStopModel called...

  ModelControlState(const std::string& name) : model_name(name) {
    lock = std::make_unique<std::mutex>();
    cond_var = std::make_unique<std::condition_variable>();
  }

  template <typename Func, typename... Args>
  void StartLoop(Func&& func, Args&&... args) {
    loop_thread_ = std::make_unique<std::thread>(std::forward<Func>(func),
                                                 std::forward<Args>(args)...);
  }

  void StopLoop() {
    if (loop_thread_) {
      loop_thread_->join();
      loop_thread_.reset();
    }
  }
};

class RequestHandle {
 public:
  std::string request_uuid;
  size_t generate_length = 0;
  size_t context_length = 0;
  size_t continue_count = 0;
};

class ResultQueueImpl : public AsEngine::ResultQueue {
 public:
  typedef std::shared_ptr<AsEngine::GeneratedElements> GenerateElementPtr;
  ResultQueueImpl();
  ~ResultQueueImpl();
  virtual AsEngine::GenerateRequestStatus GenerateStatus();
  virtual size_t GeneratedLength();
  virtual GenerateElementPtr Get();
  virtual GenerateElementPtr GetNoWait();

  GenerateElementPtr GetNoLock();
  void SetStatus(AsEngine::GenerateRequestStatus);

  // append new data to latest queue, or create a new element.
  void AppendGenerateData(std::vector<int64_t>&& new_tokens);
  void AppendGenerateElement(
      std::shared_ptr<AsEngine::GeneratedElements> new_ele);

  // private class can add function that push elements.
 private:
  std::queue<GenerateElementPtr> store_queue_;
  std::mutex queue_mutex_;
  AsEngine::GenerateRequestStatus status_ =
      AsEngine::GenerateRequestStatus::Init;
  size_t generate_length_;

  std::condition_variable cond_var_;

  bool closed_ = false;

  typedef std::unique_lock<std::mutex> queue_read_lock_t;
  typedef std::unique_lock<std::mutex> queue_write_lock_t;
};
};  // namespace allspark
