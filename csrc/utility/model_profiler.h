/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_profiler.h
 */

#pragma once

#include <assert.h>
#include <common/device_context.h>
#include <float.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#include <cuda/gpu_profiler.h>
#include <cuda_runtime.h>

#include "check_cuda.h"
#endif

namespace allspark {

class AsModel;
enum class ProfileType {
  OP,     // model 's op '
  SCOPE,  // scope defined by model logic part.
};

class TracerLog {
 public:
  TracerLog(DeviceType device, const char* name, int the_color_id) {
#ifdef ENABLE_CUDA
    if (device == DeviceType::CUDA) {
      cuda_tracer_ = std::make_unique<Tracer>(name, the_color_id);
    }
#endif
    if (device == DeviceType::CPU) {
      // cpu TBD
    }
  }

 private:
#ifdef ENABLE_CUDA
  std::unique_ptr<Tracer> cuda_tracer_;
#endif
};

class ProfileEvent {
 public:
  ProfileEvent(std::string& name, ProfileType type, int card_id, float time_ms)
      : name_(name), type_(type), card_id_(card_id), time_ms_(time_ms){};

  ProfileEvent(ProfileEvent& rhs) = default;

  std::string name_;
  ProfileType type_;
  int card_id_;  // current card id, not support cross node.
  float time_ms_;
};

class ProfileEventStatistic {
 public:
  ProfileEventStatistic() = default;

  ProfileEventStatistic(std::string& name, ProfileType type, int card_id)
      : name_(name), type_(type), card_id_(card_id) {}

  ProfileEventStatistic(const ProfileEventStatistic& rhs) = default;

  void UpdateEvent(const ProfileEvent& event) {
    min_ms_ = std::min(min_ms_, event.time_ms_);
    max_ms_ = std::max(max_ms_, event.time_ms_);
    count_++;
    sum_ms_ += event.time_ms_;
  }

  void Reset() {
    min_ms_ = FLT_MAX;
    max_ms_ = 0;
    count_ = 0;
    sum_ms_ = 0;
  }

  std::string name_;
  ProfileType type_;
  int card_id_;
  float min_ms_ = FLT_MAX;  // min ms
  float max_ms_ = 0;        // max ms
  long count_ = 0;          // total count
  double sum_ms_ = 0;       // total time
};

class ModelProfiler {
 public:
  // name, [min,max,avg,total_cnt,sum,percentage]
  using stat_type = std::pair<std::string, std::array<double, 6>>;
  using profile_events = std::unordered_map<std::string, ProfileEventStatistic>;
  using event_map = std::unordered_map<std::string, profile_events>;
#ifdef ENABLE_CUDA
  using queue_item = std::tuple<std::string, std::shared_ptr<cudaEvent_t>,
                                std::shared_ptr<cudaEvent_t>>;
#endif

  ModelProfiler(AsModel* model);

  // TODO: for GPU, needs add stream to async collect data.
  void CollectBy(event_map& events, const std::string& tag, std::string& name,
                 float time_ms, ProfileType type);

  void CollectByOP(const std::string& tag, std::string& name, float time_ms);
  // collect current result, and calc the min/max/avg, and sort by large to
  // small and return.
  void CollectByScope(const std::string& tag, std::string& name, float time_ms);
  // report min max avg stat in profile op.
  std::vector<stat_type> ReportOpStat(const std::string& tag);

  // report min max avg stat in profile op.
  std::vector<stat_type> ReportScopeStat(const std::string& tag);

#ifdef ENABLE_CUDA
  // push queue
  void PushQueue(const std::string& tag, const std::string& name,
                 std::shared_ptr<cudaEvent_t> start,
                 std::shared_ptr<cudaEvent_t> stop) {
    auto it = cuda_event_queues_.find(tag);
    if (it == cuda_event_queues_.end()) {
      auto ret = cuda_event_queues_.emplace(tag, std::queue<queue_item>());
      it = ret.first;
    }
    auto& cuda_event_queue_ = it->second;
    // push only in a specific thread, not need to add mutex
    cuda_event_queue_.push(std::make_tuple(name, start, stop));
  }

  // try update time
  void TryCollectOpTime(const std::string& tag) {
    auto it = cuda_event_queues_.find(tag);
    if (it == cuda_event_queues_.end()) {
      return;
    }
    auto& cuda_event_queue = it->second;
    while (!cuda_event_queue.empty()) {
      auto item = cuda_event_queue.front();
      auto query_ret = cudaEventQuery(*std::get<2>(item));
      if (query_ret != cudaSuccess) {
        break;
      } else {
        cuda_event_queue.pop();
        float ms = 0;
        AS_CHECK_CUDA(
            cudaEventElapsedTime(&ms, *std::get<1>(item), *std::get<2>(item)));
        AS_CHECK_CUDA(cudaEventDestroy(*std::get<1>(item)));
        AS_CHECK_CUDA(cudaEventDestroy(*std::get<2>(item)));
        CollectByOP(tag, std::get<0>(item), ms);
      }
    }
  }
#endif

  void Reset();

 private:
  // with a default size to make sure memory footprint.
  size_t size_limit = 10000;
  event_map op_times_map_;
  event_map scope_times_map_;
  int card_id_;
#ifdef ENABLE_CUDA
  std::unordered_map<std::string, std::queue<queue_item>> cuda_event_queues_;
#endif
};

class ProfilerAdder {
 public:
  ProfilerAdder(ModelProfiler& profiler, std::string tag, std::string name,
                const DeviceContext* ctx)
      : profiler_(profiler),
        tag_(std::move(tag)),
        name_(std::move(name)),
        ctx_(ctx),
        start_(std::chrono::steady_clock::now()) {
#ifdef ENABLE_CUDA
    if (ctx->GetDeviceType() == DeviceType::CUDA) {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      start_event_ = std::make_shared<cudaEvent_t>();
      stop_event_ = std::make_shared<cudaEvent_t>();
      AS_CHECK_CUDA(cudaEventCreate(start_event_.get()));
      AS_CHECK_CUDA(cudaEventCreate(stop_event_.get()));
      AS_CHECK_CUDA(cudaEventRecord(*start_event_, gpu_ctx->GetStream()));
      // trace only rank 0
      if (gpu_ctx->GetRank() == 0) {
        trace_id_++;
        std::stringstream ss;
        ss << tag_ << ":" << name_;
        std::string trace_name = ss.str();
        trace_t_ = std::make_unique<TracerLog>(ctx->GetDeviceType(),
                                               trace_name.c_str(), trace_id_);
      }
    }
#endif
  }
  ~ProfilerAdder() {
#ifdef ENABLE_CUDA
    if (ctx_->GetDeviceType() == DeviceType::CUDA) {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaEventRecord(*stop_event_, gpu_ctx->GetStream());
      profiler_.PushQueue(tag_, name_, start_event_, stop_event_);
      profiler_.TryCollectOpTime(tag_);
    }
#endif
    if (ctx_->GetDeviceType() == DeviceType::CPU) {
      auto end = std::chrono::steady_clock::now();
      // record profiler information.
      auto ms =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
              .count() /
          1000.f;
      profiler_.CollectByOP(tag_, name_, ms);
    }
  }

  ModelProfiler& profiler_;
  std::string tag_;
  std::string name_;
  const std::chrono::time_point<std::chrono::steady_clock> start_;
  const DeviceContext* ctx_;
#ifdef ENABLE_CUDA
  std::shared_ptr<cudaEvent_t> start_event_;
  std::shared_ptr<cudaEvent_t> stop_event_;
  static int trace_id_;
  std::unique_ptr<TracerLog> trace_t_;
#endif
};

}  // namespace allspark
