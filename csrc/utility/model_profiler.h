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

namespace allspark {
class AsModel;
enum class ProfileType {
  OP,     // model 's op '
  SCOPE,  // scope defined by model logic part.
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
  using stat_type = std::pair<std::string, std::array<double, 6> >;
  using profile_events = std::unordered_map<std::string, ProfileEventStatistic>;
  using event_map = std::unordered_map<std::string, profile_events>;

  ModelProfiler(AsModel* model);

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

  void Reset();

 private:
  // with a default size to make sure memory footprint.
  size_t size_limit = 10000;
  event_map op_times_map_;
  event_map scope_times_map_;
  int card_id_;
};

class ProfilerAdder {
 public:
  ProfilerAdder(ModelProfiler& profiler, std::string tag, std::string name,
                const DeviceContext* ctx)
      : profiler_(profiler),
        tag_(std::move(tag)),
        name_(std::move(name)),
        ctx_(ctx),
        start_(std::chrono::steady_clock::now()) {}

  ~ProfilerAdder() {
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
};

}  // namespace allspark
