/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_profiler.cpp
 */

#include "model_profiler.h"

#include "../core/model/model.h"

namespace allspark {
ModelProfiler::ModelProfiler(AsModel* model) {
  assert(model != nullptr);
  card_id_ = model->GetRankId();
}

void ModelProfiler::CollectBy(event_map& events, const std::string& tag,
                              std::string& name, float time_ms,
                              ProfileType type) {
  auto eit = events.find(tag);
  if (eit == events.end()) {
    auto ret = events.emplace(tag, profile_events());
    eit = ret.first;
  }

  auto& event = eit->second;
  auto it = event.find(name);
  if (it != event.end()) {
    it->second.UpdateEvent({name, type, card_id_, time_ms});
  } else {
    ProfileEventStatistic stat(name, type, card_id_);
    stat.UpdateEvent({name, type, card_id_, time_ms});
    event.emplace(name, std::move(stat));
  }
}

void ModelProfiler::CollectByOP(const std::string& tag, std::string& name,
                                float time_ms) {
  CollectBy(op_times_map_, tag, name, time_ms, ProfileType::OP);
}

// collect current result, and calc the min/max/avg, and sort by large to small
// and return.
void ModelProfiler::CollectByScope(const std::string& tag, std::string& name,
                                   float time_ms) {
  CollectBy(scope_times_map_, tag, name, time_ms, ProfileType::SCOPE);
}

// report min max avg stat in profile op.
std::vector<ModelProfiler::stat_type> ModelProfiler::ReportOpStat(
    const std::string& tag) {
  auto it = op_times_map_.find(tag);
  if (it == op_times_map_.end()) {
    return std::vector<ModelProfiler::stat_type>(0);
  }

  // loop the op and sort the output by avg.
  auto& op_times = it->second;
  std::vector<ModelProfiler::stat_type> op_stat_unsort;
  op_stat_unsort.reserve(op_times.size());
  double sum_ms = 0;
  for (auto const& r : op_times) {
    sum_ms += r.second.sum_ms_;
  }
  for (auto const& r : op_times) {
    auto& name = r.first;
    ModelProfiler::stat_type e;
    e.first = name;
    e.second[0] = r.second.min_ms_;
    e.second[1] = r.second.max_ms_;
    e.second[2] = r.second.sum_ms_ / r.second.count_;
    e.second[3] = r.second.count_;
    e.second[4] = r.second.sum_ms_;
    e.second[5] = r.second.sum_ms_ / sum_ms * 100;
    op_stat_unsort.push_back(std::move(e));
  }
  // sort by percentage.
  std::sort(op_stat_unsort.begin(), op_stat_unsort.end(),
            [](ModelProfiler::stat_type& l, ModelProfiler::stat_type& r) {
              return l.second[5] > r.second[5];
            });
  // clear op_times
  op_times.clear();
  return op_stat_unsort;
}

// report min max avg stat in profile op.
std::vector<ModelProfiler::stat_type> ModelProfiler::ReportScopeStat(
    const std::string& tag) {
  // TBD
  return std::vector<ModelProfiler::stat_type>(0);
}

void ModelProfiler::Reset() {
  for (auto& pair : op_times_map_) {
    auto& events = pair.second;
    for (auto& pair_sub : events) {
      pair_sub.second.Reset();
    }
  }

  for (auto& pair : scope_times_map_) {
    auto& events = pair.second;
    for (auto& pair_sub : events) {
      pair_sub.second.Reset();
    }
  }
}
}  // namespace allspark
