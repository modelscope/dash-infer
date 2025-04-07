/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    memory_reuser.cpp
 */

#include "memory_reuser.h"

// #include <check.h>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace allspark {

using std::map;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;
// some tensor can't be reused
string noreused[] = {"cross_attention.key_value.out",
                     "input_ids",
                     "attention_mask",
                     "fusion_input_ids",
                     "fusion_attention_mask",
                     "fusion_relative_position",
                     "batch_offset",
                     "transmask.out",
                     "max_dec_ids",
                     "dec_ids",
                     "next_beam_id",
                     "hyps_ids",
                     "hyps_beam_score"};
bool checknoused(string name) {
  for (string s : noreused) {
    if (name.find(s) != -1) {
      return true;
    }
  }
  return false;
}
void MemoryReuser::binding_with_algo_0(
    std::vector<std::vector<AsTensor*>>& visit_list, DeviceContext* ctx) {
  vector<AsTensor*> root_list;
  for (size_t i = 0; i < visit_list.size(); i++) {
    for (AsTensor* t : visit_list[i]) {
      root_list.emplace_back(t);
    }
  }

  unordered_map<AsTensor*, int> vis;
  for (AsTensor* root : root_list) {
    vis[root]++;
  }

  unordered_set<AsTensor*> binded;
  for (size_t i = 0; i < visit_list.size(); i++) {
    for (AsTensor* t : visit_list[i]) {
      if (checknoused(t->GetName())) continue;
      const auto it = binded.find(t);
      if (it == binded.end()) {
        Block::Ptr block = ctx->AllocBlock(0);
        t->BindingBlock(block);
        binded.insert(t);
      }
      vis[t]--;
    }
    for (auto& it : vis) {
      if (it.second == 0) {
        AsTensor* t = it.first;
        Block::Ptr block = t->GetBlock();
        ctx->FreeBlock(block);
        it.second = -1;
      }
    }
  }
}

// size_t MemoryReuser::caculate_size_algo_1(const std::string& graph_name,
// Workspace& ws,
//                                           std::map<DeviceType,
//                                           interval_list_t>& intervals_map,
//                                           assign_strategy_t& strategy) {
//     size_t alloc_total = 0;

//     for (auto& kv : intervals_map) {
//         DeviceType device_type = kv.first;
//         interval_list_t& intervals = kv.second;

//         vector<int64_t> v_maxsize;
//         vector<assignment_t>& assignments = strategy[device_type];

//         std::sort(intervals.begin(), intervals.end());
//         for (Interval& interval : intervals) {
//             int64_t bytes_needed = interval.max_bytes;

//             int assigned_id = -1;
//             for (size_t id = 0; id < assignments.size(); id++) {
//                 assignment_t& assigment = assignments[id];
//                 if (interval.l > assigment.back().r &&
//                     (assigned_id == -1 ||
//                      (bytes_needed <= v_maxsize[id] && bytes_needed >
//                      v_maxsize[assigned_id]) || (abs(bytes_needed -
//                      v_maxsize[id]) <
//                       abs(bytes_needed - v_maxsize[assigned_id])))) {
//                     assigned_id = id;
//                 }
//             }
//             if (assigned_id == -1) {
//                 assignments.emplace_back(assignment_t{interval});
//                 v_maxsize.emplace_back(bytes_needed);
//             } else {
//                 assignments[assigned_id].emplace_back(interval);
//                 v_maxsize[assigned_id] = std::max(v_maxsize[assigned_id],
//                 bytes_needed);
//             }
//         }
//         HIE_ENFORCE_EQ(v_maxsize.size(), assignments.size(),
//                        "length of maxsize && assignments does not match");
//         for (size_t id = 0; id < assignments.size(); id++) {
//             int64_t maxbytes = v_maxsize[id];
//             alloc_total += maxbytes;
//         }
//     }
//     return alloc_total;
// }

// void MemoryReuser::binding_with_algo_1(const std::string& graph_name,
//                                        std::vector<std::vector<AsTensor*>>&
//                                        visit_list, Workspace& ws, const
//                                        assign_strategy_t& strategy) {
//     std::unordered_map<AsTensor*, Block::Ptr> blocks_map;

//     for (const auto& kv : strategy) {
//         DeviceType device_type = kv.first;
//         const vector<assignment_t>& assignments = kv.second;
//         DeviceContext* context = ws.GetDeviceContext(device_type);

//         for (size_t id = 0; id < assignments.size(); id++) {
//             if (assignments[id].size() == 0) {
//                 HIE_ENFORCE_EQ(id, 0, "assignment container is empty");
//                 continue;
//             }
//             size_t maxbytes = 0;
//             for (const Interval& interval : assignments[id]) {
//                 maxbytes = std::max(maxbytes, interval.max_bytes);
//             }
//             Block::Ptr block = context->AllocBlock(maxbytes);
//             for (const Interval& interval : assignments[id]) {
//                 HIE_ENFORCE(blocks_map.find(interval.root_tensor) ==
//                 blocks_map.end(),
//                             "duplicates root tensor");
//                 blocks_map.emplace(interval.root_tensor, block);
//             }
//         }
//     }

//     for (size_t i = 0; i < visit_list.size(); i++) {
//         for (AsTensor* t : visit_list[i]) {
//             if (ws.GetLocalTensor(graph_name, t->GetName()) == nullptr) {
//                 continue;
//             }
//             AsTensor* root = t->GetRootTensor();
//             HIE_ENFORCE(blocks_map.find(root) != blocks_map.end(), "no root
//             tensor in blocks map"); t->BindingBlock(blocks_map[root]);
//         }
//     }
// }

}  // namespace allspark
