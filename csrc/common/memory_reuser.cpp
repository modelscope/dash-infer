/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    memory_reuser.cpp
 */
#include "memory_reuser.h"

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
                     "generated_ids",
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
}  // namespace allspark
