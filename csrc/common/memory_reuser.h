/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    memory_reuser.h
 */
#include <core/tensor/tensor.h>

#include "device_context.h"

namespace allspark {

struct Interval {
  Interval(AsTensor* _root_tensor, int _l = 0, int _r = 0, int64_t _bytes = 0)
      : root_tensor(_root_tensor), l(_l), r(_r), max_bytes(_bytes) {}

  bool operator<(const Interval& other) const {
    if (r == other.r) return l < other.l;
    return r < other.r;
  }

  bool operator==(const Interval& other) const {
    return root_tensor == other.root_tensor;
  }

  AsTensor* root_tensor;
  int l, r;
  int64_t max_bytes;
};

using assignment_t = std::vector<Interval>;
using interval_list_t = assignment_t;
using assign_strategy_t = std::map<DeviceType, std::vector<assignment_t>>;

class MemoryReuser {
 public:
  void binding_with_algo_0(std::vector<std::vector<AsTensor*>>& visit_list,
                           DeviceContext* ctx);
};

}  // namespace allspark
