/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_service_parallel.h
 */

#pragma once
#include <functional>
namespace allspark {
namespace allspark_service {
using ParallelForBody = std::function<void(int idx)>;

void parallel_loop(const int begin, const int end, const ParallelForBody& body);

}  // namespace allspark_service
}  // namespace allspark