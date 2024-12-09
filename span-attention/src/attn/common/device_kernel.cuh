/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    device_kernel.cuh
 */

#pragma once

namespace span {

template <typename Operator>
__global__ __launch_bounds__(Operator::kThreadCount) void Kernel(
    typename Operator::Params params) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage* shared_storage =
      reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage);
}

}  // namespace span
