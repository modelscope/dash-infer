/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_util.h
 */
#pragma once

namespace allspark {
namespace cuda_util {

template <typename T>
void CheckInfNan(const T* d_data, size_t size);

}
}  // namespace allspark
