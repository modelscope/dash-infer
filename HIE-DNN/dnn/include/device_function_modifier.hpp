/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    device_function_modifier.hpp
 */

#ifndef DNN_INCLUDE_DEVICE_FUNCTION_MODIFIER_HPP_
#define DNN_INCLUDE_DEVICE_FUNCTION_MODIFIER_HPP_

#ifdef __CUDACC__
#define DEVICE_FUNCTION __device__ __forceinline__
#else
#define DEVICE_FUNCTION inline
#endif

#ifdef __CUDACC__
#define HOST_DEVICE_FUNCTION __host__ __device__ inline
#else
#define HOST_DEVICE_FUNCTION inline
#endif

#endif  // DNN_INCLUDE_DEVICE_FUNCTION_MODIFIER_HPP_

