/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    func_modifier.h
 */

#pragma once

#ifdef __CUDACC__
#define DEVICE_FUNC __device__ __forceinline__
#else
#define DEVICE_FUNC inline
#endif

#ifdef __CUDACC__
#define HOST_DEVICE_FUNC __host__ __device__ inline
#else
#define HOST_DEVICE_FUNC inline
#endif
