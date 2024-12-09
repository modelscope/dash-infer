/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    framework_utils.hpp
 */

#pragma once

#include <random>
#include <thread>
#include <vector>

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_BF16
#include <hie_bfloat16.hpp>
#endif

#include <cuda_runtime.h>

#include "framework/allspark.min.hpp"
#include "test_common.h"

#define asTensor allspark::AsTensor
#define asCUDA allspark::DeviceType::CUDA

#define asINT8 allspark::DataType::INT8
#define asINT32 allspark::DataType::INT32
#define asINT64 allspark::DataType::INT64
#define asFP32 allspark::DataType::FLOAT32
#define asPTR allspark::DataType::POINTER
#ifdef ENABLE_FP16
#define asFP16 allspark::DataType::FLOAT16
#endif
#ifdef ENABLE_BF16
#define asBF16 allspark::DataType::BFLOAT16
#endif

#define DEBUG_TENSOR(MAP, ALIAS)
/*
#define DEBUG_TENSOR(MAP, ALIAS)           \
  printf("[" #MAP "]\t" #ALIAS ": \t%s\n", \
         tensor_map_##MAP.at(ALIAS##_name.c_str())->ToString().c_str());
*/

#define CREATE_TENSOR(MAP, ALIAS, TYPE, ...)                          \
  std::string name_##ALIAS = #ALIAS;                                  \
  common::AddTensor(MAP, name_##ALIAS, common::toDataType<TYPE>::dt); \
  std::vector<allspark::dim_t> shape_##ALIAS = {__VA_ARGS__};         \
  MAP.at(name_##ALIAS)->SetShape(Shape(shape_##ALIAS));

#define CREATE_WORKSPACE(MAP, BYTES, ...)         \
  std::string name_workspace = "workspace";       \
  common::AddTensor(MAP, name_workspace, asINT8); \
  MAP.at(name_workspace)->SetShape(Shape({dim_t(BYTES)}));

namespace common {

template <typename T>
struct toDataType {
  constexpr static allspark::DataType dt =
      allspark::DataType::DATATYPE_UNDEFINED;
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_t = cudaDataType::CUDA_R_32F;
#endif  // __LIBRARY_TYPES_H__
};

template <>
struct toDataType<float> {
  constexpr static allspark::DataType dt = asFP32;
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_t = cudaDataType::CUDA_R_32F;
#endif  // __LIBRARY_TYPES_H__
};

#ifdef ENABLE_FP16
template <>
struct toDataType<half> {
  constexpr static allspark::DataType dt = asFP16;
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_t = cudaDataType::CUDA_R_16F;
#endif  // __LIBRARY_TYPES_H__
};
#endif

#ifdef ENABLE_BF16
using bfloat16 = hie::bfloat16;
template <>
struct toDataType<bfloat16> {
  constexpr static allspark::DataType dt = asBF16;
#ifdef __LIBRARY_TYPES_H__
  constexpr static cudaDataType cuda_t = cudaDataType::CUDA_R_16BF;
#endif  // __LIBRARY_TYPES_H__
};
#endif

bool AsyncH2D(const void* host, asTensor* device, size_t size,
              cudaStream_t stream);
bool AsyncD2H(const asTensor* device, void* host, size_t size,
              cudaStream_t stream);
void AddTensor(allspark::TensorMap& t_map, const std::string& name,
               allspark::DataType data_type);

}  // namespace common
