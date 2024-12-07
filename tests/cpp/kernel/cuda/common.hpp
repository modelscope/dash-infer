/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    common.hpp
 */

#ifndef __KERNEL_CUDA_COMMON_HPP__
#define __KERNEL_CUDA_COMMON_HPP__

#include <random>
#include <thread>
#include <vector>

#ifdef ENABLE_FP16
#include <cuda_fp16.h>
#endif
#ifdef ENABLE_BF16
#include <hie_bfloat16.hpp>
#endif

#include <core/kernel/cuda/cuda_kernel.h>
#include <core/tensor/tensor.h>
#include <cuda/cuda_context.h>
#include <cuda_runtime.h>
#include <library_types.h>  // cuda type
#include <test_common.h>

#define asTensor allspark::AsTensor
#define asCUDA allspark::DeviceType::CUDA
#define asDense allspark::DataMode::DENSE

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
// #define DEBUG_TENSOR(MAP, ALIAS) \
// printf("[" #MAP "]\t" #ALIAS ": \t%s\n", tensor_map_##MAP.at(ALIAS##_name.c_str())->ToString().c_str());

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

// ---------------------------------
// Random
// ---------------------------------
template <typename T>
std::vector<T> rand_normal_float(size_t count, float scale = 1.f,
                                 float fallback = 0.f, int nworkers = 32) {
  std::vector<T> rtn(count, 0.f);
  auto rng_func = [&rtn, scale, fallback](size_t start, size_t end, int seed) {
    std::normal_distribution<float> dis(0.f, 1.f);
    std::default_random_engine generator;
    generator.seed(seed);

    std::generate(rtn.begin() + start, rtn.begin() + end,
                  [&generator, &dis, scale, fallback]() {
                    auto val = static_cast<T>(scale * dis(generator));
                    float fval = static_cast<float>(val);
                    if (std::isnan(fval) || std::isinf(fval)) {
                      val = fallback;
                    }
                    return val;
                  });
    return;
  };

  std::vector<std::unique_ptr<std::thread>> thds(nworkers);
  size_t chunk_size = count / nworkers;
  for (int i = 0; i < nworkers - 1; ++i) {
    thds[i] = std::make_unique<std::thread>(rng_func, i * chunk_size,
                                            (i + 1) * chunk_size, i);
  }
  thds[nworkers - 1] = std::make_unique<std::thread>(
      rng_func, (nworkers - 1) * chunk_size, count, nworkers - 1);

  for (auto& th : thds) {
    th->join();
  }

  return rtn;
}

// ---------------------------------
// Diff
// ---------------------------------
template <typename T>
void DiffWithMaxIndex(const std::vector<T>& ref, const std::vector<T>& val,
                      float eps, float& max_diff, size_t& max_index,
                      int& num_exceed, int& num_nan) {
  constexpr int nan_print = 4;
  float sum = 0.f;
  for (auto tr : ref) sum += static_cast<float>(tr);
  float ref_avg = fabs(sum) / static_cast<float>(ref.size());
  eps = ref_avg > eps ? ref_avg : eps;

  std::vector<size_t> nan_list;
  std::vector<size_t> exc_list;

  max_diff = 0.f;
  max_index = -1;
  // val.size() may lower than ref.size()
  for (size_t i = 0; i < val.size(); i++) {
    if (std::isnan(static_cast<float>(val[i]))) {
      nan_list.push_back(i);
      continue;
    }
    float diff = fabs(static_cast<float>(val[i]) - static_cast<float>(ref[i]));
    if (max_diff < diff) {
      max_diff = diff;
      max_index = i;
    }
    if (diff > eps) {
      exc_list.push_back(i);
    }
  }

  num_nan = nan_list.size();
  num_exceed = exc_list.size();

  if (num_exceed > 0) {
    printf(
        "[ERR] num_exceed=%d, max index = %d, diff / eps = %2.5f / %2.5f, ref "
        "/ val = %2.5f / %2.5f\n",
        num_exceed, int(max_index), max_diff, eps, float(ref[max_index]),
        float(val[max_index]));
  }
  if (num_nan > 0) {
    printf("[ERR] num_nan=%d\n", num_nan);
    for (int i = 0; i < nan_list.size() && i < nan_print; i++) {
      printf("[ERR] nan_list[%d] ref / val = %2.5f / %2.5f\n", nan_list[i],
             ref[nan_list[i]], val[nan_list[i]]);
    }
  }
  return;
}

}  // namespace common

#endif  // __KERNEL_CUDA_COMMON_HPP__
