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

#include "test_common.h"

namespace common {

// ---------------------------------
// Random
// ---------------------------------
template <typename T>
std::vector<T> rand_normal_float(size_t count, float mean = 0.f,
                                 float scale = 1.f, float fallback = 0.f,
                                 int nworkers = 32) {
  std::vector<T> rtn(count, 0.f);
  auto rng_func = [&rtn, mean, scale, fallback](size_t start, size_t end,
                                                int seed) {
    std::normal_distribution<float> dis(mean, scale);
    std::default_random_engine generator;
    generator.seed(seed);

    std::generate(rtn.begin() + start, rtn.begin() + end,
                  [&generator, &dis, fallback]() {
                    auto val = static_cast<T>(dis(generator));
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
template <typename T>
std::vector<T> nan_float(size_t count, int nworkers = 32) {
  std::vector<T> rtn(count);
  auto gen_func = [&rtn](size_t start, size_t end, int /* seed */) {
    std::generate(rtn.begin() + start, rtn.begin() + end, []() {
      auto val = static_cast<T>(nanf("0xDEAD"));
      float fval = static_cast<float>(val);
      if (!std::isnan(fval)) {
        throw std::runtime_error("generate nanf failed");
      }
      return val;
    });
    return;
  };

  std::vector<std::unique_ptr<std::thread>> thds(nworkers);
  size_t chunk_size = count / nworkers;
  for (int i = 0; i < nworkers - 1; ++i) {
    thds[i] = std::make_unique<std::thread>(gen_func, i * chunk_size,
                                            (i + 1) * chunk_size, i);
  }
  thds[nworkers - 1] = std::make_unique<std::thread>(
      gen_func, (nworkers - 1) * chunk_size, count, nworkers - 1);

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
  max_index = static_cast<size_t>(-1);
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
