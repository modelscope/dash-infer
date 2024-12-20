/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_lowp_common.h
 */

#pragma once

namespace allspark {

inline uint64_t aligned_size(uint64_t n, uint64_t aligned = 16) {
  return (n + aligned - 1) / aligned * aligned;
}

inline int64_t perc_qgemm_a8w8_gpu_workspace_size(const int M, const int N,
                                                  const int K) {
  return (aligned_size(M * K) * sizeof(int8_t)  // For A qdata
          + aligned_size(M) * sizeof(float)     // For A scale
          + aligned_size(M) * sizeof(float)     // For A red_max
          + aligned_size(M) * sizeof(uint32_t)  // For A red_count
          + aligned_size(M) * sizeof(int32_t)   // For A red_sum
          + aligned_size(M * N) *
                sizeof(int32_t));  // For int32 immediate result of I8GEMM
}

struct SplitKParams {
  bool EnableSplitK;
  int SplitK;
};

enum TileSchedule : uint8_t { M_BLK_CONTINUOUS = 0, N_BLK_CONTINUOUS = 1 };

template <typename FType, typename QType>
struct SM70_GEMM_A16W8_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  int M;
  int N;
  int K;
  int GroupCnt;
  int GroupSize;
  FType* C_split_ptr;
  int SplitK;
  TileSchedule schedule_mn;
};

template <typename FType, typename QType>
struct SM70_GEMM_A16W4_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  int M;
  int N;
  int K;
  FType* C_split_ptr;
  int SplitK;
  int GroupCnt;              // SubChannel
  int GroupSize;             // SubChannel
  TileSchedule schedule_mn;  // reserved for standard GEMM
};

template <typename FType, typename QType>
struct SM8x_GEMM_A16W8_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  int M;
  int N;
  int K;
  int GroupCnt;
  int GroupSize;
  FType* C_split_ptr;
  int SplitK;
  TileSchedule schedule_mn;
};


template <typename FType, typename QType>
struct GEMM_A16W8_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  FType* C_split_ptr;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t GroupSize;
  uint32_t SplitK;
};

}  // namespace allspark
