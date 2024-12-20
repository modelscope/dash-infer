/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_perc_kernel.cu
 */

#include <cuda/std/type_traits>

#include "gemm_a16w8_kernel.h"
#include "gemm_lowp_utils.cuh"

namespace allspark {
namespace cuda {

template <uint32_t BLOCK, uint32_t UNROLL, typename FT>
__global__ void get_input_padded_k_align_kernel(const FT* in, FT* in_padded,
                                                const uint32_t m,
                                                const uint32_t k,
                                                const uint32_t k_padded,
                                                U32DivMod kDivMod) {
  int base_offset = blockIdx.x * BLOCK * UNROLL + threadIdx.x * UNROLL;
  FT regs[UNROLL]{};
  if (base_offset < m * k_padded) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      auto dm = kDivMod.DivMod(base_offset + i);
      uint32_t m_idx = dm.div;
      uint32_t k_idx = dm.mod;
      if (k_idx < k) {
        regs[i] = in[m_idx * k + k_idx];
      }
    }
    store_gdata<FT, UNROLL>(regs, in_padded + base_offset);
  }
}
/*
 * padding tensor in from (m, k) to (m, k_padded)
 * here k_padded is at least multiples of 8,
 * FT supports FLOAT16 and BFLOAT16
 */
template <typename FT>
void get_input_padded_k_align(const FT* in, FT* in_padded, const uint32_t m,
                              const uint32_t k, const uint32_t k_padded,
                              cudaStream_t stream) {
  const uint32_t BLOCK = 128;
  const uint32_t UNROLL = 8;
  const uint32_t grid = DivCeil(m * k_padded, BLOCK * UNROLL);
  U32DivMod kDivMod(k_padded);
  get_input_padded_k_align_kernel<BLOCK, UNROLL, FT>
      <<<grid, BLOCK, 0, stream>>>(in, in_padded, m, k, k_padded, kDivMod);
}

template <uint32_t BLOCK, uint32_t UNROLL, typename FT>
__global__ void remove_padded_n_align_kernel(const FT* out_padded, FT* out,
                                             const uint32_t m, const uint32_t n,
                                             const uint32_t padded_n,
                                             U32DivMod nDivMod,
                                             PackedEltwiseConfig packConfig) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

  if (idx < packConfig.nPack) {
    FT regs[UNROLL]{};
    int base_offset = idx * UNROLL;

#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      auto dm = nDivMod.DivMod(base_offset + i);
      uint32_t m_idx = dm.div;
      uint32_t n_idx = dm.mod;
      regs[i] = out_padded[m_idx * padded_n + n_idx];
    }

    store_gdata<FT, UNROLL>(regs, out + base_offset);
  } else if (idx < packConfig.nThread) {
    idx = idx - packConfig.nPack + packConfig.nPack * UNROLL;
    auto dm = nDivMod.DivMod(idx);
    uint32_t m_idx = dm.div;
    uint32_t n_idx = dm.mod;
    out[idx] = out_padded[m_idx * padded_n + n_idx];
  }
}

/*
 * remove padding of tensor out from (m, n_padded) to (m, n)
 * FT supports FLOAT16 and BFLOAT16
 */
template <typename FT>
void remove_padded_n_align(const FT* out_padded, FT* out, const uint32_t m,
                           const uint32_t n, const uint32_t padded_n,
                           cudaStream_t stream) {
  const uint32_t BLOCK = 128;
  int packSize = GetPackSize(out);

  PackedEltwiseConfig packConfig(m * n, packSize, BLOCK);
  U32DivMod nDivMod(n);
  switch (packSize) {
    case 8:
      remove_padded_n_align_kernel<BLOCK, 8, FT>
          <<<packConfig.nBlock, BLOCK, 0, stream>>>(
              out_padded, out, m, n, padded_n, nDivMod, packConfig);
      break;
    default:
      LOG(ERROR) << "Now only support out ptr is 16-byte aligned";
      break;
  }
}

template <typename FT, typename QT>
__global__ void dequantize_rhs_a16w8_kernel(const QT* qdata, const FT* scales,
                                            const FT* zeros, FT* fdata,
                                            const uint32_t N,
                                            const uint32_t K) {
  const int k_idx = blockIdx.y;
  const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx < N && k_idx < K) {
    const FT scale = scales[n_idx];
    const FT zero = zeros[n_idx];
    FT val = (FT(qdata[k_idx * N + n_idx]) - zero) * scale;
    fdata[k_idx * N + n_idx] = val;
  }
}

// Now only support UNROLL = 8
template <int BLOCK, int UNROLL, typename FT, typename QT>
__global__ void dequantize_rhs_a16w8_opt_kernel(
    const QT* qdata, const FT* scales, const FT* zeros, FT* fdata,
    const U32DivMod scaleDivMod, PackedEltwiseConfig packConfig) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

  if (idx < packConfig.nPack) {
    QT qval_reg[UNROLL];
    FT fval_reg[UNROLL];

    *reinterpret_cast<int2*>(qval_reg) =
        *(reinterpret_cast<const int2*>(qdata) + idx);

#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      int n_idx = scaleDivMod.Mod(idx * UNROLL + i);
      fval_reg[i] =
          (static_cast<FT>(qval_reg[i]) - zeros[n_idx]) * scales[n_idx];
    }

    // will raise compile error in debug mode
    // *(reinterpret_cast<int4*>(fdata) + idx) =
    // *reinterpret_cast<int4*>(fval_reg);
    stg128(*reinterpret_cast<uint32_t*>(fval_reg),
           *reinterpret_cast<uint32_t*>(fval_reg + 2),
           *reinterpret_cast<uint32_t*>(fval_reg + 4),
           *reinterpret_cast<uint32_t*>(fval_reg + 6),
           reinterpret_cast<int4*>(fdata) + idx);
  } else if (idx < packConfig.nThread) {
    idx = idx - packConfig.nPack + packConfig.nPack * UNROLL;
    int n_idx = scaleDivMod.Mod(idx);
    fdata[idx] = (static_cast<FT>(qdata[idx]) - zeros[n_idx]) * scales[n_idx];
  }
}

template <typename FT, typename QT>
__global__ void dequantize_rhs_a16w8_subc_kernel(
    const QT* data_in, const FT* scales, const FT* zeros, FT* data_out,
    const uint32_t N, const uint32_t K, const uint32_t GroupSize) {
  const int k_idx = blockIdx.y;
  const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx < N && k_idx < K) {
    const int group_idx = k_idx / GroupSize * N + n_idx;
    const FT ld_reg = FT(data_in[k_idx * N + n_idx]);
    const FT s_reg = scales[group_idx];
    const FT z_reg = zeros[group_idx];
    FT res = (ld_reg - z_reg) * s_reg;
    data_out[k_idx * N + n_idx] = res;
  }
}

template <typename T, int N>
struct SimpleArray {
  T data[N];
};

// Now only support UNROLL = 8
template <int BLOCK, int UNROLL, typename FT, typename QT>
__global__ void dequantize_rhs_a16w8_subc_opt_kernel(
    const QT* qdata, const FT* scales, const FT* zeros, FT* fdata,
    const SimpleArray<U32DivMod, 2> scaleDivMod, PackedEltwiseConfig packConfig,
    const uint32_t N, const uint32_t GroupSize) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

  if (idx < packConfig.nPack) {
    QT qval_reg[UNROLL];
    FT fval_reg[UNROLL];

    *reinterpret_cast<int2*>(qval_reg) =
        *(reinterpret_cast<const int2*>(qdata) + idx);

#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      auto dm = scaleDivMod.data[0].DivMod(idx * UNROLL + i);
      int n_idx = dm.mod;
      int k_idx = dm.div;
      int kgroup_idx = scaleDivMod.data[1].Div(k_idx);
      fval_reg[i] =
          (static_cast<FT>(qval_reg[i]) - zeros[kgroup_idx * N + n_idx]) *
          scales[kgroup_idx * N + n_idx];
    }

    // will raise compile error in debug mode
    // *(reinterpret_cast<int4*>(fdata) + idx) =
    // *reinterpret_cast<int4*>(fval_reg);
    stg128(*reinterpret_cast<uint32_t*>(fval_reg),
           *reinterpret_cast<uint32_t*>(fval_reg + 2),
           *reinterpret_cast<uint32_t*>(fval_reg + 4),
           *reinterpret_cast<uint32_t*>(fval_reg + 6),
           reinterpret_cast<int4*>(fdata) + idx);
  } else if (idx < packConfig.nThread) {
    idx = idx - packConfig.nPack + packConfig.nPack * UNROLL;
    auto dm = scaleDivMod.data[0].DivMod(idx);
    int n_idx = dm.mod;
    int k_idx = dm.div;
    int kgroup_idx = scaleDivMod.data[1].Div(k_idx);
    fdata[idx] = (static_cast<FT>(qdata[idx]) - zeros[kgroup_idx * N + n_idx]) *
                 scales[kgroup_idx * N + n_idx];
  }
}

template <typename FT, typename QT>
void dequantize_rhs_a16w8(const QT* qdata, const FT* scales, const FT* zeros,
                          FT* fdata, const uint32_t N, const uint32_t K,
                          const int GroupSize, cudaStream_t stream) {
  if (GroupSize == -1) {
// Note : PerChannel
#if 0
        const uint32_t THREADS = 512;
        const dim3 DIM = dim3(DivCeil(N, THREADS), K, 1);
        dequantize_rhs_a16w8_kernel<FT, QT> << < DIM, THREADS, 0, stream >> > (
            qdata, scales, zeros, fdata, N, K);
#endif
    // OPT-Kernel
    int packSize = std::min(GetPackSize(qdata), GetPackSize(fdata));
    const int64_t BLOCK_SIZE = 128;
    PackedEltwiseConfig packConfig(N * K, packSize, BLOCK_SIZE);
    U32DivMod scaleDivMod(N);
    switch (packSize) {
      case 8:
        dequantize_rhs_a16w8_opt_kernel<BLOCK_SIZE, 8, FT, QT>
            <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
                qdata, scales, zeros, fdata, scaleDivMod, packConfig);
        break;
      default:
        LOG(ERROR) << "Now only support in/out ptr is 16-byte aligned";
        break;
    }
  } else {
// Note : For SubChannel
#if 0
        const uint32_t THREADS_PER_BLOCK = 256;
        const dim3 BLOCK_DIM = dim3(DivCeil(N, THREADS_PER_BLOCK), K, 1);
        dequantize_rhs_a16w8_subc_kernel<FT, QT> << <BLOCK_DIM, THREADS_PER_BLOCK, 0, stream >> > (
            qdata, scales, zeros, fdata, N, K, GroupSize);
#else
    int packSize = std::min(GetPackSize(qdata), GetPackSize(fdata));
    const int64_t BLOCK_SIZE = 128;
    PackedEltwiseConfig packConfig(N * K, packSize, BLOCK_SIZE);
    SimpleArray<U32DivMod, 2> scaleDivMod;
    scaleDivMod.data[0] = U32DivMod(N);
    scaleDivMod.data[1] = U32DivMod(GroupSize);

    switch (packSize) {
      case 8:
        dequantize_rhs_a16w8_subc_opt_kernel<BLOCK_SIZE, 8, FT, QT>
            <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
                qdata, scales, zeros, fdata, scaleDivMod, packConfig, N,
                GroupSize);
        break;
      default:
        LOG(ERROR) << "Now only support in/out ptr is 16-byte aligned";
        break;
    }
#endif
  }
}

// restore from N32K16 order to original N-major order
// K % 16 == 0, N % 8 == 0
// each block process 64(k) * 32(n) result elements
template <typename FT, typename QT>
__global__ void restore_N32_K16_dequantize_rhs_a16w8_perc_kernel(
    const QT* qdata, const FT* scales, const FT* zeros, FT* fdata,
    const int N_32align, const int N, const int K) {
  __shared__ FT smem[64 * 32];
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  const int src_row_idx = blockIdx.x * 8 + lane_id / 4;
  const int src_col_idx =
      blockIdx.y * 64 * 4 + warp_id * 16 * 4 + (lane_id % 4) * 16;
  const int src_offset = src_row_idx * K * 4 + src_col_idx;
  int params_nidx = blockIdx.x * 32 + (lane_id / 4) * 4;

  QT qval_reg[16];
  if (src_col_idx < (K * 4)) {
    *(reinterpret_cast<uint4*>(qval_reg)) =
        *(reinterpret_cast<const uint4*>(qdata + src_offset));
  }
  FT scale_reg[4];
  *(reinterpret_cast<uint2*>(scale_reg)) =
      *(reinterpret_cast<const uint2*>(scales + params_nidx));
  FT zero_reg[4];
  *(reinterpret_cast<uint2*>(zero_reg)) =
      *(reinterpret_cast<const uint2*>(zeros + params_nidx));
  FT fval_reg[16];

  const int sts_base_offset =
      (warp_id * 16 + (lane_id % 4) * 2) * 32 + lane_id / 4;
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      QT val = qval_reg[ni * 4 + ki];
      fval_reg[ni * 4 + ki] =
          (static_cast<FT>(val) - zero_reg[ni]) * scale_reg[ni];
      int sts_offset = sts_base_offset + ((ki / 2) * 8 + (ki % 2)) * 32 +
                       ((ni + lane_id % 4) % 4) * 8;
      smem[sts_offset] = fval_reg[ni * 4 + ki];
    }
  }
  __syncthreads();

  const int lds_base_offset =
      (threadIdx.x / 4) * 32 + ((threadIdx.x % 4 + threadIdx.x / 8) % 4) * 8;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    *reinterpret_cast<uint4*>(fval_reg + i * 8) =
        *reinterpret_cast<uint4*>(smem + lds_base_offset + i * 32 * 32);
  }

  const int dst_row_base_kidx = blockIdx.y * 64 + threadIdx.x / 4;
  const int dst_col_nidx = blockIdx.x * 32 + (threadIdx.x % 4) * 8;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int dst_row_kidx = dst_row_base_kidx + i * 32;
    int dst_offset = dst_row_kidx * N + dst_col_nidx;
    if (dst_row_kidx < K && dst_col_nidx < N) {
      *reinterpret_cast<uint4*>(fdata + dst_offset) =
          *reinterpret_cast<uint4*>(fval_reg + i * 8);
    }
  }
}

template <typename FT, typename QT>
void restore_N32_K16_dequantize_rhs_a16w8(const QT* qdata, const FT* scales,
                                          const FT* zeros, FT* fdata,
                                          const int N_32align, const int N,
                                          const int K, const int GroupSize,
                                          cudaStream_t stream) {
  assert(N % 8 == 0 && K % 16 == 0 && N_32align % 32 == 0);
  if (GroupSize == -1) {
    const int BLOCK = 128;
    dim3 grid(N_32align / 32, ((K / 16) + 3) / 4);
    restore_N32_K16_dequantize_rhs_a16w8_perc_kernel<FT, QT>
        <<<grid, BLOCK, 0, stream>>>(qdata, scales, zeros, fdata, N_32align, N,
                                     K);
  }
  // TODO: Support SubChannel
  else {
    LOG(ERROR) << "Now only support PerChannel";
  }
}

/**
 * GEMV F16W8 + SplitK
 * SplitK==512
 *
 */
template <typename QT, int UNROLL_M, int UNROLL_N, int UNROLL_K, int WARP_NUM,
          int SplitK>
__global__ void gemv_f16w8_perc_splitk_kernel(
    const half* data_in, const QT* weight, const half* scales,
    const half* zeros, const int M, const int N, const int K, half* data_out) {
  const int UNROLL_N_2 = UNROLL_N / 2;
  const int32_t WARP_SIZE = 32;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int n_idx = (blockIdx.x * WARP_SIZE + lane_id) * UNROLL_N;
  const int k_start = blockIdx.y * SplitK;

  // if (n_idx >= N || (k_start + warp_id) >= K) return;
  __shared__ __half2 smem_redsum[WARP_NUM][UNROLL_M][WARP_SIZE * UNROLL_N_2];
#pragma unroll
  for (int mi = 0; mi < UNROLL_M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < UNROLL_N_2; ++ni) {
      smem_redsum[warp_id][mi][ni * WARP_SIZE + lane_id] =
          __half2half2(half(0));
    }
  }

  // Load A to smem
  __shared__ half smem_a[UNROLL_M][SplitK];
// init 0
#pragma unroll
  for (int mi = warp_id; mi < UNROLL_M; mi += WARP_NUM) {
#pragma unroll
    for (int ki = lane_id; ki < SplitK; ki += WARP_SIZE) {
      smem_a[mi][ki] = half(0);
    }
  }
  const int depth = min(K - k_start, SplitK);
  for (int mi = warp_id; mi < M; mi += WARP_NUM) {
    // const __half2* in_ptr = reinterpret_cast<const __half2*>(data_in + mi * K
    // + k_start);
    // __half2* smem_ptr = reinterpret_cast<__half2*>(&smem_a[mi][0]);
    // for (int ki = lane_id;ki < depth / 2;ki += WARP_SIZE) {
    //     smem_ptr[ki] = in_ptr[ki];
    // }
    for (int ki = lane_id; ki < depth; ki += WARP_SIZE) {
      smem_a[mi][ki] = data_in[mi * K + k_start + ki];
    }
  }

  if (n_idx >= N || (k_start + warp_id) >= K) return;
  __half2 scale[UNROLL_N_2];
  __half2 zero[UNROLL_N_2];
  load_gdata<__half2, UNROLL_N_2>(scale, scales + n_idx);
  load_gdata<__half2, UNROLL_N_2>(zero, zeros + n_idx);

  __syncthreads();

  const QT* weight_ptr = weight + k_start * N + n_idx;

  __half2 reg_a[UNROLL_M];
  __half2 reg_b[UNROLL_N_2];
  __half2 reg_c[UNROLL_M][UNROLL_N_2];
#pragma unroll
  for (int i = 0; i < UNROLL_M; ++i) {
#pragma unroll
    for (int j = 0; j < UNROLL_N_2; ++j) {
      reg_c[i][j] = __half2half2(half(0));
    }
  }

  half lda[UNROLL_M][UNROLL_K];
  QT ldb[UNROLL_K][UNROLL_N];
  for (int di = warp_id * UNROLL_K; di < depth; di += (WARP_NUM * UNROLL_K)) {
    // Load Weight
    // #pragma unroll
    // for (int i = 0;i < UNROLL_K;++i) {
    //     load_gdata<QT, UNROLL_N>(ldb[i], weight_ptr + (di + i) * N);
    // }
    if (di + UNROLL_K <= depth) {
#pragma unroll
      for (int i = 0; i < UNROLL_K; ++i) {
        load_gdata<QT, UNROLL_N>(ldb[i], weight_ptr + (di + i) * N);
      }
    } else {
#pragma unroll
      for (int i = 0; i < UNROLL_K; ++i) {
#pragma unroll
        for (int j = 0; j < UNROLL_N; ++j) {
          ldb[i][j] = 0;
        }
        if (di + i < depth) {
          load_gdata<QT, UNROLL_N>(ldb[i], weight_ptr + (di + i) * N);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < UNROLL_M; ++i) {
#pragma unroll
      for (int j = 0; j < UNROLL_K; ++j) {
        lda[i][j] = smem_a[i][di + j];
      }
    }

#pragma unroll
    for (int ki = 0; ki < UNROLL_K; ++ki) {
#pragma unroll
      for (int i = 0; i < UNROLL_M; ++i) {
        reg_a[i] = __half2half2(lda[i][ki]);
      }
#pragma unroll
      for (int i = 0; i < UNROLL_N_2; ++i) {
        reg_b[i] =
            __halves2half2(half(ldb[ki][i * 2]), half(ldb[ki][i * 2 + 1]));
        reg_b[i] = (reg_b[i] - zero[i]) * scale[i];
      }
// FMA
#pragma unroll
      for (int mi = 0; mi < UNROLL_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < UNROLL_N_2; ++ni) {
          reg_c[mi][ni] = __hfma2(reg_a[mi], reg_b[ni], reg_c[mi][ni]);
        }
      }
    }
  }

// __shared__ __half2 smem_redsum[WARP_NUM][UNROLL_M][WARP_SIZE * UNROLL_N_2];
#pragma unroll
  for (int mi = 0; mi < UNROLL_M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < UNROLL_N_2; ++ni) {
      smem_redsum[warp_id][mi][ni * WARP_SIZE + lane_id] = reg_c[mi][ni];
    }
  }
  __syncthreads();

  for (int mi = warp_id; mi < M; mi += WARP_NUM) {
    __half2 sum[UNROLL_N_2] = {__half2half2(half(0))};

#pragma unroll
    for (int wi = 0; wi < WARP_NUM; ++wi) {
#pragma unroll
      for (int ni = 0; ni < UNROLL_N_2; ++ni) {
        // sum[ni] += smem_redsum[wi][mi][ni * WARP_SIZE + lane_id];
        sum[ni] =
            __hadd2(sum[ni], smem_redsum[wi][mi][ni * WARP_SIZE + lane_id]);
      }
    }
    store_gdata<__half2, UNROLL_N_2>(
        sum, data_out + mi * N + blockIdx.y * M * N + n_idx);
  }
}

// [NumSplitK, M, N]
template <template <class> class ActiveFunc>
__global__ void reduce_sum(const half* data_in, const half* bias,
                           half* data_out, const int M, const int N,
                           const int K, const int NumSplitK,
                           const float alpha) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t m_idx = blockIdx.y;

  const half* data_in_ptr = data_in + m_idx * N;
  half* data_out_ptr = data_out + m_idx * N;
  if (tid < N) {
    half bias_val = half(0);
    if (bias != nullptr) {
      bias_val = bias[tid];
    }
    half sum = 0;
    for (int i = 0; i < NumSplitK; ++i) {
      half val = data_in_ptr[i * M * N + tid];
      sum = __hadd(sum, val);
    }
    // C = alpha * A * B
    sum = static_cast<half>(float(sum) * alpha);
    sum += bias_val;
    data_out_ptr[tid] = ActiveFunc<half>::Op(sum);
  }
}

template <typename QT, template <class> class ActiveFunc>
void gemv_f16w8_perc_splitk(const half* lhs, const QT* rhs, const half* scales,
                            const half* zeros, const half* bias, half* data_out,
                            const uint32_t M, const uint32_t N,
                            const uint32_t K, void* workspace,
                            const float alpha, cudaStream_t stream) {
  const uint32_t SplitK = 512;
  {
    const uint32_t UNROLL_M = 8;
    const uint32_t UNROLL_K = 4;
    const uint32_t WARP_NUM = 8;
    const uint32_t THREADS_PER_BLOCK = 32 * WARP_NUM;
    if (N % 8 == 0) {
      const uint32_t UNROLL_N = 8;  // 范围【2-8】
      const dim3 BLOCK_DIM =
          dim3(DivCeil(N, 32 * UNROLL_N), DivCeil(K, SplitK), 1);
      gemv_f16w8_perc_splitk_kernel<QT, UNROLL_M, UNROLL_N, UNROLL_K, WARP_NUM,
                                    SplitK>
          <<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
              lhs, rhs, scales, zeros, M, N, K, static_cast<half*>(workspace));
    } else if (N % 2 == 0) {
      const uint32_t UNROLL_N = 2;  // 范围【2-8】
      const dim3 BLOCK_DIM =
          dim3(DivCeil(N, 32 * UNROLL_N), DivCeil(K, SplitK), 1);
      gemv_f16w8_perc_splitk_kernel<QT, UNROLL_M, UNROLL_N, UNROLL_K, WARP_NUM,
                                    SplitK>
          <<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
              lhs, rhs, scales, zeros, M, N, K, static_cast<half*>(workspace));
    }
  }

  {
    const uint32_t THREADS_PER_BLOCK = 128;
    const dim3 BLOCK_DIM = dim3(DivCeil(N, THREADS_PER_BLOCK), M, 1);
    reduce_sum<ActiveFunc><<<BLOCK_DIM, THREADS_PER_BLOCK, 0, stream>>>(
        static_cast<const half*>(workspace), bias, data_out, M, N, K,
        DivCeil(K, SplitK), alpha);
  }
}

/**
 * @brief
 *
 */

/*
 * matrix_A/B/C: row_major
 * k % 8 == 0, n % 8 == 0
 * accumulator precision: FP16 or FP32
 * output datatype: FP16
 *
 * m128n128k32 thread block tile
 * m64n64k32 warp tile
 */
template <typename QT, typename AccumulatorType>
__global__ void __launch_bounds__(128, 2)
    hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8_kernel(
        const half* A, const QT* B, const half* B_scale, const half* B_zero,
        half* C, uint32_t m, uint32_t n, uint32_t k,
        uint32_t A_ldg_step,    // k * sizeof(half) * 32
        uint32_t B_ldg_step) {  // n * sizeof(int8_t) * 16
  static_assert(::cuda::std::is_same_v<AccumulatorType, half> ||
                    ::cuda::std::is_same_v<AccumulatorType, float>,
                "");
  constexpr int PACK = ::cuda::std::is_same_v<AccumulatorType, half> ? 2 : 1;
  /*
   * matrix A & B thread block tile shared memory (double buffer)
   * matrix A: 128 * 32 * 2Byte/item * double buffer = 8KB * 2
   * matrix B: 128 * 32 * 2Byte/item * double buffer = 8KB * 2
   */
  __shared__ __align__(1024 * 16) char smem[1024 * 32];
  half* A_smem = reinterpret_cast<half*>(smem);
  half* B_smem = reinterpret_cast<half*>(smem + 1024 * 8);

  // A, B and C register fragment
  uint32_t A_frag[2][8];
  uint32_t B_frag[2][8];
  uint32_t C_frag[8][16 / PACK];

  if (::cuda::std::is_same_v<AccumulatorType, half>) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        C_frag[i][j] = 0;
      }
    }
  } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        C_frag[i][j] = 0;
      }
    }
  }

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = threadIdx.x / 32;

  /*
   * A_tile & B_tile ldg pointer.
   * A_tile: 32x4 thread, 4 LDG each thread (16 register, 4 predicate)
   * B_tile: 16x8 thread, 2x2 LDG each thread (16 register, 2 predicate)
   */
  const char* A_ldg_ptr = reinterpret_cast<const char*>(
      A + (blockIdx.x * 128 + threadIdx.x / 4) * k + (threadIdx.x % 4) * 8);
  const int B_n_idx = blockIdx.y * 128 + (threadIdx.x % 8) * 8;
  const char* B_ldg_ptr =
      reinterpret_cast<const char*>(B + (threadIdx.x / 8) * n + B_n_idx);

  /*
   * A_tile & B_tile sts addr.
   * using uint32_t pointer for faster double buffer switch
   */
  uint32_t A_sts_addr =
      smem_u32addr(A_smem + (threadIdx.x % 4) * (128 * 8) +
                   ((threadIdx.x / 4) ^ (threadIdx.x % 4 * 2)) * 8);
  uint32_t B_sts_addr =
      smem_u32addr(B_smem + (threadIdx.x / 8) * 128 +
                   ((threadIdx.x % 8) ^ (threadIdx.x / 8 % 8)) * 8);

  // A_tile lds addr, 1 addr for 1 k_frag
  uint32_t A_lds_addr[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    A_lds_addr[i] = smem_u32addr(A_smem + i * 128 * 8 + (warp_id / 2) * 64 * 8 +
                                 (lane_id ^ (i * 2)) * 8);
  }

  // B_tile lds addr
  uint32_t B_lds_addr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    B_lds_addr[i] =
        smem_u32addr(B_smem + (lane_id % 8) * 128 + (warp_id % 2) * 64 +
                     ((lane_id / 8 + i * 4) ^ (lane_id % 8)) * 8);
  }

  // ldg_guard to avoid LDG out of bound
  bool A_ldg_guard[4]{};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m_idx = blockIdx.x * 128 + threadIdx.x / 4 + i * 32;
    if (m_idx < m) {
      A_ldg_guard[i] = true;
    }
  }

  bool B_ldg_guard[2]{};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int n_idx = blockIdx.y * 128 + (threadIdx.x % 8) * 8 + i * 64;
    if (n_idx < n) {
      B_ldg_guard[i] = true;
    }
  }

  uint32_t A_ldg_reg[4][4];
  QT B_ldg_reg_q[4][8];
  __half2 B_scale_reg[2][4];
  __half2 B_zero_reg[2][4];
  __half2 B_ldg_reg[4][4];

  // 1'st A&B tile loaded before the k_tile loop
  uint32_t k_tiles = (k + 31) / 32 - 1;

  // load 1'st tile to shared memory
  {
    uint32_t first_k_tile = k - k_tiles * 32;

    const char* first_A_ldg_ptr = A_ldg_ptr + (k - first_k_tile) * sizeof(half);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bool guard = A_ldg_guard[i] && (threadIdx.x % 4) * 8 < first_k_tile;
      ldg128_cg_0(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2],
                  A_ldg_reg[i][3], first_A_ldg_ptr + i * A_ldg_step, guard);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts128(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2], A_ldg_reg[i][3],
             A_sts_addr + i * 32 * 8 * sizeof(half));
    }

    // load B, B_scale, B_zero
    const char* first_B_ldg_ptr =
        B_ldg_ptr + (k - first_k_tile) * n * sizeof(QT);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        bool guard = B_ldg_guard[j] && threadIdx.x / 8 + i * 16 < first_k_tile;
        ldg64_nc_0(
            *reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j]),
            *reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j] + 4),
            first_B_ldg_ptr + i * 16 * n * sizeof(QT) + j * 64 * sizeof(QT),
            guard);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ldg128_ca_0(B_scale_reg[i][0], B_scale_reg[i][1], B_scale_reg[i][2],
                  B_scale_reg[i][3], B_scale + B_n_idx + i * 64,
                  B_ldg_guard[i]);
      ldg128_ca_0(B_zero_reg[i][0], B_zero_reg[i][1], B_zero_reg[i][2],
                  B_zero_reg[i][3], B_zero + B_n_idx + i * 64, B_ldg_guard[i]);
    }

// dequant B from int8 -> fp16
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          B_ldg_reg[i * 2 + j][k].x =
              __hmul(__hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k]),
                            B_zero_reg[j][k].x),
                     B_scale_reg[j][k].x);
          B_ldg_reg[i * 2 + j][k].y = __hmul(
              __hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k + 1]),
                     B_zero_reg[j][k].y),
              B_scale_reg[j][k].y);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        sts128(
            B_ldg_reg[i * 2 + j][0], B_ldg_reg[i * 2 + j][1],
            B_ldg_reg[i * 2 + j][2], B_ldg_reg[i * 2 + j][3],
            B_sts_addr + i * 16 * 128 * sizeof(half) + j * 64 * sizeof(half));
      }
    }

    __syncthreads();
  }

  // smem double buffer offset
  uint32_t lds_offset = 0;
  uint32_t sts_offset = 16 * 1024;

  // load 1'st fragment
  ldsm_4(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
         A_lds_addr[0] + lds_offset);
  ldsm_4(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
         A_lds_addr[0] + lds_offset + 32 * 8 * sizeof(half));
  ldsm_4_trans(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
               B_lds_addr[0] + lds_offset);
  ldsm_4_trans(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
               B_lds_addr[1] + lds_offset);

  // k_tiles loop
  for (; k_tiles > 0; --k_tiles) {
#pragma unroll
    for (int k_frag = 0; k_frag < 4; ++k_frag) {
      // store next A&B tile to shared memory
      if (k_frag == 3) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sts128(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2],
                 A_ldg_reg[i][3],
                 A_sts_addr + sts_offset + i * 32 * 8 * sizeof(half));
        }

// dequant B from int8 -> fp16
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              B_ldg_reg[i * 2 + j][k].x = __hmul(
                  __hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k]),
                         B_zero_reg[j][k].x),
                  B_scale_reg[j][k].x);
              B_ldg_reg[i * 2 + j][k].y = __hmul(
                  __hsub(static_cast<half>(B_ldg_reg_q[i * 2 + j][2 * k + 1]),
                         B_zero_reg[j][k].y),
                  B_scale_reg[j][k].y);
            }
          }
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            sts128(B_ldg_reg[i * 2 + j][0], B_ldg_reg[i * 2 + j][1],
                   B_ldg_reg[i * 2 + j][2], B_ldg_reg[i * 2 + j][3],
                   B_sts_addr + sts_offset + i * 16 * 128 * sizeof(half) +
                       j * 64 * sizeof(half));
          }
        }

        __syncthreads();

        // switch double buffer
        lds_offset ^= 16 * 1024;
        sts_offset ^= 16 * 1024;
      }

      // load next A&B fragment from shared memory to register
      ldsm_4(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
             A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset);
      ldsm_4(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
             A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset + 32 * 8 * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr[0] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr[1] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));

      // load next A&B tile
      if (k_frag == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ldg128_cache_hint(A_ldg_reg[i][0], A_ldg_reg[i][1], A_ldg_reg[i][2],
                            A_ldg_reg[i][3], A_ldg_ptr + i * A_ldg_step,
                            A_ldg_guard[i], LoadCacheOperator::CG);
        }

        A_ldg_ptr += 32 * sizeof(half);

#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            ldg64_nc(*reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j]),
                     *reinterpret_cast<uint32_t*>(B_ldg_reg_q[i * 2 + j] + 4),
                     B_ldg_ptr + j * 64 * sizeof(QT), B_ldg_guard[j]);
          }
          B_ldg_ptr += B_ldg_step;
        }
      }

// HMMA loop
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (::cuda::std::is_same_v<AccumulatorType, half>) {
            hmma1688_f16(C_frag[i * 2][j], C_frag[i * 2 + 1][j],
                         A_frag[k_frag % 2][i * 2],
                         A_frag[k_frag % 2][i * 2 + 1], B_frag[k_frag % 2][j]);
          } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
            hmma1688_f32(C_frag[i * 2][j * 2], C_frag[i * 2][j * 2 + 1],
                         C_frag[i * 2 + 1][j * 2], C_frag[i * 2 + 1][j * 2 + 1],
                         A_frag[k_frag % 2][i * 2],
                         A_frag[k_frag % 2][i * 2 + 1], B_frag[k_frag % 2][j]);
          }
        }
      }
    }
  }

// FFMA for the last tile
#pragma unroll
  for (int k_frag = 0; k_frag < 4; ++k_frag) {
    if (k_frag < 3) {
      // load next A&B fragment from shared memory to register
      ldsm_4(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
             A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset);
      ldsm_4(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
             A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
             A_lds_addr[(k_frag + 1) % 4] + lds_offset + 32 * 8 * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr[0] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));
      ldsm_4_trans(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr[1] + lds_offset +
                       ((k_frag + 1) % 4) * (8 * 128) * sizeof(half));
    }

// HMMA loop
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (::cuda::std::is_same_v<AccumulatorType, half>) {
          hmma1688_f16(C_frag[i * 2][j], C_frag[i * 2 + 1][j],
                       A_frag[k_frag % 2][i * 2], A_frag[k_frag % 2][i * 2 + 1],
                       B_frag[k_frag % 2][j]);
        } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
          hmma1688_f32(C_frag[i * 2][j * 2], C_frag[i * 2][j * 2 + 1],
                       C_frag[i * 2 + 1][j * 2], C_frag[i * 2 + 1][j * 2 + 1],
                       A_frag[k_frag % 2][i * 2], A_frag[k_frag % 2][i * 2 + 1],
                       B_frag[k_frag % 2][j]);
        }
      }
    }
  }

  /*
   * C_tile write back, reuse A&B tile shared memory buffer
   * 32x64 (32x72 with padding) FP16 C_tile_writeback
   * padding 8 items each row to avoid bank conflict.
   *     =>32x32 (32x36 with padding) uint32_t writeback tile
   *
   * 4x8 thread STG.128
   */
  uint32_t C_sts_size = 32 * 72 * sizeof(half);
  uint32_t mma_tid_x = lane_id % 4;
  uint32_t mma_tid_y = lane_id / 4;
  uint32_t* C_sts_ptr =
      reinterpret_cast<uint32_t*>(smem + warp_id * C_sts_size) +
      mma_tid_y * 36 + mma_tid_x;
  uint4* C_lds_ptr = reinterpret_cast<uint4*>(smem + warp_id * C_sts_size) +
                     lane_id / 8 * 9 + lane_id % 8;

  uint32_t m_idx = blockIdx.x * 128 + warp_id / 2 * 64 + lane_id / 8;
  uint32_t n_idx = blockIdx.y * 128 + warp_id % 2 * 64 + lane_id % 8 * 8;

  half* C_stg_ptr = C + m_idx * n + n_idx;

  bool n_guard = n_idx < n;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    __syncthreads();

// C_tile sts
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        if (::cuda::std::is_same_v<AccumulatorType, half>) {
          C_sts_ptr[j * 8 * 36 + p * 4] = C_frag[i * 4 + j][p];
        } else if (::cuda::std::is_same_v<AccumulatorType, float>) {
          half low16 = static_cast<half>(
              reinterpret_cast<float&>(C_frag[i * 4 + j][p * 2]));
          half high16 = static_cast<half>(
              reinterpret_cast<float&>(C_frag[i * 4 + j][p * 2 + 1]));
          C_sts_ptr[j * 8 * 36 + p * 4] =
              reinterpret_cast<uint32_t&>(low16) & 0xffff;
          C_sts_ptr[j * 8 * 36 + p * 4] |= reinterpret_cast<uint32_t&>(high16)
                                           << 16;
        }
      }
    }

    __syncthreads();

// C_tile stg
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint4 stg_reg = C_lds_ptr[j * 9 * 4];
      stg128(stg_reg.x, stg_reg.y, stg_reg.z, stg_reg.w,
             C_stg_ptr + (i * 32 + j * 4) * n,
             m_idx + i * 32 + j * 4 < m && n_guard);
    }
  }
}

template <typename QT, template <class> class ActiveFunc>
void hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8(
    const half* A, const QT* B, const half* B_scale, const half* B_zero,
    const half* bias, half* C, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const float alpha, cudaStream_t stream) {
  {
    uint32_t grid_x = (M + 127) / 128;
    uint32_t grid_y = (N + 127) / 128;
    dim3 grid(grid_x, grid_y);
    uint32_t A_ldg_step = K * sizeof(half) * 32;
    uint32_t B_ldg_step = N * sizeof(QT) * 16;

    hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8_kernel<QT, float>
        <<<grid, 128, 0, stream>>>(A, B, B_scale, B_zero, C, M, N, K,
                                   A_ldg_step, B_ldg_step);
  }
  {
    const uint32_t BLOCK_SIZE = 128;
    const uint32_t UNROLL = 4;
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
    add_bias<UNROLL, ActiveFunc>
        <<<grid, BLOCK_SIZE, 0, stream>>>(bias, C, M, N, alpha);
  }
}

/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * FType is half ot bfloat16, QType is int8
 * requiring N % 8 == 0， K % 16 == 0 by loading uint
 * BN is obtained by padding the original N to a multiple of 32
 * weight B is rearranged as N32K16 order,
 * i.e. a initial data block of size 32(n)x16(k) is reordered as n8k4n4k4，
 * in order to put data loaded by the same thread of 32x16 data block together
 * continuously (see
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type)
 */
template <typename FType, typename QType, int NStage, int BLOCK_SIZE>
struct GmemTile_A16W8_PerC_64x128x32_multistage_SM8x_SplitK {
  // element num loaded by a LDG inst.
  const int LDG_ELEMENT_CNT_A = 8;
  const int LDG_ELEMENT_CNT_B = 16;
  const int WARP_SIZE = 32;

  __device__ GmemTile_A16W8_PerC_64x128x32_multistage_SM8x_SplitK(
      const SM8x_GEMM_A16W8_Params<FType, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& BQ_smem_addr,
      const uint32_t& A_stage_stride, const uint32_t& BQ_stage_stride)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        BQ_smem_base_addr(BQ_smem_addr),
        A_smem_stage_stride(A_stage_stride),
        BQ_smem_stage_stride(BQ_stage_stride) {
    this_block_A_base_ptr =
        params.A_ptr + blockIdx.x * 64 * params.K + blockIdx.z * params.SplitK;
    // here B is rearranged as N32K16 order, i.e. 4 continuous N-direction
    // 8(N)x16(K) size data blocks are packed together
    this_block_B_base_ptr = params.B_ptr + blockIdx.y * 128 * params.K +
                            blockIdx.z * params.SplitK * 4;

    const int lane_id = threadIdx.x % WARP_SIZE;

    // For matrix A, a block load/store 64(row) x 32(col) elements in 2 iter
    // and a block load/store 32(row) * 32(col) elements per iter
    // 8x4 warp load/store 8(row) x 32(col) elements per iter
    const int Aldg_row_base_idx = threadIdx.x / 4;
    Aldg_col_idx = (threadIdx.x % 4) * LDG_ELEMENT_CNT_A;
    const int Aldg_base_offset = Aldg_row_base_idx * params.K + Aldg_col_idx;

    // For matrix B, a block load/store  elements of 32(row) x 128(col) of
    // N32K16_n4k2 packing in 2 iter and a block load/store 16(row) * 128(col)
    // elements per iter 4x8 warp load/store 16(row) * 128(col) per iter
    Bldg_col_idx = (threadIdx.x % 8) * LDG_ELEMENT_CNT_B;
    const int Bldg_row_base_idx = threadIdx.x / 8;
    const int Bldg_base_offset =
        Bldg_row_base_idx * params.K * 4 + Bldg_col_idx;

    this_block_A_base_ptr += Aldg_base_offset;
    this_block_B_base_ptr += Bldg_base_offset;

    const int sts_a_base_offset =
        (threadIdx.x / 4) * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;
    const int sts_bq_base_offset =
        Bldg_row_base_idx * 32 * 4 +
        ((threadIdx.x % 8) ^ (((threadIdx.x / 8) % 2) * 4)) * LDG_ELEMENT_CNT_B;

    A_smem_base_addr += sts_a_base_offset * sizeof(FType);
    BQ_smem_base_addr += sts_bq_base_offset * sizeof(int8_t);

    A_ldg_guard = 0;
    B_ldg_guard = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int m_idx = blockIdx.x * 64 + Aldg_row_base_idx + i * 32;
      if (m_idx < params.M) {
        A_ldg_guard |= (1u << i);
      }
    }

    const int N_padded = (params.N + 31) / 32 * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int n_idx = blockIdx.y * 128 + (Bldg_row_base_idx / 8) * 32 + i * 64;
      if (n_idx < N_padded) {
        B_ldg_guard |= (1u << i);
      }
    }
  }

  __device__ void ldgsts_first_ktiles(const int& first_k_tile,
                                      const int& k_tiles) {
    // load first k_tile
    // load A
    const int A_src_size = Aldg_col_idx < first_k_tile ? 16 : 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cp_async<16>(A_smem_base_addr + (i * 32 * 32) * sizeof(FType),
                   this_block_A_base_ptr + i * 32 * params.K, A_src_size,
                   (A_ldg_guard & (1u << i)) != 0);
    }

    // load B
    const int B_src_size = (Bldg_col_idx / 4) < first_k_tile ? 16 : 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cp_async<16>(BQ_smem_base_addr + (i * 64 * 32) * sizeof(int8_t),
                   this_block_B_base_ptr + i * 64 * params.K, B_src_size,
                   (B_ldg_guard & (1u << i)) != 0);
    }

    cp_async_commit_group();
    this_block_A_base_ptr += first_k_tile;
    this_block_B_base_ptr += (first_k_tile * 4);

    // load second to (N-stage - 1) k_tiles
    for (int stage_idx = 1; stage_idx < NStage - 1; ++stage_idx) {
      if (stage_idx < k_tiles) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          cp_async<16>(A_smem_base_addr + stage_idx * A_smem_stage_stride +
                           (i * 32 * 32) * sizeof(FType),
                       this_block_A_base_ptr + i * 32 * params.K, 16,
                       (A_ldg_guard & (1u << i)) != 0);
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
          cp_async<16>(BQ_smem_base_addr + stage_idx * BQ_smem_stage_stride +
                           (i * 64 * 32) * sizeof(int8_t),
                       this_block_B_base_ptr + i * 64 * params.K, 16,
                       (B_ldg_guard & (1u << i)) != 0);
        }

        this_block_A_base_ptr += 32;
        this_block_B_base_ptr += (32 * 4);
      }
      cp_async_commit_group();
    }
  }

  __device__ void ldgsts(const int& sts_stage_idx) {
    const int a_stage_offset = sts_stage_idx * A_smem_stage_stride;
    const int bq_stage_offset = sts_stage_idx * BQ_smem_stage_stride;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cp_async<16>(
          A_smem_base_addr + a_stage_offset + (i * 32 * 32) * sizeof(FType),
          this_block_A_base_ptr + i * 32 * params.K, 16,
          (A_ldg_guard & (1u << i)) != 0);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cp_async<16>(
          BQ_smem_base_addr + bq_stage_offset + (i * 64 * 32) * sizeof(int8_t),
          this_block_B_base_ptr + i * 64 * params.K, 16,
          (B_ldg_guard & (1u << i)) != 0);
    }

    cp_async_commit_group();
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += (32 * 4);
  }

  const FType* this_block_A_base_ptr = nullptr;
  const QType* this_block_B_base_ptr = nullptr;

  int Aldg_col_idx;
  int Bldg_col_idx;

  uint32_t A_ldg_guard;
  uint32_t B_ldg_guard;

  uint32_t A_smem_base_addr, BQ_smem_base_addr;
  const uint32_t A_smem_stage_stride, BQ_smem_stage_stride;

  const SM8x_GEMM_A16W8_Params<FType, QType>& params;
};

/*
 * warp_tile : m64n32k32
 * N % 8 == 0
 */
template <typename FType, typename QType>
struct ComputeTile_A16W8_PerC_64x128x32_multistage_SM8x_SplitK {
  __device__ ComputeTile_A16W8_PerC_64x128x32_multistage_SM8x_SplitK(
      const SM8x_GEMM_A16W8_Params<FType, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& BQ_smem_addr,
      const uint32_t& A_stage_stride, const uint32_t& BQ_stage_stride)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        BQ_smem_base_addr(BQ_smem_addr),
        A_smem_stage_stride(A_stage_stride),
        BQ_smem_stage_stride(BQ_stage_stride) {
    const int WARP_SIZE = 32;
    const int warp_id = threadIdx.x / WARP_SIZE;
    lane_id = threadIdx.x % WARP_SIZE;

    load_a_base_offset[0] =
        (lane_id % 16) * 32 +
        ((lane_id / 16) ^ (lane_id % 4) ^ ((lane_id / 4) % 2)) * 8;
    load_a_base_offset[1] =
        (lane_id % 16) * 32 +
        ((lane_id / 16 + 2) ^ (lane_id % 4) ^ ((lane_id / 4) % 2)) * 8;

    load_b_offset[0] = (lane_id / 4 + warp_id * 8) * 32 * 4 +
                       (lane_id % 4) * 16 + ((lane_id / 4) % 2) * 16 * 4;
    load_b_offset[1] = (lane_id / 4 + warp_id * 8) * 32 * 4 +
                       (lane_id % 4) * 16 + (((lane_id / 4) % 2) ^ 1) * 16 * 4;
    const int C_sts_size = 64 * 32;
    sts_c_base_offset =
        warp_id * C_sts_size + (lane_id / 4) * 32 + (lane_id % 4) * 2;
    lds_c_base_offset = warp_id * C_sts_size + (lane_id / 4) * 32 +
                        ((lane_id % 4 + lane_id / 8) % 4) * 8;

    this_block_C_base_ptr = params.C_split_ptr +
                            blockIdx.z * params.M * params.N +
                            blockIdx.x * 64 * params.N + blockIdx.y * 128;
    store_c_row_base_idx = lane_id / 4;
    store_c_col_idx = warp_id * 32 + (lane_id % 4) * 8;
    store_c_base_offset = store_c_row_base_idx * params.N + store_c_col_idx;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          C_frag[i][j][k] = 0.f;
        }
      }
    }
    params_n_idx = blockIdx.y * 128 + (threadIdx.x / 4) * 4;
  }

  __device__ void lds(const int& smem_stage_idx, const int& reg_buf_idx,
                      const int& k_phase_idx) {
    uint32_t A_smem_addr =
        A_smem_base_addr + A_smem_stage_stride * smem_stage_idx;
    uint32_t B_smem_addr =
        BQ_smem_base_addr + BQ_smem_stage_stride * smem_stage_idx;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldsm_4(A_frag[reg_buf_idx][i][0], A_frag[reg_buf_idx][i][1],
             A_frag[reg_buf_idx][i][2], A_frag[reg_buf_idx][i][3],
             A_smem_addr + (load_a_base_offset[k_phase_idx] + i * 16 * 32) *
                               sizeof(FType));
    }

    lds128(BQ_frag[reg_buf_idx][0], BQ_frag[reg_buf_idx][1],
           BQ_frag[reg_buf_idx][2], BQ_frag[reg_buf_idx][3],
           B_smem_addr + load_b_offset[k_phase_idx] * sizeof(int8_t));

// dequant B
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      BF_frag[reg_buf_idx][2 * i][0].x =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i].x)) -
           B_zero[i].x) *
          B_scale[i].x;
      BF_frag[reg_buf_idx][2 * i][0].y =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i].y)) -
           B_zero[i].x) *
          B_scale[i].x;
      BF_frag[reg_buf_idx][2 * i][1].x =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i].z)) -
           B_zero[i].x) *
          B_scale[i].x;
      BF_frag[reg_buf_idx][2 * i][1].y =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i].w)) -
           B_zero[i].x) *
          B_scale[i].x;
      BF_frag[reg_buf_idx][2 * i + 1][0].x =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i + 1].x)) -
           B_zero[i].y) *
          B_scale[i].y;
      BF_frag[reg_buf_idx][2 * i + 1][0].y =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i + 1].y)) -
           B_zero[i].y) *
          B_scale[i].y;
      BF_frag[reg_buf_idx][2 * i + 1][1].x =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i + 1].z)) -
           B_zero[i].y) *
          B_scale[i].y;
      BF_frag[reg_buf_idx][2 * i + 1][1].y =
          (static_cast<typename HalfType<FType>::T1>(
               static_cast<float>(BQ_frag[reg_buf_idx][2 * i + 1].w)) -
           B_zero[i].y) *
          B_scale[i].y;
    }
  }

  __device__ void ldg_params() {
    const int N_padded = (params.N + 31) / 32 * 32;
    // load B scale and zero_point
    ldg64_ca(B_scale[0], B_scale[1], params.B_scale_ptr + params_n_idx,
             params_n_idx < N_padded);
    ldg64_ca(B_zero[0], B_zero[1], params.B_zero_ptr + params_n_idx,
             params_n_idx < N_padded);
  }

  __device__ void mma(const int& reg_buf_idx) {
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < 4; ++n_idx) {
        hmma16816_f32<FType>(
            C_frag[m_idx][n_idx], A_frag[reg_buf_idx][m_idx],
            reinterpret_cast<uint32_t(&)[2]>(BF_frag[reg_buf_idx][n_idx]));
      }
    }
  }

  __device__ void stg(char* smem) {
    uint32_t* C_sts_ptr =
        reinterpret_cast<uint32_t*>(smem + sts_c_base_offset * sizeof(FType));
    uint4* C_lds_ptr =
        reinterpret_cast<uint4*>(smem + lds_c_base_offset * sizeof(FType));
// C_tile sts
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
      for (int n_idx = 0; n_idx < 4; ++n_idx) {
#pragma unroll
        for (int k_idx = 0; k_idx < 2; ++k_idx) {
          FType low16 = static_cast<FType>(C_frag[m_idx][n_idx][k_idx * 2]);
          FType high16 =
              static_cast<FType>(C_frag[m_idx][n_idx][k_idx * 2 + 1]);
          uint32_t tmp = (reinterpret_cast<uint32_t&>(low16) & 0xffff) |
                         (reinterpret_cast<uint32_t&>(high16) << 16);
          C_sts_ptr[m_idx * 16 * 16 + (((lane_id / 8) + n_idx) % 4) * 4 +
                    k_idx * 8 * 16] = tmp;
        }
      }
    }

    __syncthreads();

    FType* C_base_ptr = this_block_C_base_ptr + store_c_base_offset;
    // C_tile lds and stg
    int m_base_idx = store_c_row_base_idx + blockIdx.x * 64;
    bool n_guard = (store_c_col_idx + blockIdx.y * 128) < params.N;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint4 stg_reg = C_lds_ptr[i * 8 * 4];
      stg128(stg_reg.x, stg_reg.y, stg_reg.z, stg_reg.w,
             C_base_ptr + i * 8 * params.N,
             (m_base_idx + i * 8) < params.M && n_guard);
    }
  }

  const SM8x_GEMM_A16W8_Params<FType, QType>& params;

  int load_a_base_offset[2];
  int load_b_offset[2];
  int sts_c_base_offset;
  int lds_c_base_offset;

  int store_c_base_offset;

  int store_c_row_base_idx, store_c_col_idx;
  FType* this_block_C_base_ptr = nullptr;

  int params_n_idx;
  const uint32_t A_smem_base_addr, BQ_smem_base_addr;
  const uint32_t A_smem_stage_stride, BQ_smem_stage_stride;

  int lane_id;
  // 2 denotes double buffer, first 4 denotes M direction
  uint32_t A_frag[2][4][4];

  typename HalfType<FType>::T2 B_scale[2];
  typename HalfType<FType>::T2 B_zero[2];
  char4 BQ_frag[2][4];
  // 2 denotes double buffer, first 4 denotes N direction, second 2 denotes K
  // direction
  typename HalfType<FType>::T2 BF_frag[2][4][2];
  // first 4 denotes M direction, second 4 denotes N direction
  float C_frag[4][4][4];
};

/*
 *  C = A x B
 *  matrix A: M x K, matrix B: N x K(N32K16), matrix C: M x N (NTN)
 *  N % 8 == 0, K % 16 == 0
 *  accumulator precision: FP32
 *  output datatype: FP16
 *
 *  BLOCK_TILE: m64n128k32
 *  BLOCK_SIZE: 128
 *  WARP_TILE:  m64n32k32
 *  NStage is 4 or 3
 */
template <typename FType, typename QType, int NStage>
__global__ void __launch_bounds__(128)
    ampere_hgemm_A16W8_perc_f16_f16_64x128x32_hmma16816_multistage_NT16N_nonfused_splitk_kernel(
        const SM8x_GEMM_A16W8_Params<FType, QType> params) {
  // A smem size = 64 * 32 * 2B/elem * 4(stage) = 16KB
  // B smem size = 128 * 32 * 1B/elem * 4(stage) = 16KB
  __shared__ char smem[NStage * 8 * 1024];
  char* A_smem = smem;
  char* BQ_smem = smem + 64 * 32 * 2 * NStage;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t BQ_smem_addr = smem_u32addr(BQ_smem);
  uint32_t A_smem_stage_stride = 64 * 32 * 2;
  uint32_t BQ_smem_stage_stride = 128 * 32;

  // initialize the data move process from GM to SMEM for this block
  GmemTile_A16W8_PerC_64x128x32_multistage_SM8x_SplitK<FType, QType, NStage,
                                                       128>
      gmem_tile(params, A_smem_addr, BQ_smem_addr, A_smem_stage_stride,
                BQ_smem_stage_stride);

  int sts_stage_idx = 0;
  int lds_stage_idx = 0;

  int tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                       ? params.SplitK
                       : params.K - blockIdx.z * params.SplitK;
  int k_tiles = (tb_k_slice + 31) / 32;
  int first_k_tile = tb_k_slice - (k_tiles - 1) * 32;

  // load first three tiles to shared memory
  gmem_tile.ldgsts_first_ktiles(first_k_tile, k_tiles);
  sts_stage_idx += (NStage - 2);
  ComputeTile_A16W8_PerC_64x128x32_multistage_SM8x_SplitK<FType, QType>
      compute_tile(params, A_smem_addr, BQ_smem_addr, A_smem_stage_stride,
                   BQ_smem_stage_stride);
  compute_tile.ldg_params();
  cp_asyc_wait_group<NStage - 2>();
  __syncthreads();

  compute_tile.lds(lds_stage_idx, 0, 0);
  int reg_buf_idx = 1;

  // main loop
  for (; k_tiles > NStage - 1; --k_tiles) {
    // load next A&B tile
    sts_stage_idx = sts_stage_idx < NStage - 1 ? sts_stage_idx + 1 : 0;
    gmem_tile.ldgsts(sts_stage_idx);

#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 2; k_phase_idx++) {
      // dequantize next B tile
      if (k_phase_idx == 1) {
        cp_asyc_wait_group<NStage - 2>();
        __syncthreads();
        lds_stage_idx = lds_stage_idx < NStage - 1 ? lds_stage_idx + 1 : 0;
      }

      compute_tile.lds(lds_stage_idx, reg_buf_idx, (k_phase_idx + 1) % 2);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

  // last NStage-1 tiles
  for (; k_tiles > 0; --k_tiles) {
    cp_async_commit_group();
#pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 2; k_phase_idx++) {
      // dequantize next B tile
      if (k_phase_idx == 1) {
        cp_asyc_wait_group<NStage - 2>();
        __syncthreads();
        lds_stage_idx = lds_stage_idx < NStage - 1 ? lds_stage_idx + 1 : 0;
      }

      compute_tile.lds(lds_stage_idx, reg_buf_idx, (k_phase_idx + 1) % 2);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

  compute_tile.stg(smem);
}

template <typename FType, typename QType, template <class> class ActiveFunc>
void ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk(
    const FType* A, const QType* B, const FType* B_scale, const FType* B_zero,
    const FType* bias, FType* C, const uint32_t M, const uint32_t N,
    const uint32_t K, void* workspace, const int sm_version,
    const SplitKParams splitk_params, const float alpha, cudaStream_t stream) {
  FType* C_split = static_cast<FType*>(workspace);
  int grid_x = (M + 63) / 64;
  int grid_y = (N + 127) / 128;
  int grid_z = (K + splitk_params.SplitK - 1) / splitk_params.SplitK;

  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(128);

  SM8x_GEMM_A16W8_Params<FType, QType> params{
      A,      B,      B_scale, B_zero, C,       (int)M,
      (int)N, (int)K, 0,       -1,     C_split, splitk_params.SplitK};

  if (sm_version == 0x0800) {
    ampere_hgemm_A16W8_perc_f16_f16_64x128x32_hmma16816_multistage_NT16N_nonfused_splitk_kernel<
        FType, QType, 4><<<grid, block, 0, stream>>>(params);
  } else {
    ampere_hgemm_A16W8_perc_f16_f16_64x128x32_hmma16816_multistage_NT16N_nonfused_splitk_kernel<
        FType, QType, 3><<<grid, block, 0, stream>>>(params);
  }

  // SplitK reduce
  gemm_f16_splitk_reduce<FType, ActiveFunc>(C_split, nullptr, bias, C, M, N,
                                            grid_z, alpha, stream);
}


// Rearrange B to facilitate Ampere Tensor Core load data
// reorder B from (K, N) to (N_32align / 4, K * 4)
// K % 16 == 0, N % 16 == 0, N_32align % 32 == 0
template <typename FType>
__global__ void __launch_bounds__(128)
    rearrange_kn_weight_as_n32k16_order_ldg16_kernel(
        const int8_t* B, const FType* B_scale, const FType* B_zero,
        int8_t* B_result, FType* B_scale_result, FType* B_zero_result,
        const int K, const int N, const int N_32align) {
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;

  if (blockIdx.x != gridDim.x - 1) {
    // Load B
    // per block process 64(k) * 128(n) B elements
    // per warp process 16(k) * 128 B elements
    const int src_row_base_idx =
        blockIdx.x * 64 + warp_id * 16 + ((lane_id % 8) / 2) * 2;
    const int src_col_idx =
        blockIdx.y * 128 + (lane_id / 8) * 32 + (lane_id % 2) * 16;
    char B_frag[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int src_row_idx = src_row_base_idx + (i / 2) * 8 + (i % 2);
      int src_offset = src_row_idx * N + src_col_idx;
      bool guard = src_row_idx < K && src_col_idx < N;
      ldg128_cg_0(*reinterpret_cast<uint32_t*>(B_frag[i]),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 1),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 2),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 3), B + src_offset,
                  guard);
    }

    // reorder B
    char B_reorder_frag[8][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int dst_i = j % 8;
        int dst_j = i + (j / 8) * 4;
        B_reorder_frag[dst_i][dst_j] = B_frag[i][j];
      }
    }

    // Store B
    const int dst_row_base_idx = blockIdx.y * (128 / 4) + (lane_id / 8) * 8;
    const int dst_col_idx =
        blockIdx.x * (64 * 4) + warp_id * 64 + (lane_id % 8) * 8;
    for (int i = 0; i < 8; ++i) {
      int dst_row_idx = dst_row_base_idx + i;
      int dst_offset = dst_row_idx * K * 4 + dst_col_idx;
      bool guard = (dst_row_base_idx < N_32align / 4) && (dst_col_idx < K * 4);
      if (guard) {
        *reinterpret_cast<int2*>(B_result + dst_offset) =
            *reinterpret_cast<int2*>(B_reorder_frag[i]);
      }
    }
  } else {
    // Load B_scale and B_zero
    FType b_scale_reg, b_zero_reg;
    int src_offset = blockIdx.y * 128 + threadIdx.x;
    ldg16_cg_0(b_scale_reg, B_scale + src_offset, src_offset < N);
    ldg16_cg_0(b_zero_reg, B_zero + src_offset, src_offset < N);
    int dst_offset =
        blockIdx.y * 128 + warp_id * 32 + (lane_id % 8) * 4 + lane_id / 8;
    if (dst_offset < N_32align) {
      B_scale_result[dst_offset] = b_scale_reg;
      B_zero_result[dst_offset] = b_zero_reg;
    }
  }
}

template <typename FType>
void rearrange_kn_weight_as_n32k16_order_ldg16(
    const int8_t* B, const FType* B_scale, const FType* B_zero,
    int8_t* B_result, FType* B_scale_result, FType* B_zero_result, const int K,
    const int N, const int N_32align, cudaStream_t stream) {
  if (N % 16 != 0 || K % 16 != 0) {
    std::cerr << "Now only support N and K is multiples of 16" << std::endl;
  }
  const int BLOCK = 128;
  int grid_x = (K + 64 - 1) / 64 + 1;
  int grid_y = (N + 128 - 1) / 128;
  dim3 grid(grid_x, grid_y);

  rearrange_kn_weight_as_n32k16_order_ldg16_kernel<FType>
      <<<grid, BLOCK, 0, stream>>>(B, B_scale, B_zero, B_result, B_scale_result,
                                   B_zero_result, K, N, N_32align);
}

//-------------------
//-------------------
template void get_input_padded_k_align<half>(const half*, half*, const uint32_t,
                                             const uint32_t, const uint32_t,
                                             cudaStream_t);
template void get_input_padded_k_align<hie::bfloat16>(
    const hie::bfloat16*, hie::bfloat16*, const uint32_t, const uint32_t,
    const uint32_t, cudaStream_t);

template void remove_padded_n_align<half>(const half*, half*, const uint32_t,
                                          const uint32_t, const uint32_t,
                                          cudaStream_t);
template void remove_padded_n_align<hie::bfloat16>(
    const hie::bfloat16*, hie::bfloat16*, const uint32_t, const uint32_t,
    const uint32_t, cudaStream_t);

template void dequantize_rhs_a16w8<half, int8_t>(const int8_t*, const half*,
                                                 const half*, half*,
                                                 const uint32_t, const uint32_t,
                                                 const int, cudaStream_t);
template void dequantize_rhs_a16w8<hie::bfloat16, int8_t>(
    const int8_t*, const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*,
    const uint32_t, const uint32_t, const int, cudaStream_t);

template void restore_N32_K16_dequantize_rhs_a16w8<half, int8_t>(
    const int8_t*, const half*, const half*, half*, const int, const int,
    const int, const int, cudaStream_t);

template void restore_N32_K16_dequantize_rhs_a16w8<hie::bfloat16, int8_t>(
    const int8_t*, const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*,
    const int, const int, const int, const int, cudaStream_t);

template void gemv_f16w8_perc_splitk<int8_t, hie::activation::Identity>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void gemv_f16w8_perc_splitk<int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void gemv_f16w8_perc_splitk<int8_t, hie::activation::GeluTanh>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void gemv_f16w8_perc_splitk<int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void gemv_f16w8_perc_splitk<int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8<
    int8_t, hie::activation::Identity>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const uint32_t, const uint32_t,
                                       const uint32_t, void*, const float,
                                       cudaStream_t);

template void
hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Gelu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8<
    int8_t, hie::activation::GeluTanh>(const half*, const int8_t*, const half*,
                                       const half*, const half*, half*,
                                       const uint32_t, const uint32_t,
                                       const uint32_t, void*, const float,
                                       cudaStream_t);

template void
hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Relu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void
hgemm_A16W8_perchannel_128x128x32_hmma1688_ldg8<int8_t, hie::activation::Silu>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const float,
    cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    hie::bfloat16, int8_t, hie::activation::Identity>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, const SplitKParams,
    const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    hie::bfloat16, int8_t, hie::activation::Gelu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, const SplitKParams,
    const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    hie::bfloat16, int8_t, hie::activation::GeluTanh>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, const SplitKParams,
    const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    hie::bfloat16, int8_t, hie::activation::Relu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, const SplitKParams,
    const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    hie::bfloat16, int8_t, hie::activation::Silu>(
    const hie::bfloat16*, const int8_t*, const hie::bfloat16*,
    const hie::bfloat16*, const hie::bfloat16*, hie::bfloat16*, const uint32_t,
    const uint32_t, const uint32_t, void*, const int, const SplitKParams,
    const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    half, int8_t, hie::activation::Identity>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const SplitKParams, const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    half, int8_t, hie::activation::Gelu>(const half*, const int8_t*,
                                         const half*, const half*, const half*,
                                         half*, const uint32_t, const uint32_t,
                                         const uint32_t, void*, const int,
                                         const SplitKParams, const float,
                                         cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    half, int8_t, hie::activation::GeluTanh>(
    const half*, const int8_t*, const half*, const half*, const half*, half*,
    const uint32_t, const uint32_t, const uint32_t, void*, const int,
    const SplitKParams, const float, cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    half, int8_t, hie::activation::Relu>(const half*, const int8_t*,
                                         const half*, const half*, const half*,
                                         half*, const uint32_t, const uint32_t,
                                         const uint32_t, void*, const int,
                                         const SplitKParams, const float,
                                         cudaStream_t);

template void
ampere_hgemm_A16W8_perc_f16_f16_64x128x32_mma16816_multistage_nonfused_splitk<
    half, int8_t, hie::activation::Silu>(const half*, const int8_t*,
                                         const half*, const half*, const half*,
                                         half*, const uint32_t, const uint32_t,
                                         const uint32_t, void*, const int,
                                         const SplitKParams, const float,
                                         cudaStream_t);


template void rearrange_kn_weight_as_n32k16_order_ldg16<half>(
    const int8_t* B, const half* B_scale, const half* B_zero, int8_t* B_result,
    half* B_scale_result, half* B_zero_result, const int K, const int N,
    const int N_32align, cudaStream_t stream);

template void rearrange_kn_weight_as_n32k16_order_ldg16<hie::bfloat16>(
    const int8_t* B, const hie::bfloat16* B_scale, const hie::bfloat16* B_zero,
    int8_t* B_result, hie::bfloat16* B_scale_result,
    hie::bfloat16* B_zero_result, const int K, const int N, const int N_32align,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
