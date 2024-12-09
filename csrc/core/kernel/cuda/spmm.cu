/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    spmm.cu
 */

#ifdef ENABLE_SPARSE
#include "cuda_common.h"
#include "cuda_kernel.h"
#include "spmm_config.h"

template <typename T, int N>
struct alignas(sizeof(T) * N) VT {
  T data[N];

  __device__ __forceinline__ T operator[](int idx) { return data[idx]; }
};

template <typename T, int WIDTH>
__device__ __forceinline__ T WarpReduce(const T& val) {
  T sum = val;
#pragma unroll
  for (int i = WIDTH; i >= 2; i /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i / 2, i);
  }
  return sum;
}
namespace allspark {
namespace cuda {
template <typename Config>
struct SpmmKernel {
  typedef typename Config::InputValType InputValType;
  typedef typename Config::OutputValType OutputValType;
  typedef typename Config::IdxType IdxType;

  typedef VT<InputValType, Config::kPackSize> InputVec;
  typedef VT<IdxType, Config::kPackSize> IdxVec;

  static constexpr int m_block =
      (1 << 16) / Config::kThreadBlockY / sizeof(OutputValType);
  static constexpr int k_block =
      Config::kSparseBlockLen * Config::kPackSize / 4;

  static __device__ __forceinline__ void Pattern0_Fn(
      int m, int n, int k, const InputValType* __restrict__ A,
      const InputValType* __restrict__ B, const int* __restrict__ col_offset,
      const IdxType* __restrict__ row_indices, OutputValType* __restrict__ C) {
    __shared__ InputValType
        inter_res[Config::kThreadBlockY][Config::kThreadBlockX / 32];
    __shared__ OutputValType output_tile[Config::kThreadBlockY];

    int tid_x = Config::kThreadBlockY == 1
                    ? threadIdx.x
                    : threadIdx.x % Config::kThreadBlockX;
    int tid_y =
        Config::kThreadBlockY == 1 ? 0 : threadIdx.x / Config::kThreadBlockX;
    int idx_m = 0;
    int idx_n = blockIdx.x * Config::kThreadBlockY + tid_y;

    if (idx_n >= n) return;

    int col_offset_start = col_offset[idx_n];
    int col_offset_end = col_offset[idx_n + 1];
    int col_nnz = (col_offset_end - col_offset_start) / Config::kPackSize;

    const InputVec* B_col_ptr =
        reinterpret_cast<const InputVec*>(B + col_offset_start);
    const IdxVec* B_row_idx_ptr =
        reinterpret_cast<const IdxVec*>(row_indices + col_offset_start);

    for (; idx_m < m; ++idx_m) {
      const InputValType* A_ptr = A + idx_m * k;

      OutputValType acc = 0;
      for (int k_iter = tid_x; k_iter < col_nnz;
           k_iter += Config::kThreadBlockX) {
        InputVec B_col_vec;
        IdxVec B_row_idx_vec;
        // load columns of B into register
        B_col_vec = *(B_col_ptr + k_iter);
        B_row_idx_vec = *(B_row_idx_ptr + k_iter);

// dot
#pragma unroll
        for (int i = 0; i < Config::kPackSize; ++i) {
          acc += B_col_vec[i] * A_ptr[B_row_idx_vec[i]];
        }
      }

      // warp reduction
      {
        const int REDUCTION_WIDTH =
            Config::kThreadBlockX > 32 ? 32 : Config::kThreadBlockX;
        acc = WarpReduce<OutputValType, REDUCTION_WIDTH>(acc);
      }

      // cross warp reduction
      if (Config::kThreadBlockX > 32) {
        if (tid_x % 32 == 0) {
          inter_res[tid_y][tid_x / 32] = acc;
        }
        __syncthreads();
        if (tid_x < 32) {
          acc = tid_x < (Config::kThreadBlockX / 32)
                    ? inter_res[tid_y][tid_x]
                    : static_cast<OutputValType>(0);
          const int REDUCTION_WIDTH = Config::kThreadBlockX / 32;
          acc = WarpReduce<OutputValType, REDUCTION_WIDTH>(acc);
        }
      }

      // write back
      OutputValType* out_ptr =
          C + idx_m * n + blockIdx.x * Config::kThreadBlockY;
      if (Config::kThreadBlockY == 1) {
        if (tid_x == 0) {
          *out_ptr = acc;
        }
      } else {
        if (tid_x == 0) {
          output_tile[tid_y] = acc;
        }
        __syncthreads();
        if (threadIdx.x < Config::kThreadBlockY) {
          out_ptr[threadIdx.x] += output_tile[threadIdx.x];
        }
      }
      __syncthreads();
    }
  }
  static __device__ __forceinline__ void Pattern1_Fn(
      int m, int n, int k, int nnzPerCol, const InputValType* __restrict__ A,
      const InputValType* __restrict__ B,
      const IdxType* __restrict__ row_indices, OutputValType* __restrict__ C) {
    InputVec B_reg[k_block / Config::kPackSize];
    IdxVec B_idx_reg[k_block / Config::kPackSize];
    int k_step_num = nnzPerCol / k_block;
    int k_tail = nnzPerCol - k_step_num * k_block;

    int idx_n = blockIdx.x * Config::kThreadBlockY + threadIdx.x;

    if (idx_n < n) {
      const InputVec* B_ptr = reinterpret_cast<const InputVec*>(B);
      const IdxVec* idx_ptr = reinterpret_cast<const IdxVec*>(row_indices);

      // deal with tail
      if (k_tail > 0) {
        // load B into reg
        for (int i = 0; i < k_tail / Config::kPackSize; ++i) {
          B_reg[i] = B_ptr[idx_n];
          B_ptr += n;
          B_idx_reg[i] = idx_ptr[idx_n];
          idx_ptr += n;
        }

        const InputValType* A_ptr = A;
        OutputValType* C_ptr = C;
        for (int idx_m = 0; idx_m < m; ++idx_m) {
          // compute and write back C
          OutputValType acc = 0;
          for (int i = 0; i < k_tail / Config::kPackSize; ++i) {
#pragma unroll
            for (int j = 0; j < Config::kPackSize; ++j) {
              acc += B_reg[i][j] * A_ptr[B_idx_reg[i][j]];
            }
          }
          /*************/
          // C_ptr[idx_n] = acc;
          C_ptr[idx_n] += acc;
          /*************/
          A_ptr += k;
          C_ptr += n;
        }
      }

      for (int k_step = 0; k_step < k_step_num; ++k_step) {
        // load B into reg
        for (int i = 0; i < k_block / Config::kPackSize; ++i) {
          B_reg[i] = B_ptr[idx_n];
          B_ptr += n;
          B_idx_reg[i] = idx_ptr[idx_n];
          idx_ptr += n;
        }

        const InputValType* A_ptr = A;
        OutputValType* C_ptr = C;
        for (int idx_m = 0; idx_m < m; ++idx_m) {
          // compute and write back C
          OutputValType acc = 0;
          for (int i = 0; i < k_block / Config::kPackSize; ++i) {
#pragma unroll
            for (int j = 0; j < Config::kPackSize; ++j) {
              acc += B_reg[i][j] * A_ptr[B_idx_reg[i][j]];
            }
          }
          C_ptr[idx_n] += acc;
          A_ptr += k;
          C_ptr += n;
        }
      }
    }
  }
  static __device__ __forceinline__ void Pattern1_Block_Fn(
      int m, int n, int k, int nnzPerCol, const InputValType* __restrict__ A,
      const InputValType* __restrict__ B,
      const IdxType* __restrict__ row_indices, OutputValType* __restrict__ C) {
    InputVec B_reg[k_block / Config::kPackSize];
    IdxVec B_idx_reg[k_block / Config::kPackSize];
    extern __shared__ char dynamic_smem[];
    OutputValType* out_block = reinterpret_cast<OutputValType*>(dynamic_smem);

    int idx_m = 0;
    int idx_n = blockIdx.x * Config::kThreadBlockY + threadIdx.x;
    int m_step_num = m / m_block;
    int m_tail = m - m_step_num * m_block;
    int k_step_num = nnzPerCol / k_block;
    int k_tail = nnzPerCol - k_step_num * k_block;
    OutputValType* C_ptr = C;
    for (int m_step = 0; m_step < m_step_num; m_step += 1, idx_m += m_block) {
      // init outBlock
      for (int i = 0; i < m_block; ++i) {
        out_block[i * Config::kThreadBlockY + threadIdx.x] = 0;
      }
      __syncthreads();

      if (idx_n < n) {
        // compute outBlock along k-axis
        const InputVec* B_ptr = reinterpret_cast<const InputVec*>(B);
        const IdxVec* idx_ptr = reinterpret_cast<const IdxVec*>(row_indices);
        for (int k_step = 0; k_step < k_step_num; k_step += 1) {
          // load B block into reg
          for (int i = 0; i < k_block / Config::kPackSize; ++i) {
            B_reg[i] = B_ptr[idx_n];
            B_ptr += n;
            B_idx_reg[i] = idx_ptr[idx_n];
            idx_ptr += n;
          }

          const InputValType* A_ptr = A + idx_m * k;
          for (int x = 0; x < m_block; ++x) {
            for (int y = 0; y < k_block / Config::kPackSize; ++y) {
#pragma unroll
              for (int z = 0; z < Config::kPackSize; ++z) {
                out_block[x * Config::kThreadBlockY + threadIdx.x] +=
                    B_reg[y][z] * A_ptr[B_idx_reg[y][z]];
              }
            }
            A_ptr += k;
          }
        }
        if (k_tail > 0) {
          for (int i = 0; i < k_tail / Config::kPackSize; ++i) {
            B_reg[i] = B_ptr[idx_n];
            B_ptr += n;
            B_idx_reg[i] = idx_ptr[idx_n];
            idx_ptr += n;
          }

          const InputValType* A_ptr = A + idx_m * k;
          for (int x = 0; x < m_block; ++x) {
            for (int y = 0; y < k_tail / Config::kPackSize; ++y) {
#pragma unroll
              for (int z = 0; z < Config::kPackSize; ++z) {
                out_block[x * Config::kThreadBlockY + threadIdx.x] +=
                    B_reg[y][z] * A_ptr[B_idx_reg[y][z]];
              }
            }
            A_ptr += k;
          }
        }
      }

      // write back
      for (int i = 0; i < m_block; ++i) {
        if (idx_n < n) {
          /*************/
          // C_ptr[idx_n] =
          //     out_block[i * Config::kThreadBlockY + threadIdx.x];
          C_ptr[idx_n] += out_block[i * Config::kThreadBlockY + threadIdx.x];
          /*************/
          C_ptr += n;
        }
      }
      __syncthreads();
    }

    // deal with tail
    if (m_tail > 0) {
      // init outBlock
      for (int i = 0; i < m_tail; ++i) {
        out_block[i * Config::kThreadBlockY + threadIdx.x] = 0;
      }
      __syncthreads();

      if (idx_n < n) {
        // compute outBlock along k-axis
        const InputVec* B_ptr = reinterpret_cast<const InputVec*>(B);
        const IdxVec* idx_ptr = reinterpret_cast<const IdxVec*>(row_indices);
        for (int k_step = 0; k_step < k_step_num; k_step += 1) {
          // load B block into reg
          for (int i = 0; i < k_block / Config::kPackSize; ++i) {
            B_reg[i] = B_ptr[idx_n];
            B_ptr += n;
            B_idx_reg[i] = idx_ptr[idx_n];
            idx_ptr += n;
          }

          const InputValType* A_ptr = A + idx_m * k;
          for (int x = 0; x < m_tail; ++x) {
            for (int y = 0; y < k_block / Config::kPackSize; ++y) {
#pragma unroll
              for (int z = 0; z < Config::kPackSize; ++z) {
                out_block[x * Config::kThreadBlockY + threadIdx.x] +=
                    B_reg[y][z] * A_ptr[B_idx_reg[y][z]];
              }
            }
            A_ptr += k;
          }
        }
        if (k_tail > 0) {
          for (int i = 0; i < k_tail / Config::kPackSize; ++i) {
            B_reg[i] = B_ptr[idx_n];
            B_ptr += n;
            B_idx_reg[i] = idx_ptr[idx_n];
            idx_ptr += n;
          }

          const InputValType* A_ptr = A + idx_m * k;
          for (int x = 0; x < m_tail; ++x) {
            for (int y = 0; y < k_tail / Config::kPackSize; ++y) {
#pragma unroll
              for (int z = 0; z < Config::kPackSize; ++z) {
                out_block[x * Config::kThreadBlockY + threadIdx.x] +=
                    B_reg[y][z] * A_ptr[B_idx_reg[y][z]];
              }
            }
            A_ptr += k;
          }
        }
      }

      // write back
      for (int i = 0; i < m_tail; ++i) {
        if (idx_n < n) {
          /*************/
          // C_ptr[idx_n] =
          //     out_block[i * Config::kThreadBlockY + threadIdx.x];
          C_ptr[idx_n] += out_block[i * Config::kThreadBlockY + threadIdx.x];
          /*************/
          C_ptr += n;
        }
      }
      __syncthreads();
    }
  }
};

template <typename Config>
__global__ void __launch_bounds__(Config::kThreadBlockSize)
    pattern0_kernel(int m, int n, int k,
                    const typename Config::InputValType* __restrict__ A,
                    const typename Config::InputValType* __restrict__ B,
                    const int* __restrict__ col_offset,
                    const typename Config::IdxType* __restrict__ row_indices,
                    typename Config::OutputValType* __restrict__ C) {
  SpmmKernel<Config>::Pattern0_Fn(m, n, k, A, B, col_offset, row_indices, C);
}
template <>
void spmm_pattern0<float>(const float* A, const float* B,
                          const int* cscColOffset, const int* cscRowInd,
                          float* C, const int M, const int N, const int K,
                          const int nnz, cudaStream_t stream) {
  // @TODO: need more classified discussion here
  typedef SpmmConfig<float, float, int, 32, 8> Config;
  pattern0_kernel<Config><<<(N + 7) / 8, 256, 0, stream>>>(
      M, N, K, A, B, cscColOffset, cscRowInd, C);
  return;
}
#ifdef ENABLE_FP16
template <>
void spmm_pattern0<half>(const half* A, const half* B, const int* cscColOffset,
                         const int* cscRowInd, half* C, const int M,
                         const int N, const int K, const int nnz,
                         cudaStream_t stream) {
  // @TODO: need more classified discussion here
  typedef SpmmConfig<half, half, int, 32, 8> Config;
  pattern0_kernel<Config><<<(N + 7) / 8, 256, 0, stream>>>(
      M, N, K, A, B, cscColOffset, cscRowInd, C);
  return;
}
#endif
template <>
void spmm_pattern0<hie::bfloat16>(const hie::bfloat16* A,
                                  const hie::bfloat16* B,
                                  const int* cscColOffset, const int* cscRowInd,
                                  hie::bfloat16* C, const int M, const int N,
                                  const int K, const int nnz,
                                  cudaStream_t stream) {
  // @TODO: need more classified discussion here
  typedef SpmmConfig<hie::bfloat16, hie::bfloat16, int, 32, 8> Config;
  pattern0_kernel<Config><<<(N + 7) / 8, 256, 0, stream>>>(
      M, N, K, A, B, cscColOffset, cscRowInd, C);
  return;
}
template <typename Config>
__global__ void __launch_bounds__(Config::kThreadBlockSize)
    pattern1_kernel(int m, int n, int k, int nnzPerCol,
                    const typename Config::InputValType* __restrict__ A,
                    const typename Config::InputValType* __restrict__ B,
                    const typename Config::IdxType* __restrict__ row_indices,
                    typename Config::OutputValType* __restrict__ C) {
  SpmmKernel<Config>::Pattern1_Fn(m, n, k, nnzPerCol, A, B, row_indices, C);
}
template <typename Config>
__global__ void __launch_bounds__(Config::kThreadBlockSize)
    pattern1_block_kernel(
        int m, int n, int k, int nnzPerCol,
        const typename Config::InputValType* __restrict__ A,
        const typename Config::InputValType* __restrict__ B,
        const typename Config::IdxType* __restrict__ row_indices,
        typename Config::OutputValType* __restrict__ C) {
  SpmmKernel<Config>::Pattern1_Block_Fn(m, n, k, nnzPerCol, A, B, row_indices,
                                        C);
}
template <>
void spmm_pattern1<float>(const float* A, const float* B,
                          const unsigned short* rowInd, float* C, const int M,
                          const int N, const int K, const int nnz,
                          cudaStream_t stream) {
  // @TODO: need more classified discussion here
  typedef SpmmConfig<float, float, unsigned short, 1, 256, 96> Config;
  int nnzPerCol = nnz / N;
  // mB = 64 / sizeof(out_type) * 4
  // kB = 96 / max(sizeof(in_type), sizeof(idx_type)) * 4
  int mB = 64;
  int kB = 96;
  bool flag = (nnzPerCol * ((M - 1) / mB)) < (2 * M * ((nnzPerCol - 1) / kB));
  if (flag) {
    cudaFuncSetAttribute(pattern1_block_kernel<Config>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    pattern1_block_kernel<Config><<<(N + 255) / 256, 256, 65536, stream>>>(
        M, N, K, nnzPerCol, A, B, rowInd, C);
  } else {
    pattern1_kernel<Config><<<(N + 255) / 256, 256, 0, stream>>>(
        M, N, K, nnzPerCol, A, B, rowInd, C);
  }
  return;
}
#ifdef ENABLE_FP16
template <>
void spmm_pattern1<half>(const half* A, const half* B,
                         const unsigned short* rowInd, half* C, const int M,
                         const int N, const int K, const int nnz,
                         cudaStream_t stream) {
  // @TODO: need more classified discussion here
  typedef SpmmConfig<half, half, unsigned short, 1, 256, 96> Config;
  int nnzPerCol = nnz / N;
  // mB = 64 / sizeof(out_type) * 4
  // kB = 96 / max(sizeof(in_type), sizeof(idx_type)) * 4
  int mB = 128;
  int kB = 96;
  bool flag = (nnzPerCol * ((M - 1) / mB)) < (2 * M * ((nnzPerCol - 1) / kB));
  if (flag) {
    cudaFuncSetAttribute(pattern1_block_kernel<Config>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    pattern1_block_kernel<Config><<<(N + 255) / 256, 256, 65536, stream>>>(
        M, N, K, nnzPerCol, A, B, rowInd, C);
  } else {
    pattern1_kernel<Config><<<(N + 255) / 256, 256, 0, stream>>>(
        M, N, K, nnzPerCol, A, B, rowInd, C);
  }
  return;
}
#endif
template <>
void spmm_pattern1<hie::bfloat16>(const hie::bfloat16* A,
                                  const hie::bfloat16* B,
                                  const unsigned short* rowInd,
                                  hie::bfloat16* C, const int M, const int N,
                                  const int K, const int nnz,
                                  cudaStream_t stream) {
  // @TODO: need more classified discussion here
  typedef SpmmConfig<hie::bfloat16, hie::bfloat16, unsigned short, 1, 256, 96>
      Config;
  int nnzPerCol = nnz / N;
  // mB = 64 / sizeof(out_type) * 4
  // kB = 96 / max(sizeof(in_type), sizeof(idx_type)) * 4
  int mB = 128;
  int kB = 96;
  bool flag = (nnzPerCol * ((M - 1) / mB)) < (2 * M * ((nnzPerCol - 1) / kB));
  if (flag) {
    cudaFuncSetAttribute(pattern1_block_kernel<Config>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    pattern1_block_kernel<Config><<<(N + 255) / 256, 256, 65536, stream>>>(
        M, N, K, nnzPerCol, A, B, rowInd, C);
  } else {
    pattern1_kernel<Config><<<(N + 255) / 256, 256, 0, stream>>>(
        M, N, K, nnzPerCol, A, B, rowInd, C);
  }
  return;
}
}  // namespace cuda
}  // namespace allspark
#endif