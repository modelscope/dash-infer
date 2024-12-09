/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a8w8_pert_kernel.cu
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef ENABLE_FP8
#include "cutlass/float8.h"
#endif
#include "gemm_a16w8_kernel.h"
#include "gemm_lowp_utils.cuh"
using namespace std;

namespace allspark {
namespace cuda {

static constexpr int WARP_SIZE = 32;

/**
 * @brief Warp Reduce and Block Reduce
 */

template <typename T>
struct MaxOp {
 public:
  static constexpr T init = std::numeric_limits<T>::min();
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
 public:
  static constexpr float init = -std::numeric_limits<float>::infinity();
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x > y ? x : y;
  }
};

template <template <class> class Func, typename T>
__device__ __forceinline__ T ReduceWarp(T val) {
#pragma unroll
  for (int i = WARP_SIZE; i > 1; i /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, val, i / 2);
    val = Func<T>::op(tmp, val);
  }
  return val;
}

template <template <class> class Func, typename T, int BLOCK>
__device__ __forceinline__ T ReduceBlock(const T& val) {
  static_assert(BLOCK >= WARP_SIZE, "Invalid Block size");
  __shared__ T smem[BLOCK / WARP_SIZE];
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  T val_reg = Func<T>::init;
  val_reg = Func<T>::op(val_reg, val);
  val_reg = ReduceWarp<Func, T>(val_reg);
  if (BLOCK / WARP_SIZE > 1) {
    if (lane_id == 0) {
      smem[warp_id] = val_reg;
    }
    __syncthreads();
    if (warp_id == 0) {
      val_reg = lane_id < BLOCK / WARP_SIZE ? smem[lane_id] : Func<T>::init;
      val_reg = ReduceWarp<Func, T>(val_reg);
    }
  }
  __syncthreads();
  return val_reg;
}

// PerTensor + Symmetric : compute quantization params
// size % 8 = 0
// FType is Half or Bfloat16, so PACK is 8
template <typename FType, typename DType, int BLOCK, int UNROLL>
__global__
__launch_bounds__(BLOCK) void per_tensor_symm_quantization_params_reduce_kernel(
    const FType* fdata, float* scale, const int size) {
  const int tid = threadIdx.x;
  const int base_offset = blockIdx.x * BLOCK * 8 * UNROLL + tid * 8;

  const DType DType_max = std::numeric_limits<DType>::max();
  float fmax = 0;
  typename HalfType<FType>::T2 ld_reg[UNROLL][4] = {};

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    ldg128_cg_0(ld_reg[i][0], ld_reg[i][1], ld_reg[i][2], ld_reg[i][3],
                fdata + base_offset + i * BLOCK * 8,
                (base_offset + i * BLOCK * 8) < size);
  }
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      fmax = max(fabsf(static_cast<float>(ld_reg[i][j].x)), fmax);
      fmax = max(fabsf(static_cast<float>(ld_reg[i][j].y)), fmax);
    }
  }
  fmax = ReduceBlock<MaxOp, float, BLOCK>(fmax);
  if (tid == 0) {
    fmax = fmax / DType_max;
    // here fmax is non-negative, so use of reinterpret_cast can realize
    // reduce_max in float Note : init value of scale must be 0
    atomicMax(reinterpret_cast<int*>(scale),
              reinterpret_cast<const int&>(fmax));
  }
}

template <typename Source, typename Target>
struct NumericConverter {
  static __device__ __forceinline__ Target Convert(Source const& s) {
    return static_cast<Target>(float(s));
  }
};

template <typename Source>
struct NumericConverter<Source, int8_t> {
  static __device__ __forceinline__ int8_t Convert(Source const& s) {
    return static_cast<int8_t>(roundf(float(s)));
  }
};

template <typename FType, typename DType, typename ComputeType = float>
struct Quantize {
  using NumConverter = NumericConverter<ComputeType, DType>;

  static __device__ __forceinline__ DType Quant(const FType& in_data,
                                                const ComputeType& scale_rec,
                                                const ComputeType& zero_point,
                                                const DType& max,
                                                const DType& min) {
    ComputeType imd = fmaxf(
        min,
        fminf(max, static_cast<ComputeType>(in_data) * scale_rec + zero_point));
    return NumConverter::Convert(imd);
  }
};

template <typename FType, typename DType, int BLOCK, int UNROLL,
          typename ComputeType = float>
__global__ __launch_bounds__(BLOCK) void per_tensor_symm_quantization_kernel(
    const FType* fdata, const float* scale, DType* qdata, const int size) {
  const int tid = threadIdx.x;
  const int base_offset = blockIdx.x * BLOCK * UNROLL * 8 + tid * 8;

  typename HalfType<FType>::T2 ld_reg[UNROLL][4] = {};
  DType st_reg[UNROLL][8];

  // load fdata
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    ldg128_cg(ld_reg[i][0], ld_reg[i][1], ld_reg[i][2], ld_reg[i][3],
              fdata + base_offset + i * BLOCK * 8,
              (base_offset + i * BLOCK * 8) < size);
  }

  ComputeType scale_rec = 1.f / scale[0];
  const DType DType_max = std::numeric_limits<DType>::max();
  const DType DType_min = -DType_max;
// quantize
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      st_reg[i][j * 2] = Quantize<typename HalfType<FType>::T1, DType>::Quant(
          ld_reg[i][j].x, scale_rec, 0.0f, DType_max, DType_min);
      st_reg[i][j * 2 + 1] =
          Quantize<typename HalfType<FType>::T1, DType>::Quant(
              ld_reg[i][j].y, scale_rec, 0.0f, DType_max, DType_min);
    }
  }

// store qdata
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int offset = base_offset + i * BLOCK * 8;
    if (offset < size) {
      *(reinterpret_cast<int2*>(qdata + offset)) =
          *reinterpret_cast<int2*>(st_reg[i]);
    }
  }
}

// FType : half or bfloat16
// DType : int8_t or float8_e4m3 or float8_e5m2
template <typename FType, typename DType = int8_t>
void per_tensor_symm_quantization(const FType* fdata, float* scale,
                                  DType* qdata, const int M, const int K,
                                  cudaStream_t stream) {
  cudaMemsetAsync(scale, 0, sizeof(float), stream);
  const int size = M * K;
  if (size % 8 != 0) {
    std::cerr << "Now this kernel only support size % 8 == 0!" << std::endl;
  }
  // step1 : get quantization params by reducing
  {
    const int BLOCK = 256;
    const int UNROLL = 8;
    const int GRID = (size + BLOCK * UNROLL * 8 - 1) / (BLOCK * UNROLL * 8);
    per_tensor_symm_quantization_params_reduce_kernel<FType, DType, BLOCK,
                                                      UNROLL>
        <<<GRID, BLOCK, 0, stream>>>(fdata, scale, size);
  }
  // step2: quantization
  {
    const int BLOCK = 128;
    const int UNROLL = 1;
    const int GRID = (size + BLOCK * UNROLL * 8 - 1) / (BLOCK * UNROLL * 8);
    per_tensor_symm_quantization_kernel<FType, DType, BLOCK, UNROLL>
        <<<GRID, BLOCK, 0, stream>>>(fdata, scale, qdata, size);
  }
}

// A : PerTensor + Symmetric Quantization
// B : PerChannel + Symmetric Quantization
// Dequant the int32-type immediate result obtained by A and B matrix
// multiplication FType is Half or BFloat16 N % 4 == 0
template <typename FType, int BLOCK, int UNROLL, typename ComputeType = float>
__global__ void A_pert_symm_B_perc_symm_dequantization_kernel(
    const int* imd_result, const float* A_scale, const FType* B_scale,
    const FType* bias, FType* result, const int M, const int N) {
  int ridx = blockIdx.y * UNROLL;
  int cidx = blockIdx.x * BLOCK * 4 + threadIdx.x * 4;
  int base_offset = ridx * N + cidx;

  int ld_reg[UNROLL][4] = {};
  FType b_scale_reg[4] = {};
  FType bias_reg[4] = {};
  FType st_reg[UNROLL][4];

  ldg64_ca(*reinterpret_cast<uint32_t*>(b_scale_reg),
           *reinterpret_cast<uint32_t*>(b_scale_reg + 2), B_scale + cidx,
           cidx < N);
  if (bias) {
    ldg64_ca(*reinterpret_cast<uint32_t*>(bias_reg),
             *reinterpret_cast<uint32_t*>(bias_reg + 2), bias + cidx, cidx < N);
  }

#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    int offset = base_offset + i * N;
    ldg128_cg(ld_reg[i][0], ld_reg[i][1], ld_reg[i][2], ld_reg[i][3],
              imd_result + offset, cidx < N && (ridx + i) < M);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ComputeType fval =
          ld_reg[i][j] * A_scale[0] * static_cast<ComputeType>(b_scale_reg[j]);
      if (bias) {
        fval += static_cast<ComputeType>(bias_reg[j]);
      }
      st_reg[i][j] = static_cast<FType>(fval);
    }
    if ((ridx + i) < M && cidx < N) {
      *reinterpret_cast<int2*>(result + offset) =
          *reinterpret_cast<int2*>(st_reg[i]);
    }
  }
}

template <typename FType>
void A_pert_symm_B_perc_symm_dequantization(const int* imd_result,
                                            const float* A_scale,
                                            const FType* B_scale,
                                            const FType* bias, FType* result,
                                            const int M, const int N,
                                            cudaStream_t stream) {
  const int BLOCK = 256;
  const int UNROLL = 4;
  const int PACK = 4;
  // const int size = M * N;

  const int grid_x = (N + BLOCK * PACK - 1) / (BLOCK * PACK);
  const int grid_y = (M + UNROLL - 1) / UNROLL;
  dim3 grid(grid_x, grid_y);

  if (N % PACK != 0) {
    std::cerr << "Now this kernel only support N % " << PACK << " == 0!"
              << std::endl;
  }

  A_pert_symm_B_perc_symm_dequantization_kernel<FType, BLOCK, UNROLL>
      <<<grid, BLOCK, 0, stream>>>(imd_result, A_scale, B_scale, bias, result,
                                   M, N);
}

//-------------------
//-------------------
template void A_pert_symm_B_perc_symm_dequantization<half>(
    const int* imd_result, const float* A_scale, const half* B_scale,
    const half* bias, half* result, const int M, const int N,
    cudaStream_t stream);
template void A_pert_symm_B_perc_symm_dequantization<hie::bfloat16>(
    const int* imd_result, const float* A_scale, const hie::bfloat16* B_scale,
    const hie::bfloat16* bias, hie::bfloat16* result, const int M, const int N,
    cudaStream_t stream);
template void per_tensor_symm_quantization<half, int8_t>(
    const half* fdata, float* scale, int8_t* qdata, const int M, const int K,
    cudaStream_t stream);
template void per_tensor_symm_quantization<hie::bfloat16, int8_t>(
    const hie::bfloat16* fdata, float* scale, int8_t* qdata, const int M,
    const int K, cudaStream_t stream);

#ifdef ENABLE_FP8
template void per_tensor_symm_quantization<half, cutlass::float_e4m3_t>(
    const half* fdata, float* scale, cutlass::float_e4m3_t* qdata, const int M,
    const int K, cudaStream_t stream);
template void
per_tensor_symm_quantization<hie::bfloat16, cutlass::float_e4m3_t>(
    const hie::bfloat16* fdata, float* scale, cutlass::float_e4m3_t* qdata,
    const int M, const int K, cudaStream_t stream);
template void per_tensor_symm_quantization<half, cutlass::float_e5m2_t>(
    const half* fdata, float* scale, cutlass::float_e5m2_t* qdata, const int M,
    const int K, cudaStream_t stream);
template void
per_tensor_symm_quantization<hie::bfloat16, cutlass::float_e5m2_t>(
    const hie::bfloat16* fdata, float* scale, cutlass::float_e5m2_t* qdata,
    const int M, const int K, cudaStream_t stream);
#endif
}  // namespace cuda
}  // namespace allspark
