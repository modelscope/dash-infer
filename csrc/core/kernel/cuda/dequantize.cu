/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dequantize.cu
 */

#include <cmath>
#include <limits>
#include <string>

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#define CUDA_HOST_DEVICE __forceinline__ __device__ __host__
#define CUDA_DEVICE __forceinline__ __device__
#define MAX_DIMS 6
namespace allspark {
namespace cuda {

/**
 * @brief quantize kernel
 *
 */

template <typename T, int N>
struct alignas(N * sizeof(T)) VT {
  T data[N];
};

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  val += __shfl_xor_sync(0xffffffff, val, 16);
  val += __shfl_xor_sync(0xffffffff, val, 8);
  val += __shfl_xor_sync(0xffffffff, val, 4);
  val += __shfl_xor_sync(0xffffffff, val, 2);
  val += __shfl_xor_sync(0xffffffff, val, 1);
  return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
  val = cmath_fmax(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = cmath_fmax(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = cmath_fmax(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = cmath_fmax(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = cmath_fmax(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
  val = cmath_fmin(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = cmath_fmin(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = cmath_fmin(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = cmath_fmin(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = cmath_fmin(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}
template <typename FType, typename QType>
__device__ __forceinline__ void do_comput_params(FType fmax, FType fmin,
                                                 FType qmax, FType qmin,
                                                 FType& scale,
                                                 QType& zero_point) {
  fmin = cmath_fmin((FType)0.0f, fmin);
  fmax = cmath_fmax((FType)0.0f, fmax);
  scale = (fmax - fmin) / (qmax - qmin);
  const FType init_zero_point = qmin - fmin / scale;
  // if (init_zero_point < qmin) {
  //     zero_point = static_cast<QType>(qmin);
  // }
  // else if (init_zero_point > qmax) {
  //     zero_point = static_cast<QType>(qmax);
  // }
  // else {
  //     zero_point = static_cast<QType>(roundf(init_zero_point));
  // }
  zero_point = static_cast<QType>(
      roundf(cmath_fmin(qmax, cmath_fmax(init_zero_point, qmin))));
}

template <typename FType>
__device__ __forceinline__ void do_comput_params(FType fmax, FType fmin,
                                                 FType qmax, FType qmin,
                                                 FType& scale,
                                                 FType& zero_point) {
  fmin = cmath_fmin((FType)0.0f, fmin);
  fmax = cmath_fmax((FType)0.0f, fmax);
  scale = (fmax - fmin) / (qmax - qmin);
  const FType init_zero_point = qmin - fmin / scale;
  zero_point = cmath_fmin(qmax, cmath_fmax(init_zero_point, qmin));
}

// flip float to uint32. Because CUDA don`t support some float atomic operation
__device__ __host__ __forceinline__ uint32_t
float_flip_uint32(const float f_data) {
  uint32_t f_i = reinterpret_cast<const uint32_t&>(f_data);
  uint32_t mask = -static_cast<int>(f_i >> 31) | 0x80000000;
  return f_i ^ mask;
}

__device__ __host__ __forceinline__ float uint32_flip_float(
    const uint32_t i_data) {
  uint32_t mask = ((i_data >> 31) - 1) | 0x80000000;
  uint32_t f_i = i_data ^ mask;
  float ret_f = reinterpret_cast<const float&>(f_i);
  return ret_f;
}

// load data from L2 or global memory.
__device__ __forceinline__ void ld_cg(uint32_t& ret, const void* ptr) {
  asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(ret) : "l"(ptr));
}
/*
 * brief:
 * for tensor with @n elements and thread block size = block:
 *     threads with tid < @nPack access memory vectorized,
 *     threads with @nPack <= tid < @nThreads access memory scalarized
 *     @nBlock is grid size, or the number of thread blocks
 */
// ----------------------------------------------------
// integer round up division
// ----------------------------------------------------
template <typename T>
inline T SIntDivUp(T x, T y) {
  T ret;

  if (x == 0) {
    ret = 0;
  } else if (x * y < 0) {
    ret = (x - y - (x > 0 ? 1 : -1)) / y;
  } else {
    ret = (x + y - (x > 0 ? 1 : -1)) / y;
  }

  return ret;
}

template <typename T>
inline T UIntDivUp(T x, T y) {
  return (x + y - 1) / y;
}

struct PackedEltwiseConfig {
  int64_t nPack;
  int64_t nThread;
  int64_t nBlock;

  /*
   * @n: number of elements
   * @packSize: packed size in elements
   * @block: size of thread block
   */
  PackedEltwiseConfig(){};
  PackedEltwiseConfig(const int64_t n, const int64_t COL_UNROLL,
                      const int64_t block, const int64_t ROW_UNROLL = 1) {
    nPack = n / (COL_UNROLL * ROW_UNROLL);
    nThread = nPack + n % (COL_UNROLL * ROW_UNROLL);
    nBlock = SIntDivUp(nThread, block);
  }

  void update(const int64_t n, const int64_t COL_UNROLL, const int64_t block,
              const int64_t ROW_UNROLL) {
    nPack = n / (COL_UNROLL * ROW_UNROLL);
    nThread = nPack + n % (COL_UNROLL * ROW_UNROLL);
    nBlock = SIntDivUp(nThread, block);
  }
};

// Get vector size (<= 4) based on alignment
template <typename T>
inline int GetPackSize(const T* ptr) {
  // only pack for T (sizeof(T) < MIN_NONPACKED_ITEM_SIZE)
  const int MIN_NONPACKED_ITEM_SIZE = 16;
  // max size in byte of the vector (packed elements)
  const int MAX_PACKED_BYTE = 16;
  // max number of packed elements
  const int MAX_PACKED_SIZE = 8;

  if (sizeof(T) > MIN_NONPACKED_ITEM_SIZE) {
    return 1;
  }

  int packSize = std::min(MAX_PACKED_BYTE / sizeof(T), size_t(MAX_PACKED_SIZE));
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
  while (packSize > 0) {
    if (addr % (sizeof(T) * packSize) == 0) {
      break;
    }
    packSize /= 2;
  }
  return packSize;
}

/**
 * @brief divmod
 *
 */
template <typename T>
struct DivMod {
  T div;
  T mod;
  __host__ __device__ DivMod(T d, T m) : div(d), mod(m) {}
};
template <typename T>
struct IntDivModer {
  uint32_t d_;

  IntDivModer() {}
  explicit IntDivModer(T d) : d_(d) {}

  __host__ __device__ __forceinline__ T div(T n) { return n / d_; }

  __host__ __device__ __forceinline__ T mod(T n) { return n % d_; }

  __host__ __device__ __forceinline__ DivMod<T> divmod(T n) {
    return DivMod<T>(n / d_, n % d_);
  }
};
template <>
struct IntDivModer<uint32_t> {
  uint32_t d_;
  uint32_t magic_;
  uint32_t shift_;

  IntDivModer() {}

  explicit IntDivModer(uint32_t d) {
    d_ = d;

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) break;
    }
    // xunle : in MSVC sizeof(long) is 4 bytes , which causes overflow
    uint64_t tmp_magic = ((1llu << 32) * ((1llu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit
  }

  __host__ __device__ __forceinline__ uint32_t div(uint32_t n) {
#if defined(__CUDA_ARCH__)
    return (__umulhi(n, magic_) + n) >> shift_;
#else
    uint32_t t = (static_cast<uint64_t>(n) * magic_) >> 32;
    return (t + n) >> shift_;
#endif  // defined(__CUDA_ARCH__)
  }

  __host__ __device__ __forceinline__ uint32_t mod(uint32_t n) {
    return n - div(n) * d_;
  }

  __host__ __device__ __forceinline__ DivMod<uint32_t> divmod(uint32_t n) {
    uint32_t d = div(n);
    return DivMod<uint32_t>(d, n - d_ * d);
  }
};
/**
 * @brief post process
 *        fp32_res = active( alpha * lhs_scale * rhs_scale *
 * static_cast<float>(int32_res + offset)
 * + beta * bias )
 *
 *  FType is float-point data type, float32 or float16
 *  QType is quantization data type, int8 or uint8. GPU use int8
 *
 * */

// Post Process without bias. Do ROW_UNROLL for Rhs.
template <int BLOCK, int ROW_UNROLL, int NUM_DIMS, typename FType,
          typename QType, typename Active>
__global__ __launch_bounds__(BLOCK) void postProcessKernel(
    hie::Array<uint32_t, MAX_DIMS> lhs_stride,
    const float* __restrict__ lhs_scale, const QType* __restrict__ lhs_zero,
    const int32_t* __restrict__ lhs_redsum,
    hie::Array<uint32_t, MAX_DIMS> rhs_stride,
    const float* __restrict__ rhs_scale, const QType* __restrict__ rhs_zero,
    const int32_t* __restrict__ rhs_redsum,
    hie::Array<IntDivModer<uint32_t>, MAX_DIMS> out_divmod,
    const int32_t* i32_input, FType* f32_output, const int K, const FType alpha,
    Active active_func, PackedEltwiseConfig packConfig) {
  int tid = blockIdx.x * BLOCK + threadIdx.x;
  if (tid < packConfig.nPack) {
    uint32_t lhs_offset[ROW_UNROLL];
    uint32_t rhs_offset[ROW_UNROLL];
    uint32_t offset[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      lhs_offset[i] = 0;
      rhs_offset[i] = 0;
      offset[i] = tid + packConfig.nPack * i;
    }

#pragma unroll
    for (int i = 0; i < NUM_DIMS; ++i) {
#pragma unroll
      for (int j = 0; j < ROW_UNROLL; ++j) {
        DivMod<uint32_t> d = out_divmod[i].divmod(offset[j]);
        uint32_t dmod = d.mod;
        offset[j] = d.div;
        lhs_offset[j] += dmod * lhs_stride[i];
        rhs_offset[j] += dmod * rhs_stride[i];
      }
    }

    int32_t i32_input_reg[ROW_UNROLL];
    uint32_t tid_t = tid;
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      i32_input_reg[i] = i32_input[tid_t];
      tid_t += packConfig.nPack;
    }

    FType f32_output_reg[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      const float scale =
          alpha * lhs_scale[lhs_offset[i]] * rhs_scale[rhs_offset[i]];
      int32_t l_zero = static_cast<int32_t>(lhs_zero[lhs_offset[i]]);
      int32_t r_zero = static_cast<int32_t>(rhs_zero[rhs_offset[i]]);
      int32_t l_redsum = lhs_redsum[lhs_offset[i]];
      int32_t r_redsum = rhs_redsum[rhs_offset[i]];
      i32_input_reg[i] +=
          (K * l_zero * r_zero - l_redsum * r_zero - r_redsum * l_zero);
      // Avoid int32 overflow float16, we convert int32 to float32 then
      // multiply by scale
      f32_output_reg[i] = static_cast<FType>(
          active_func(static_cast<float>(i32_input_reg[i]) * scale));
    }

    tid_t = tid;
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      f32_output[tid_t] = f32_output_reg[i];
      tid_t += packConfig.nPack;
    }
  } else if (tid < packConfig.nThread) {
    tid = tid - packConfig.nPack + packConfig.nPack * ROW_UNROLL;
    uint32_t lhs_offset = 0;
    uint32_t rhs_offset = 0;
    uint32_t offset = tid;
#pragma unroll
    for (int i = 0; i < NUM_DIMS; ++i) {
      DivMod<uint32_t> d = out_divmod[i].divmod(offset);
      uint32_t dmod = d.mod;
      offset = d.div;
      lhs_offset += dmod * lhs_stride[i];
      rhs_offset += dmod * rhs_stride[i];
    }

    int32_t idata = i32_input[tid];
    const float scale = alpha * lhs_scale[lhs_offset] * rhs_scale[rhs_offset];
    int32_t l_zero = static_cast<int32_t>(lhs_zero[lhs_offset]);
    int32_t r_zero = static_cast<int32_t>(rhs_zero[rhs_offset]);
    int32_t l_redsum = lhs_redsum[lhs_offset];
    int32_t r_redsum = rhs_redsum[rhs_offset];
    idata += (K * l_zero * r_zero - l_redsum * r_zero - r_redsum * l_zero);
    // Avoid int32 overflow float16, we convert int32 to float32 then
    // multiply by scale
    f32_output[tid] =
        static_cast<FType>(active_func(static_cast<float>(idata) * scale));
  }
}

// Post Process with bias. Do ROW_UNROLL for Rhs.
template <int BLOCK, int ROW_UNROLL, int NUM_DIMS, typename FType,
          typename QType, typename Active>
__global__ __launch_bounds__(BLOCK) void postProcessWithBiasKernel(
    hie::Array<uint32_t, MAX_DIMS> lhs_stride,
    const float* __restrict__ lhs_scale, const QType* __restrict__ lhs_zero,
    const int32_t* __restrict__ lhs_redsum,
    hie::Array<uint32_t, MAX_DIMS> rhs_stride,
    const float* __restrict__ rhs_scale, const QType* __restrict__ rhs_zero,
    const int32_t* __restrict__ rhs_redsum,
    hie::Array<uint32_t, MAX_DIMS> bias_stride, const FType* __restrict__ bias,
    hie::Array<IntDivModer<uint32_t>, MAX_DIMS> out_divmod,
    const int32_t* i32_input, FType* f32_output, const int K, const FType alpha,
    const FType beta, Active active_func, PackedEltwiseConfig packConfig) {
  int tid = blockIdx.x * BLOCK + threadIdx.x;
  if (tid < packConfig.nPack) {
    uint32_t lhs_offset[ROW_UNROLL];
    uint32_t rhs_offset[ROW_UNROLL];
    uint32_t bias_offset[ROW_UNROLL];
    uint32_t offset[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      lhs_offset[i] = 0;
      rhs_offset[i] = 0;
      bias_offset[i] = 0;
      offset[i] = tid + packConfig.nPack * i;
    }

#pragma unroll
    for (int i = 0; i < NUM_DIMS; ++i) {
#pragma unroll
      for (int j = 0; j < ROW_UNROLL; ++j) {
        DivMod<uint32_t> d = out_divmod[i].divmod(offset[j]);
        uint32_t dmod = d.mod;
        offset[j] = d.div;
        lhs_offset[j] += dmod * lhs_stride[i];
        rhs_offset[j] += dmod * rhs_stride[i];
        bias_offset[j] += dmod * bias_stride[i];
      }
    }

    int32_t i32_input_reg[ROW_UNROLL];
    uint32_t tid_t = tid;
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      i32_input_reg[i] = i32_input[tid_t];
      tid_t += packConfig.nPack;
    }

    FType f32_output_reg[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      const float scale =
          alpha * lhs_scale[lhs_offset[i]] * rhs_scale[rhs_offset[i]];
      int32_t l_zero = static_cast<int32_t>(lhs_zero[lhs_offset[i]]);
      int32_t r_zero = static_cast<int32_t>(rhs_zero[rhs_offset[i]]);
      int32_t l_redsum = lhs_redsum[lhs_offset[i]];
      int32_t r_redsum = rhs_redsum[rhs_offset[i]];
      i32_input_reg[i] +=
          (K * l_zero * r_zero - l_redsum * r_zero - r_redsum * l_zero);
      // Avoid int32 overflow float16, we convert int32 to float32 then
      // multiply by scale
      f32_output_reg[i] = static_cast<FType>(
          active_func(static_cast<float>(i32_input_reg[i]) * scale +
                      beta * bias[bias_offset[i]]));
    }

    tid_t = tid;
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      f32_output[tid_t] = f32_output_reg[i];
      tid_t += packConfig.nPack;
    }
  } else if (tid < packConfig.nThread) {
    tid = tid - packConfig.nPack + packConfig.nPack * ROW_UNROLL;
    uint32_t lhs_offset = 0;
    uint32_t rhs_offset = 0;
    uint32_t bias_offset = 0;
    uint32_t offset = tid;

#pragma unroll
    for (int i = 0; i < NUM_DIMS; ++i) {
      DivMod<uint32_t> d = out_divmod[i].divmod(offset);
      uint32_t dmod = d.mod;
      offset = d.div;
      lhs_offset += dmod * lhs_stride[i];
      rhs_offset += dmod * rhs_stride[i];
      bias_offset += dmod * bias_stride[i];
    }

    int32_t idata = i32_input[tid];
    const float scale = alpha * lhs_scale[lhs_offset] * rhs_scale[rhs_offset];
    int32_t l_zero = static_cast<int32_t>(lhs_zero[lhs_offset]);
    int32_t r_zero = static_cast<int32_t>(rhs_zero[rhs_offset]);
    int32_t l_redsum = lhs_redsum[lhs_offset];
    int32_t r_redsum = rhs_redsum[rhs_offset];
    idata += (K * l_zero * r_zero - l_redsum * r_zero - r_redsum * l_zero);
    // Avoid int32 overflow float16, we convert int32 to float32 then
    // multiply by scale
    f32_output[tid] = static_cast<FType>(active_func(
        static_cast<float>(idata) * scale + beta * bias[bias_offset]));
  }
}

// Post Process with bias and elementwise(residual_add or residual_mul). Do
// ROW_UNROLL for Rhs. T1 float T2 int32 QType int8 elemwiseA is int8-gemm`s
// int32 result
template <int BLOCK, int ROW_UNROLL, int NUM_DIMS, typename FType,
          typename QType, typename Active>
__global__ __launch_bounds__(BLOCK) void postProcessWithBiasAndElemWiseBKernel(
    hie::Array<uint32_t, MAX_DIMS> lhs_stride,
    const float* __restrict__ lhs_scale, const QType* __restrict__ lhs_zero,
    const int32_t* __restrict__ lhs_redsum,
    hie::Array<uint32_t, MAX_DIMS> rhs_stride,
    const float* __restrict__ rhs_scale, const QType* __restrict__ rhs_zero,
    const int32_t* __restrict__ rhs_redsum,
    hie::Array<uint32_t, MAX_DIMS> bias_stride, const FType* __restrict__ bias,
    hie::Array<uint32_t, MAX_DIMS> elemwiseA_stride,
    const int32_t* elemwiseA_input,
    hie::Array<uint32_t, MAX_DIMS> elemwiseB_stride,
    const FType* elemwiseB_input,
    hie::Array<IntDivModer<uint32_t>, MAX_DIMS> out_divmod, FType* f32_output,
    const int K, const FType alpha, const FType beta, Active active_func,
    PackedEltwiseConfig packConfig) {
  int tid = blockIdx.x * BLOCK + threadIdx.x;
  if (tid < packConfig.nPack) {
    uint32_t lhs_offset[ROW_UNROLL];
    uint32_t rhs_offset[ROW_UNROLL];
    uint32_t bias_offset[ROW_UNROLL];
    uint32_t elemwiseA_offset[ROW_UNROLL];
    uint32_t elemwiseB_offset[ROW_UNROLL];
    uint32_t offset[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      lhs_offset[i] = 0;
      rhs_offset[i] = 0;
      bias_offset[i] = 0;
      elemwiseA_offset[i] = 0;
      elemwiseB_offset[i] = 0;
      offset[i] = tid + packConfig.nPack * i;
    }

#pragma unroll
    for (int i = 0; i < NUM_DIMS; ++i) {
#pragma unroll
      for (int j = 0; j < ROW_UNROLL; ++j) {
        DivMod<uint32_t> d = out_divmod[i].divmod(offset[j]);
        uint32_t dmod = d.mod;
        offset[j] = d.div;
        lhs_offset[j] += dmod * lhs_stride[i];
        rhs_offset[j] += dmod * rhs_stride[i];
        bias_offset[j] += dmod * bias_stride[i];
        elemwiseA_offset[j] += dmod * elemwiseA_stride[i];
        elemwiseB_offset[j] += dmod * elemwiseB_stride[i];
      }
    }

    int32_t elemwiseA_input_reg[ROW_UNROLL];
    FType elemwiseB_input_reg[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      elemwiseA_input_reg[i] = elemwiseA_input[elemwiseA_offset[i]];
      elemwiseB_input_reg[i] = elemwiseB_input[elemwiseB_offset[i]];
    }

    FType f32_output_reg[ROW_UNROLL];
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      const float scale =
          alpha * lhs_scale[lhs_offset[i]] * rhs_scale[rhs_offset[i]];
      int32_t l_zero = static_cast<int32_t>(lhs_zero[lhs_offset[i]]);
      int32_t r_zero = static_cast<int32_t>(rhs_zero[rhs_offset[i]]);
      int32_t l_redsum = lhs_redsum[lhs_offset[i]];
      int32_t r_redsum = rhs_redsum[rhs_offset[i]];
      elemwiseA_input_reg[i] +=
          (K * l_zero * r_zero - l_redsum * r_zero - r_redsum * l_zero);
      // Avoid int32 overflow float16, we convert int32 to float32 then
      // multiply by scale
      f32_output_reg[i] =
          static_cast<FType>(
              active_func(static_cast<float>(elemwiseA_input_reg[i]) * scale +
                          beta * bias[bias_offset[i]])) +
          elemwiseB_input_reg[i];
    }
#pragma unroll
    for (int i = 0; i < ROW_UNROLL; ++i) {
      f32_output[tid] = f32_output_reg[i];
      tid += packConfig.nPack;
    }
  } else if (tid < packConfig.nThread) {
    tid = tid - packConfig.nPack + packConfig.nPack * ROW_UNROLL;
    uint32_t lhs_offset = 0;
    uint32_t rhs_offset = 0;
    uint32_t bias_offset = 0;
    uint32_t elemwiseA_offset = 0;
    uint32_t elemwiseB_offset = 0;
    uint32_t offset = tid;

#pragma unroll
    for (int j = 0; j < NUM_DIMS; ++j) {
      DivMod<uint32_t> d = out_divmod[j].divmod(offset);
      uint32_t dmod = d.mod;
      offset = d.div;
      lhs_offset += dmod * lhs_stride[j];
      rhs_offset += dmod * rhs_stride[j];
      bias_offset += dmod * bias_stride[j];
      elemwiseA_offset += dmod * elemwiseA_stride[j];
      elemwiseB_offset += dmod * elemwiseB_stride[j];
    }

    int32_t elemwiseA_input_reg = elemwiseA_input[elemwiseA_offset];
    const float scale = alpha * lhs_scale[lhs_offset] * rhs_scale[rhs_offset];
    int32_t l_zero = static_cast<int32_t>(lhs_zero[lhs_offset]);
    int32_t r_zero = static_cast<int32_t>(rhs_zero[rhs_offset]);
    int32_t l_redsum = lhs_redsum[lhs_offset];
    int32_t r_redsum = rhs_redsum[rhs_offset];
    elemwiseA_input_reg +=
        (K * l_zero * r_zero - l_redsum * r_zero - r_redsum * l_zero);
    // Avoid int32 overflow float16, we convert int32 to float32 then
    // multiply by scale
    f32_output[tid] = static_cast<FType>(active_func(
                          static_cast<float>(elemwiseA_input_reg) * scale +
                          beta * bias[bias_offset])) +
                      elemwiseB_input[elemwiseA_offset];
  }
}

template <int BLOCK_SIZE, int ROW_UNROLL, int NUM_DIMS, typename FType,
          typename QType, typename Active>
void postProcessUnrollImp(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const QType* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const QType* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const FType* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const FType* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, FType* output_ptr,
    const FType& alpha, const FType& beta, const int& K, Active active_func,
    cudaStream_t& stream) {
  bool with_bias = (bias != nullptr) ? true : false;
  bool with_elemwiseB = (elemwiseB != nullptr) ? true : false;

  // Compute stride for broadcast.
  hie::Array<uint32_t, MAX_DIMS> lhs_stride;
  hie::Array<uint32_t, MAX_DIMS> rhs_stride;
  hie::Array<uint32_t, MAX_DIMS> bias_stride;
  hie::Array<uint32_t, MAX_DIMS> elemwiseA_stride;
  hie::Array<uint32_t, MAX_DIMS> elemwiseB_stride;
  hie::Array<IntDivModer<uint32_t>, MAX_DIMS> out_divmod;
  uint32_t lhs_mul = 1;
  uint32_t rhs_mul = 1;
  uint32_t bias_mul = 1;
  uint32_t elemwiseA_mul = 1;
  uint32_t elemwiseB_mul = 1;
  for (int i = 0; i < out_ndims; ++i) {
    out_divmod[i] = IntDivModer<uint32_t>(out_dims[out_ndims - 1 - i]);
    lhs_stride[i] = (i >= lhs_ndims || out_dims[out_ndims - 1 - i] >
                                           lhs_reduce_dims[lhs_ndims - 1 - i])
                        ? 0
                        : lhs_mul;
    rhs_stride[i] = (i >= rhs_ndims || out_dims[out_ndims - 1 - i] >
                                           rhs_reduce_dims[rhs_ndims - 1 - i])
                        ? 0
                        : rhs_mul;
    elemwiseA_stride[i] =
        (i >= elemwiseA_ndims ||
         out_dims[out_ndims - 1 - i] > elemwiseA_dims[elemwiseA_ndims - 1 - i])
            ? 0
            : elemwiseA_mul;

    if (i < lhs_ndims) lhs_mul *= lhs_reduce_dims[lhs_ndims - 1 - i];
    if (i < rhs_ndims) rhs_mul *= rhs_reduce_dims[rhs_ndims - 1 - i];
    if (i < elemwiseA_ndims)
      elemwiseA_mul *= elemwiseA_dims[elemwiseA_ndims - 1 - i];

    if (with_bias == true) {
      bias_stride[i] = (i >= bias_ndims || out_dims[out_ndims - 1 - i] >
                                               bias_dims[bias_ndims - 1 - i])
                           ? 0
                           : bias_mul;
      if (i < bias_ndims) bias_mul *= bias_dims[bias_ndims - 1 - i];
    }
    if (with_elemwiseB == true) {
      elemwiseB_stride[i] =
          (i >= elemwiseB_ndims || out_dims[out_ndims - 1 - i] >
                                       elemwiseB_dims[elemwiseB_ndims - 1 - i])
              ? 0
              : elemwiseB_mul;
      if (i < elemwiseB_ndims)
        elemwiseB_mul *= elemwiseB_dims[elemwiseB_ndims - 1 - i];
    }
  }

  size_t output_size = 1;
  for (int i = 0; i < out_ndims; ++i) {
    output_size *= out_dims[i];
  }
  PackedEltwiseConfig packConfig(output_size, 1, BLOCK_SIZE, ROW_UNROLL);
  // Lanuch Kernel
  if (with_bias == false && with_elemwiseB == false) {
    postProcessKernel<BLOCK_SIZE, ROW_UNROLL, NUM_DIMS, FType, QType>
        <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
            lhs_stride, lhs_scale, lhs_zero, lhs_redsum, rhs_stride, rhs_scale,
            rhs_zero, rhs_redsum, out_divmod, elemwiseA, output_ptr, K, alpha,
            active_func, packConfig);
  } else if (with_bias == true && with_elemwiseB == false) {
    postProcessWithBiasKernel<BLOCK_SIZE, ROW_UNROLL, NUM_DIMS, FType, QType>
        <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
            lhs_stride, lhs_scale, lhs_zero, lhs_redsum, rhs_stride, rhs_scale,
            rhs_zero, rhs_redsum, bias_stride, bias, out_divmod, elemwiseA,
            output_ptr, K, alpha, beta, active_func, packConfig);
  } else if (with_bias == true && with_elemwiseB == true) {
    postProcessWithBiasAndElemWiseBKernel<BLOCK_SIZE, ROW_UNROLL, NUM_DIMS,
                                          FType, QType>
        <<<packConfig.nBlock, BLOCK_SIZE, 0, stream>>>(
            lhs_stride, lhs_scale, lhs_zero, lhs_redsum, rhs_stride, rhs_scale,
            rhs_zero, rhs_redsum, bias_stride, bias, elemwiseA_stride,
            elemwiseA, elemwiseB_stride, elemwiseB, out_divmod, output_ptr, K,
            alpha, beta, active_func, packConfig);
  }
}

// Instantiate NUM_DIMS and reduce number of cycles(for) in kernel, that improve
// kernel performance by 10%. template<int NUM_DIMS, typename QType, typename
// Active>
template <int NUM_DIMS, typename FType, typename QType, typename Active>
void postProcessNDimsImp(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const QType* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const QType* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const FType* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const FType* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, FType* output_ptr,
    const FType& alpha, const FType& beta, const int& K, Active active_func,
    cudaStream_t& stream) {
  size_t output_size = 1;
  for (int i = 0; i < out_ndims; ++i) {
    output_size *= out_dims[i];
  }

  const int BLOCK_SIZE = 128;
  int ROW_UNROLL = 4;
  PackedEltwiseConfig packConfig;
  for (; ROW_UNROLL >= 1; ROW_UNROLL /= 2) {
    packConfig.update(output_size, 1, BLOCK_SIZE, ROW_UNROLL);
    // Most GPUs` SM Count >= 32, for example T4_SM_Count = 40.
    // The purpose of nBlock >= 32 is that improve SM utilization.
    if (packConfig.nBlock >= 32 || ROW_UNROLL == 1) {
      break;
    }
  }
  switch (ROW_UNROLL) {
    case 4:
      postProcessUnrollImp<BLOCK_SIZE, 4, NUM_DIMS>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 2:
      postProcessUnrollImp<BLOCK_SIZE, 2, NUM_DIMS>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 1:
      postProcessUnrollImp<BLOCK_SIZE, 1, NUM_DIMS>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    default:
      break;
      // HIE_ABORT("ROW_UNROLL Unsupported : ", ROW_UNROLL);
  }
}

// template<typename QType, typename GetOp, typename ...Arg>
template <typename FType, typename QType, typename Active>
void postProcessImp(const int lhs_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
                    const float* lhs_scale, const QType* lhs_zero,
                    const int* lhs_redsum, const int rhs_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
                    const float* rhs_scale, const QType* rhs_zero,
                    const int* rhs_redsum, const int bias_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
                    const FType* bias, const int elemwiseA_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims,
                    const int* elemwiseA, const int elemwiseB_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
                    const FType* elemwiseB, const int out_ndims,
                    const hie::Array<uint32_t, MAX_DIMS>& out_dims,
                    FType* output_ptr, const FType& alpha, const FType& beta,
                    const int& K, cudaStream_t& stream, Active active) {
  // avtive
  auto active_func = active;
  switch (out_ndims) {
    case 6:
      postProcessNDimsImp<6>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 5:
      postProcessNDimsImp<5>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 4:
      postProcessNDimsImp<4>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 3:
      postProcessNDimsImp<3>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 2:
      postProcessNDimsImp<2>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    case 1:
      postProcessNDimsImp<1>(
          lhs_ndims, lhs_reduce_dims, lhs_scale, lhs_zero, lhs_redsum,
          rhs_ndims, rhs_reduce_dims, rhs_scale, rhs_zero, rhs_redsum,
          bias_ndims, bias_dims, bias, elemwiseA_ndims, elemwiseA_dims,
          elemwiseA, elemwiseB_ndims, elemwiseB_dims, elemwiseB, out_ndims,
          out_dims, output_ptr, alpha, beta, K, active_func, stream);
      break;
    default:
      break;
      // HIE_ABORT("Output`s NUM_DIMS ERROR : ", out_ndims);
  }
}
template void postProcessImp<float, int8_t, hie::NONE<float>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const float* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const float* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, float* output_ptr,
    const float& alpha, const float& beta, const int& K, cudaStream_t& stream,
    hie::NONE<float> active);
template void postProcessImp<float, int8_t, hie::GELU<float>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const float* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const float* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, float* output_ptr,
    const float& alpha, const float& beta, const int& K, cudaStream_t& stream,
    hie::GELU<float> active);
template void postProcessImp<float, int8_t, hie::GELU_TANH<float>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const float* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const float* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, float* output_ptr,
    const float& alpha, const float& beta, const int& K, cudaStream_t& stream,
    hie::GELU_TANH<float> active);
#ifdef ENABLE_FP16
template void postProcessImp<half, int8_t, hie::NONE<half>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const half* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims, const half* elemwiseB,
    const int out_ndims, const hie::Array<uint32_t, MAX_DIMS>& out_dims,
    half* output_ptr, const half& alpha, const half& beta, const int& K,
    cudaStream_t& stream, hie::NONE<half> active);
template void postProcessImp<half, int8_t, hie::GELU<half>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const half* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims, const half* elemwiseB,
    const int out_ndims, const hie::Array<uint32_t, MAX_DIMS>& out_dims,
    half* output_ptr, const half& alpha, const half& beta, const int& K,
    cudaStream_t& stream, hie::GELU<half> active);
template void postProcessImp<half, int8_t, hie::GELU_TANH<half>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const half* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims, const half* elemwiseB,
    const int out_ndims, const hie::Array<uint32_t, MAX_DIMS>& out_dims,
    half* output_ptr, const half& alpha, const half& beta, const int& K,
    cudaStream_t& stream, hie::GELU_TANH<half> active);
#endif
template void postProcessImp<hie::bfloat16, int8_t, hie::NONE<hie::bfloat16>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const hie::bfloat16* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const hie::bfloat16* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, hie::bfloat16* output_ptr,
    const hie::bfloat16& alpha, const hie::bfloat16& beta, const int& K,
    cudaStream_t& stream, hie::NONE<hie::bfloat16> active);
template void postProcessImp<hie::bfloat16, int8_t, hie::GELU<hie::bfloat16>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const hie::bfloat16* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const hie::bfloat16* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, hie::bfloat16* output_ptr,
    const hie::bfloat16& alpha, const hie::bfloat16& beta, const int& K,
    cudaStream_t& stream, hie::GELU<hie::bfloat16> active);
template void
postProcessImp<hie::bfloat16, int8_t, hie::GELU_TANH<hie::bfloat16>>(
    const int lhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& lhs_reduce_dims,
    const float* lhs_scale, const int8_t* lhs_zero, const int* lhs_redsum,
    const int rhs_ndims, const hie::Array<uint32_t, MAX_DIMS>& rhs_reduce_dims,
    const float* rhs_scale, const int8_t* rhs_zero, const int* rhs_redsum,
    const int bias_ndims, const hie::Array<uint32_t, MAX_DIMS>& bias_dims,
    const hie::bfloat16* bias, const int elemwiseA_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseA_dims, const int* elemwiseA,
    const int elemwiseB_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& elemwiseB_dims,
    const hie::bfloat16* elemwiseB, const int out_ndims,
    const hie::Array<uint32_t, MAX_DIMS>& out_dims, hie::bfloat16* output_ptr,
    const hie::bfloat16& alpha, const hie::bfloat16& beta, const int& K,
    cudaStream_t& stream, hie::GELU_TANH<hie::bfloat16> active);
}  // namespace cuda
}  // namespace allspark
