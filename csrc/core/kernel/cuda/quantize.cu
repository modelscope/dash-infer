/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    quantize.cu
 */

#include <cmath>
#include <limits>
#include <string>

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#define CUDA_HOST_DEVICE __forceinline__ __device__ __host__
#define CUDA_DEVICE __forceinline__ __device__
namespace allspark {
namespace cuda {
enum class FloatRoundStyle { round_to_nearest, round_to_upper, round_to_lower };

template <typename Target, typename Source,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct NumericConverter;

template <typename Target>
struct NumericConverter<Target, float, FloatRoundStyle::round_to_nearest> {
  static CUDA_HOST_DEVICE Target Convert(float const& s) { return roundf(s); }
};

#ifdef ENABLE_FP16
template <typename Target>
struct NumericConverter<Target, half, FloatRoundStyle::round_to_nearest> {
  static CUDA_HOST_DEVICE Target Convert(half const& s) {
    return roundf(float(s));
  }
};
#endif
template <typename Target>
struct NumericConverter<Target, hie::bfloat16,
                        FloatRoundStyle::round_to_nearest> {
  static CUDA_HOST_DEVICE Target Convert(hie::bfloat16 const& s) {
    return roundf(float(s));
  }
};
/**
 * @brief
 *
 */

// Scale + ZeroPoint
template <typename FType, typename ComputeType = float>
CUDA_HOST_DEVICE ComputeType QLinear(const FType& in_data,
                                     const ComputeType& scale,
                                     const ComputeType& zero_point) {
  return ComputeType(in_data) / scale + zero_point;
}
template <typename FType, typename QType, typename ComputeType = float>
CUDA_HOST_DEVICE ComputeType QLinear(const FType& in_data,
                                     const ComputeType& scale,
                                     const QType& zero_point) {
  return QLinear<FType, ComputeType>(in_data, ComputeType(scale),
                                     ComputeType(zero_point));
}

// (Data - ZeroPoint) * Scale
template <typename QType, typename FType, typename ComputeType = float>
CUDA_HOST_DEVICE FType DQLinear(const QType& in_data, const float& scale,
                                const QType& zero_point) {
  return FType((ComputeType(in_data) - ComputeType(zero_point)) *
               ComputeType(scale));
}

// Clamp
template <typename ComputeType = float>
CUDA_HOST_DEVICE ComputeType Clamp(const ComputeType& v, const ComputeType& lo,
                                   const ComputeType& hi) {
  return fmax(fmin(v, hi), lo);
}

/**
 * @brief Compute quantize params
 *
 */

template <typename FType, typename QType, typename ComputeType = float,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct ComputeQuatizationParam {
  using NumConverter = NumericConverter<QType, ComputeType, Round>;
  static constexpr QType QMax = QType(std::numeric_limits<QType>::max());
  static constexpr QType QMin = QType(std::numeric_limits<QType>::min());

  static CUDA_DEVICE void Run(const FType fmax, const FType fmin,
                              ComputeType& scale, ComputeType& zero_point) {
    const ComputeType fmin_t = std::fmin(ComputeType(0), ComputeType(fmin));
    const ComputeType fmax_t = std::fmax(ComputeType(0), ComputeType(fmax));
    scale = (fmax_t - fmin_t) / ComputeType(QMax - QMin);
    const ComputeType init_zero_point = ComputeType(QMin) - fmin_t / scale;
    zero_point = Clamp(init_zero_point, ComputeType(QMin), ComputeType(QMax));
  }

  static CUDA_DEVICE void Run(const FType fmax, const FType fmin,
                              ComputeType& scale, QType& zero_point) {
    ComputeType zero_point_t;
    Run(fmax, fmin, scale, zero_point_t);
    zero_point = NumConverter::Convert(zero_point_t);
  }
};

/**
 * @brief Quantize
 *
 */

template <typename FType, typename QType, typename ComputeType = float,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
struct Quantize {
  using NumConverter = NumericConverter<QType, ComputeType, Round>;
  static constexpr QType QMax = QType(std::numeric_limits<QType>::max());
  static constexpr QType QMin = QType(std::numeric_limits<QType>::min());

  CUDA_HOST_DEVICE QType operator()(const FType& in_data, const float& scale,
                                    const QType& zero_point) {
    return Quant(in_data, scale, zero_point);
  }

  static CUDA_DEVICE QType Quant(const FType& in_data, const ComputeType& scale,
                                 const QType& zero_point) {
    ComputeType imd = QLinear(in_data, scale, zero_point);
    return NumConverter::Convert(
        Clamp(imd, ComputeType(QMin), ComputeType(QMax)));
  }

  static CUDA_DEVICE QType Quant(const FType& in_data, const ComputeType& scale,
                                 const ComputeType& zero_point) {
    ComputeType imd = QLinear(in_data, scale, zero_point);
    return NumConverter::Convert(
        Clamp(imd, ComputeType(QMin), ComputeType(QMax)));
  }

  template <int UNROLL>
  static CUDA_DEVICE void Quant(const FType (&in_data)[UNROLL],
                                QType (&out_data)[UNROLL],
                                const ComputeType& scale,
                                const QType& zero_point) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      out_data[i] = Quant(in_data[i], scale, zero_point);
    }
  }

  template <int UNROLL>
  static CUDA_DEVICE void Quant(const FType (&in_data)[UNROLL],
                                QType (&out_data)[UNROLL],
                                const ComputeType& scale,
                                const ComputeType& zero_point) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      out_data[i] = Quant(in_data[i], scale, zero_point);
    }
  }
};

/**
 * @brief Dequantize
 *
 */

template <typename QType, typename FType, typename ComputeType = float>
struct DeQuantize {
  CUDA_HOST_DEVICE FType operator()(const QType& in_data, const float& scale,
                                    const QType& zero_point) {
    return DQLinear<QType, FType, ComputeType>(in_data, scale, zero_point);
  }
};

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
template <typename T>
struct MinOp {
 public:
  static constexpr T init = std::numeric_limits<T>::max();
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x < y ? x : y;
  }
};
template <>
struct MinOp<float> {
 public:
  static constexpr float init = std::numeric_limits<float>::infinity();
  static float __device__ __forceinline__ op(const float& x, const float& y) {
    return x < y ? x : y;
  }
};

template <typename T>
struct SumOp {
 public:
  static constexpr T init = T(0);
  static T __device__ __forceinline__ op(const T& x, const T& y) {
    return x + y;
  }
};

static constexpr int WARP_SIZE = 32;

#if 0
template <template <class> class Func, typename T> CUDA_DEVICE T ReduceWarp(T val) {
#pragma unroll
    for (int i = WARP_SIZE; i > 1; i /= 2) {
        T tmp = __shfl_down_sync(0xffffffff >> (WARP_SIZE - i), val, i / 2, i);
        val = Func<T>::op(tmp, val);
    }
    return val;
}
#else
template <template <class> class Func, typename T>
CUDA_DEVICE T ReduceWarp(T val) {
#pragma unroll
  for (int i = WARP_SIZE; i > 1; i /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, val, i / 2);
    val = Func<T>::op(tmp, val);
  }
  return val;
}
#endif

template <template <class> class Func, typename T, int BLOCK>
CUDA_DEVICE T ReduceBlock(const T& val) {
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

template <template <class> class Func, typename T, int BLOCK, int UNROLL>
CUDA_DEVICE T ReduceBlock(const T (&data)[UNROLL]) {
  T val_reg = Func<T>::init;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    val_reg = Func<T>::op(data[i], val_reg);
  }
  return ReduceBlock<Func, T, BLOCK>(val_reg);
}

/**
 * @brief Quantization PerChannel
 */

template <typename T>
__device__ __forceinline__ T ldg_set0(const void* ptr, bool guard);

template <>
__device__ __forceinline__ float ldg_set0<float>(const void* ptr, bool guard) {
  float ret;
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cs.L2::128B.b32 %0, [%1];}\n"
#else
      " @p ld.global.cs.b32 %0, [%1];}\n"
#endif
      : "=f"(ret)
      : "l"(ptr), "r"((int)guard));
  return ret;
}

#ifdef ENABLE_FP16
template <>
__device__ __forceinline__ half ldg_set0<half>(const void* ptr, bool guard) {
  half ret;
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b16 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cs.L2::128B.b16 %0, [%1];}\n"
#else
      " @p ld.global.cs.b16 %0, [%1];}\n"
#endif
      : "=h"(reinterpret_cast<uint16_t&>(ret))
      : "l"(ptr), "r"((int)guard));
  return ret;
}
#endif

template <>
__device__ __forceinline__ int8_t ldg_set0<int8_t>(const void* ptr,
                                                   bool guard) {
  uint32_t ret;
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cs.L2::128B.b8 %0, [%1];}\n"
#else
      " @p ld.global.cs.b8 %0, [%1];}\n"
#endif
      : "=r"(ret)
      : "l"(ptr), "r"((int32_t)guard));
  return ret;
}

template <>
__device__ __forceinline__ int32_t ldg_set0<int32_t>(const void* ptr,
                                                     bool guard) {
  int32_t ret;
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750 && !defined(__HGGCCC__)
      " @p ld.global.cs.L2::128B.b32 %0, [%1];}\n"
#else
      " @p ld.global.cs.b32 %0, [%1];}\n"
#endif
      : "=r"(ret)
      : "l"(ptr), "r"((int32_t)guard));
  return ret;
}

template <typename T>
__device__ __forceinline__ void stg(const T& reg, void* ptr);

template <>
__device__ __forceinline__ void stg<float>(const float& reg, void* ptr) {
  asm volatile("st.global.cs.b32 [%1], %0;\n" : : "f"(reg), "l"(ptr));
}

#ifdef ENABLE_FP16
template <>
__device__ __forceinline__ void stg<half>(const half& reg, void* ptr) {
  asm volatile("st.global.cs.b16 [%1], %0;\n"
               :
               : "h"(reinterpret_cast<const uint16_t&>(reg)), "l"(ptr));
}
#endif
template <int MAX_STEP>
struct KernelConfig {
  static constexpr int UNROLL = 4;
  static constexpr int BLOCK = MAX_STEP / UNROLL;
};
template <typename FType, typename QType, int BLOCK, int UNROLL,
          typename ComputeType = float>
__global__ __launch_bounds__(BLOCK) void quantize_per_channel_one_pass(
    const FType* __restrict__ fdata, QType* __restrict__ qdata,
    ComputeType* __restrict__ scale, QType* __restrict__ zero_point,
    int* __restrict__ redsum, const int inner) {
  const int tid = threadIdx.x;
  const FType* fdata_ptr = fdata + blockIdx.x * inner + tid;
  ComputeType ld_reg[UNROLL] = {0};
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    bool guard = (tid + i * BLOCK) < inner;
    ld_reg[i] = ComputeType(ldg_set0<FType>(fdata_ptr + i * BLOCK, guard));
  }
  ComputeType fmax = ReduceBlock<MaxOp, ComputeType, BLOCK, UNROLL>(ld_reg);
  ComputeType fmin = ReduceBlock<MinOp, ComputeType, BLOCK, UNROLL>(ld_reg);

  __shared__ ComputeType scale_smem;
  __shared__ ComputeType zero_point_smem;
  if (tid == 0) {
    ComputeQuatizationParam<ComputeType, QType>::Run(fmax, fmin, scale_smem,
                                                     zero_point_smem);
    scale[blockIdx.x] = scale_smem;
    zero_point[blockIdx.x] = static_cast<QType>(roundf(zero_point_smem));
    redsum[blockIdx.x] = 0;  // Clean redsum to Zero-Value, Because we're
                             // going to use atomic add!
  }
  __syncthreads();

  const ComputeType scale_tmp = scale_smem;
  const ComputeType zero_point_tmp = zero_point_smem;
  QType st_reg[UNROLL];
  Quantize<ComputeType, QType>::Quant(ld_reg, st_reg, scale_tmp,
                                      zero_point_tmp);

  QType* qdata_ptr = qdata + blockIdx.x * inner;
  int sum_tmp = 0;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int col_idx = tid + i * BLOCK;
    if (col_idx < inner) {
      qdata_ptr[col_idx] = st_reg[i];
      sum_tmp += static_cast<int>(st_reg[i]);
    }
  }
  sum_tmp = ReduceWarp<SumOp, int>(sum_tmp);
  if (tid % WARP_SIZE == 0) {
    atomicAdd(redsum + blockIdx.x, sum_tmp);
  }
}
template <typename FType, typename QType, int BLOCK, int UNROLL,
          typename ComputeType = float>
__global__ __launch_bounds__(BLOCK) void quantize_per_channel_one_pass_warp(
    const FType* __restrict__ fdata, QType* __restrict__ qdata,
    ComputeType* __restrict__ scale, QType* __restrict__ zero_point,
    int* __restrict__ redsum, const int inner, const int outer) {
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int outer_idx = blockIdx.x * (BLOCK / WARP_SIZE) + warp_id;
  if (outer_idx >= outer) return;

  const FType* fdata_ptr = fdata + outer_idx * inner + lane_id;

  ComputeType fmax = ComputeType(-INFINITY);
  ComputeType fmin = ComputeType(INFINITY);
  ComputeType ld_reg[UNROLL] = {0};
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    bool guard = (lane_id + i * WARP_SIZE) < inner;
    ld_reg[i] = ldg_set0<FType>(fdata_ptr + i * WARP_SIZE, guard);
    fmax = std::fmax(ld_reg[i], fmax);
    fmin = std::fmin(ld_reg[i], fmin);
  }
  fmax = ReduceWarp<MaxOp, ComputeType>(fmax);
  fmin = ReduceWarp<MinOp, ComputeType>(fmin);

  __shared__ ComputeType scale_smem[BLOCK / WARP_SIZE];
  __shared__ ComputeType zero_point_smem[BLOCK / WARP_SIZE];
  if (lane_id == 0) {
    ComputeQuatizationParam<ComputeType, QType>::Run(
        fmax, fmin, scale_smem[warp_id], zero_point_smem[warp_id]);
  }
  __syncthreads();

  QType st_reg[UNROLL];
  Quantize<ComputeType, QType>::Quant(ld_reg, st_reg, scale_smem[warp_id],
                                      zero_point_smem[warp_id]);

  QType* qdata_ptr = qdata + outer_idx * inner;
  int sum_tmp = 0;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int col_idx = lane_id + i * WARP_SIZE;
    if (col_idx < inner) {
      qdata_ptr[col_idx] = st_reg[i];
      sum_tmp += static_cast<int>(st_reg[i]);
    }
  }
  sum_tmp = ReduceWarp<SumOp, int>(sum_tmp);
  if (lane_id == 0) {
    scale[outer_idx] = scale_smem[warp_id];
    zero_point[outer_idx] =
        static_cast<QType>(roundf(zero_point_smem[warp_id]));
    redsum[outer_idx] = sum_tmp;
  }
}
template <typename FType, typename QType, typename ComputeType>
void QuantizePerChannelImp(const FType* fdata_ptr, QType* qdata_ptr,
                           ComputeType* scale_ptr, QType* zero_point_ptr,
                           int* redsum_ptr, const int inner_len,
                           const int outer_len, cudaStream_t stream) {
  const int GRID_SIZE = outer_len;
  if (inner_len <= 64) {
    const int UNROLL = 64 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 96) {
    const int UNROLL = 96 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 128) {
    const int UNROLL = 128 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 192) {
    const int UNROLL = 192 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 256) {
    const int UNROLL = 256 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 384) {
    const int UNROLL = 384 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 512) {
    const int UNROLL = 512 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 768) {
    const int UNROLL = 768 / WARP_SIZE;
    const int BLOCK = 128;
    const int NUM_WARP = BLOCK / WARP_SIZE;
    const int grid_t = (outer_len + NUM_WARP - 1) / NUM_WARP;
    // quantize_test<<<1, 128, 0, stream>>>(fdata_ptr, qdata_ptr, inner_len,
    //                                      outer_len);
    quantize_per_channel_one_pass_warp<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<grid_t, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                       zero_point_ptr, redsum_ptr, inner_len,
                                       outer_len);
  } else if (inner_len <= 1024) {
    using Cfg = KernelConfig<1024>;
    const int UNROLL = Cfg::UNROLL;
    const int BLOCK = Cfg::BLOCK;
    quantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr,
                                          inner_len);
  } else if (inner_len <= 1536) {
    const int UNROLL = 6;
    const int BLOCK = 256;
    quantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr,
                                          inner_len);
  } else if (inner_len <= 2048) {
    using Cfg = KernelConfig<2048>;
    const int UNROLL = Cfg::UNROLL;
    const int BLOCK = Cfg::BLOCK;
    quantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr,
                                          inner_len);
  } else if (inner_len <= 4096) {
    using Cfg = KernelConfig<4096>;
    const int UNROLL = Cfg::UNROLL;
    const int BLOCK = Cfg::BLOCK;
    quantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr,
                                          inner_len);
  } else if (inner_len <= 8192) {
    const int UNROLL = 8;
    const int BLOCK = 1024;
    quantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr,
                                          inner_len);
  } else if (inner_len <= 32768) {
    const int UNROLL = 32;
    const int BLOCK = 1024;
    quantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr,
                                          inner_len);
  }
}
template void QuantizePerChannelImp<float, int8_t, float>(
    const float* fdata_ptr, int8_t* qdata_ptr, float* scale_ptr,
    int8_t* zero_point_ptr, int* redsum_ptr, const int inner_len,
    const int outer_len, cudaStream_t stream);
#ifdef ENABLE_FP16
template void QuantizePerChannelImp<half, int8_t, float>(
    const half* fdata_ptr, int8_t* qdata_ptr, float* scale_ptr,
    int8_t* zero_point_ptr, int* redsum_ptr, const int inner_len,
    const int outer_len, cudaStream_t stream);
#endif
template <typename FType, typename QType, int BLOCK, int UNROLL,
          typename ComputeType = float>
__global__ __launch_bounds__(BLOCK) void dequantize_per_channel_one_pass(
    FType* __restrict__ fdata, const QType* __restrict__ qdata,
    const ComputeType* __restrict__ scale,
    const ComputeType* __restrict__ zero_point, const int* __restrict__ redsum,
    const int inner, const int outer) {
  const int tid = threadIdx.x;
  FType* fdata_ptr = fdata + blockIdx.x * inner;
  const QType* qdata_ptr = qdata + blockIdx.x * inner;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    const int col_idx = tid + i * BLOCK;
    if (col_idx < inner) {
      fdata_ptr[col_idx] =
          (ComputeType(qdata_ptr[col_idx]) - ComputeType(zero_point[col_idx])) *
          ComputeType(scale[col_idx]);
    }
  }
}
template <typename FType, typename QType, int BLOCK, int UNROLL,
          typename ComputeType = float>
__global__ __launch_bounds__(BLOCK) void dequantize_per_channel_one_pass_warp(
    FType* __restrict__ fdata, const QType* __restrict__ qdata,
    const ComputeType* __restrict__ scale, const QType* __restrict__ zero_point,
    const int* __restrict__ redsum, const int inner, const int outer) {
  //     const int lane_id = threadIdx.x % WARP_SIZE;
  //     const int warp_id = threadIdx.x / WARP_SIZE;
  //     const int outer_idx = blockIdx.x * (BLOCK / WARP_SIZE) + warp_id;
  //     if (outer_idx >= outer) return;
  //     FType* fdata_ptr = fdata + outer_idx * inner + lane_id;
  //     const QType* qdata_ptr = qdata + outer_idx * inner;
  // #pragma unroll
  //     for (int i = 0; i < UNROLL; ++i) {
  //         const int col_idx = lane_id + i * WARP_SIZE;
  //         if (col_idx < inner) {
  //             fdata_ptr[col_idx] = (ComputeType(qdata_ptr[col_idx]) -
  //             ComputeType(zero_point[outer_idx])) *
  //             ComputeType(scale[outer_idx]);
  //         }
  //     }
}
template <typename FType, typename QType, typename ComputeType>
void DeQuantizePerChannelImp(FType* fdata_ptr, const QType* qdata_ptr,
                             const ComputeType* scale_ptr,
                             const ComputeType* zero_point_ptr,
                             const int* redsum_ptr, const int inner_len,
                             const int outer_len, cudaStream_t stream) {
  const int GRID_SIZE = outer_len;
  if (inner_len <= 1024) {
    using Cfg = KernelConfig<1024>;
    const int UNROLL = Cfg::UNROLL;
    const int BLOCK = Cfg::BLOCK;
    dequantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr, inner_len,
                                          outer_len);
  } else if (inner_len <= 4096) {
    using Cfg = KernelConfig<4096>;
    const int UNROLL = Cfg::UNROLL;
    const int BLOCK = Cfg::BLOCK;
    dequantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr, inner_len,
                                          outer_len);
  } else if (inner_len <= 8192) {
    const int UNROLL = 8;
    const int BLOCK = 1024;
    dequantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr, inner_len,
                                          outer_len);
  } else if (inner_len <= 16384) {
    const int UNROLL = 16;
    const int BLOCK = 1024;
    dequantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr, inner_len,
                                          outer_len);
  } else if (inner_len <= 32768) {
    const int UNROLL = 32;
    const int BLOCK = 1024;
    dequantize_per_channel_one_pass<FType, QType, BLOCK, UNROLL, ComputeType>
        <<<GRID_SIZE, BLOCK, 0, stream>>>(fdata_ptr, qdata_ptr, scale_ptr,
                                          zero_point_ptr, redsum_ptr, inner_len,
                                          outer_len);
  }
}
template void DeQuantizePerChannelImp<float, int8_t, float>(
    float* fdata_ptr, const int8_t* qdata_ptr, const float* scale_ptr,
    const float* zero_point_ptr, const int* redsum_ptr, const int inner_len,
    const int outer_len, cudaStream_t stream);
#ifdef ENABLE_FP16
template void DeQuantizePerChannelImp<half, int8_t, float>(
    half* fdata_ptr, const int8_t* qdata_ptr, const float* scale_ptr,
    const float* zero_point_ptr, const int* redsum_ptr, const int inner_len,
    const int outer_len, cudaStream_t stream);
#endif
template void DeQuantizePerChannelImp<hie::bfloat16, int8_t, float>(
    hie::bfloat16* fdata_ptr, const int8_t* qdata_ptr, const float* scale_ptr,
    const float* zero_point_ptr, const int* redsum_ptr, const int inner_len,
    const int outer_len, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
