
/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    batched_gemv.cu
 */

#include <cassert>
#include <cstdint>
#include <limits>

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "utility/check_cuda.h"
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <cassert>
#include <cstdint>

namespace allspark {
namespace cuda {

template <typename T>
struct DivModT {
  T div;
  T mod;
  __host__ __device__ DivModT(T d, T m) : div(d), mod(m) {}
};

struct U32DivMod {
  uint32_t d_;
  uint32_t magic_;
  uint32_t shift_;

  U32DivMod() {}

  explicit U32DivMod(uint32_t d) : d_(d) {
    assert(d >= 1 && d <= INT32_MAX);

    for (shift_ = 0; shift_ < 32; ++shift_) {
      if ((1u << shift_) >= d) break;
    }
    uint64_t tmp_magic = ((1lu << 32) * ((1lu << shift_) - d)) / d + 1;
    magic_ = tmp_magic;  // copy lower 32-bit

    assert(magic_ != 0 && magic_ == tmp_magic);
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) {
    return (__umulhi(n, magic_) + n) >> shift_;
  }

  __device__ __forceinline__ DivModT<uint32_t> DivMod(uint32_t n) {
    uint32_t d = Div(n);
    return DivModT<uint32_t>(d, n - d_ * d);
  }
};

namespace shfl_helper {

template <int SIZE>
struct ShflUIntT {
  using type = uint32_t;
};

template <>
struct ShflUIntT<1> {
  using type = uint8_t;
};

template <>
struct ShflUIntT<2> {
  using type = uint16_t;
};

__device__ __forceinline__ uint32_t ShflBfly(uint32_t mask, uint32_t var,
                                             uint32_t laneMask,
                                             uint32_t width) {
  const uint32_t SHFL_C = ((32 - width) << 8) | (32 - 1);

  uint32_t ret;
  asm volatile(
#if __CUDACC_VER_MAJOR__ < 9
      "shfl.bfly.b32 %0, %1, %2, %3;\n"
#else
      "shfl.sync.bfly.b32 %0, %1, %2, %3, %4;\n"
#endif
      : "=r"(ret)
      : "r"(var), "r"(laneMask), "r"(SHFL_C), "r"(mask));
  return ret;
}

}  // namespace shfl_helper

template <typename T>
__device__ __forceinline__ T ShflBfly(uint32_t mask, const T& var,
                                      uint32_t laneMask, uint32_t width) {
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0, "");

  T ret;
  using ShflT = typename shfl_helper::ShflUIntT<sizeof(T)>::type;
  const ShflT* x = reinterpret_cast<const ShflT*>(&var);
  ShflT* y = reinterpret_cast<ShflT*>(&ret);

#pragma unroll
  for (int i = 0; i < sizeof(T) / sizeof(ShflT); ++i) {
    y[i] = shfl_helper::ShflBfly(mask, static_cast<uint32_t>(x[i]), laneMask,
                                 width);
  }

  return ret;
}

template <typename T, int N>
struct alignas(N * sizeof(T)) PackT {
  T data[N];
};

template <int CTA_X, int CTA_Y, int PACK, int UNROLL, bool USE_WS, typename CT,
          typename DT>
__global__ __launch_bounds__(CTA_X* CTA_Y) void BatchedGEMVKernel(
    const DT* const* xArray, const DT* const* matrixArray, DT* const* yArray,
    DT* ws, int k, int n, int ldMatrix, U32DivMod vGridXDivMod,
    U32DivMod vGridYDivMod) {
  static_assert((CTA_X & (CTA_X - 1)) == 0, "CTA_X must be power of 2");
  static_assert((CTA_Y & (CTA_Y - 1)) == 0, "CTA_Y must be power of 2");
  static_assert(CTA_X * CTA_Y >= 32, "");

  __shared__ CT smem[PACK * CTA_Y * CTA_X];

  auto vGridXDm = vGridXDivMod.DivMod(blockIdx.x);
  auto vGridYDm = vGridYDivMod.DivMod(vGridXDm.div);
  int blockIdX = vGridXDm.mod;
  int blockIdY = vGridYDm.mod;
  int blockIdZ = vGridYDm.div;

  int batch = blockIdZ;
  const DT* x;
  const DT* matrix;
  asm("ld.global.nc.b64 %0, [%2];\n"
      "ld.global.nc.b64 %1, [%3];"
      : "=l"(x), "=l"(matrix)
      : "l"(xArray + batch), "l"(matrixArray + batch));

  int tidX = threadIdx.x % CTA_X;
  int tidY = threadIdx.x / CTA_X;
  int idxX = (blockIdX * CTA_X + tidX) * PACK;
  int idxY = blockIdY * CTA_Y * UNROLL + tidY;

  using PT = PackT<DT, PACK>;
  const DT* __restrict__ xPtr = x + idxY;
  const DT* matrixPtr = matrix + idxY * ldMatrix + idxX;

  DT xReg[UNROLL];
  PT matrixReg[UNROLL];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xReg[i] = DT(0);
  }
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
    for (int j = 0; j < PACK; ++j) {
      matrixReg[i].data[j] = DT(0);
    }
  }

  // load input vector and matrix
  bool matrixInBound = idxX < n;
  if (blockIdY * CTA_Y * UNROLL + CTA_Y * UNROLL <= k) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      xReg[i] = xPtr[i * CTA_Y];
      if (matrixInBound) {
        matrixReg[i] =
            *reinterpret_cast<const PT*>(matrixPtr + i * ldMatrix * CTA_Y);
      }
    }
  } else {
    int nLdg = k > idxY ? (k - idxY + CTA_Y - 1) / CTA_Y : 0;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (i < nLdg) {
        xReg[i] = xPtr[i * CTA_Y];
        if (matrixInBound) {
          matrixReg[i] =
              *reinterpret_cast<const PT*>(matrixPtr + i * ldMatrix * CTA_Y);
        }
      }
    }
  }

  // convert input data to compute type
  CT xCReg[UNROLL];
  CT matrixCReg[UNROLL][PACK];
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    xCReg[i] = static_cast<CT>(xReg[i]);
    for (int j = 0; j < PACK; ++j) {
      matrixCReg[i][j] = static_cast<CT>(matrixReg[i].data[j]);
    }
  }

  // dot product
  CT dot[PACK];
#pragma unroll
  for (int i = 0; i < PACK; ++i) {
    dot[i] = xCReg[0] * matrixCReg[0][i];
  }
#pragma unroll
  for (int i = 1; i < UNROLL; ++i) {
    for (int j = 0; j < PACK; ++j) {
      dot[j] += xCReg[i] * matrixCReg[i][j];
    }
  }

  // block reduce
  if (CTA_Y > 1) {
#pragma unroll
    for (int i = 0; i < PACK; ++i) {
      smem[threadIdx.x + i * CTA_Y * CTA_X] = dot[i];
    }
    __syncthreads();

    const int REDUCE_THREADS_X = CTA_X;
    const int REDUCE_THREADS_Y = CTA_X < 32 ? 32 / CTA_X : 1;
    const int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

    if (threadIdx.x < REDUCE_THREADS) {
#pragma unroll
      for (int i = 0; i < PACK; ++i) {
        dot[i] = smem[threadIdx.x + i * CTA_Y * CTA_X];
      }
#pragma unroll
      for (int i = 1; i < CTA_Y / REDUCE_THREADS_Y; ++i) {
#pragma unroll
        for (int j = 0; j < PACK; ++j) {
          dot[j] += smem[threadIdx.x + i * REDUCE_THREADS + j * CTA_Y * CTA_X];
        }
      }

      if (REDUCE_THREADS_Y > 1) {
#pragma unroll
        for (int i = 32; i >= REDUCE_THREADS_X; i /= 2) {
#pragma unroll
          for (int j = 0; j < PACK; ++j) {
            dot[j] += ShflBfly(0xffffffff, dot[j], i / 2, 32);
          }
        }
      }
    }
  }

  // convert to DT and store the output vector
  if (threadIdx.x < CTA_X && matrixInBound) {
    DT* stgPtr;
    if (USE_WS) {
      stgPtr = ws + (vGridXDm.div * n + idxX);
    } else {
      DT* y;
      asm("ld.global.nc.b64 %0, [%1];" : "=l"(y) : "l"(yArray + batch));
      stgPtr = y + idxX;
    }

    PT yReg;
#pragma unroll
    for (int i = 0; i < PACK; ++i) {
      yReg.data[i] = dot[i];
    }
    *reinterpret_cast<PT*>(stgPtr) = yReg;
  }
}

template <int CTA_X, int CTA_Y, int PACK, int UNROLL, typename CT, typename DT>
__global__ __launch_bounds__(CTA_X* CTA_Y) void BatchedGEMVReduceKernel(
    const DT* ws, DT* const* yArray, int reduceLength, int n,
    U32DivMod vGridXDivMod) {
  static_assert((CTA_X & (CTA_X - 1)) == 0, "CTA_X must be power of 2");
  static_assert((CTA_Y & (CTA_Y - 1)) == 0, "CTA_Y must be power of 2");
  static_assert(CTA_X * CTA_Y >= 32, "");

  __shared__ CT smem[PACK * CTA_Y * CTA_X];

  auto vGridXDm = vGridXDivMod.DivMod(blockIdx.x);
  int blockIdX = vGridXDm.mod;
  int blockIdY = vGridXDm.div;

  int tidX = threadIdx.x % CTA_X;
  int tidY = threadIdx.x / CTA_X;
  int batch = blockIdY;
  int idxX = (blockIdX * CTA_X + tidX) * PACK;

  using PT = PackT<DT, PACK>;
  const PT* wsPtr = reinterpret_cast<const PT*>(
      ws + (batch * reduceLength + tidY) * n + idxX);
  int nFullTile = reduceLength / (CTA_Y * UNROLL);
  int lastTile = reduceLength % (CTA_Y * UNROLL);
  bool wsInBound = idxX < n;

  PT wsReg[UNROLL];
  CT wsCReg[UNROLL][PACK];
  CT acc[PACK];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0;
  }

  // fully unrolled tile loop
  for (; nFullTile > 0; --nFullTile) {
// load the input
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      if (wsInBound) {
        wsReg[i] = wsPtr[i * CTA_Y * n];
      }
    }

// convert input data to compute type
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK; ++j) {
        wsCReg[i][j] = static_cast<CT>(wsReg[i].data[j]);
      }
    }

// thread local reduce
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK; ++j) {
        acc[j] += wsCReg[i][j];
      }
    }

    wsPtr += CTA_Y * UNROLL * n;
  }

  if (lastTile != 0) {
    int nLdg = lastTile > tidY ? (lastTile - tidY + CTA_Y - 1) / CTA_Y : 0;

    if (wsInBound) {
// load the input
#pragma unroll
      for (int i = 0; i < UNROLL; ++i) {
        if (i < nLdg) {
          wsReg[i] = wsPtr[i * CTA_Y * n];
        } else {
#pragma unroll
          for (int j = 0; j < PACK; ++j) {
            wsReg[i].data[j] = DT(0);
          }
        }
      }
    }

// convert input data to compute type
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK; ++j) {
        wsCReg[i][j] = static_cast<CT>(wsReg[i].data[j]);
      }
    }

// thread local reduce
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
#pragma unroll
      for (int j = 0; j < PACK; ++j) {
        acc[j] += wsCReg[i][j];
      }
    }
  }

  // block reduce
  if (CTA_Y > 1) {
#pragma unroll
    for (int i = 0; i < PACK; ++i) {
      smem[threadIdx.x + i * CTA_Y * CTA_X] = acc[i];
    }
    __syncthreads();

    const int REDUCE_THREADS_X = CTA_X;
    const int REDUCE_THREADS_Y = CTA_X < 32 ? 32 / CTA_X : 1;
    const int REDUCE_THREADS = REDUCE_THREADS_X * REDUCE_THREADS_Y;

    if (threadIdx.x < REDUCE_THREADS) {
#pragma unroll
      for (int i = 0; i < PACK; ++i) {
        acc[i] = smem[threadIdx.x + i * CTA_Y * CTA_X];
      }
#pragma unroll
      for (int i = 1; i < CTA_Y / REDUCE_THREADS_Y; ++i) {
#pragma unroll
        for (int j = 0; j < PACK; ++j) {
          acc[j] += smem[threadIdx.x + i * REDUCE_THREADS + j * CTA_Y * CTA_X];
        }
      }

      if (REDUCE_THREADS_Y > 1) {
#pragma unroll
        for (int i = 32; i >= REDUCE_THREADS_X; i /= 2) {
#pragma unroll
          for (int j = 0; j < PACK; ++j) {
            acc[j] += ShflBfly(0xffffffff, acc[j], i / 2, 32);
          }
        }
      }
    }
  }

  // convert to DT and store the output vector
  if (threadIdx.x < CTA_X && wsInBound) {
    DT* y;
    asm("ld.global.nc.b64 %0, [%1];" : "=l"(y) : "l"(yArray + batch));
    DT* stgPtr = y + idxX;

    PT yReg;
#pragma unroll
    for (int i = 0; i < PACK; ++i) {
      yReg.data[i] = acc[i];
    }
    *reinterpret_cast<PT*>(stgPtr) = yReg;
  }
}

template <typename CT, typename DT>
struct KernelLauncher {
  KernelLauncher(int k_, int n_, int batch_) {
    k = k_;
    n = n_;
    batch = batch_;
    int nByte = n * sizeof(DT);

    if (nByte >= 768 || nByte % 256 == 0) {
      DispatchPack<256>();
    } else if (nByte >= 384 || nByte % 128 == 0) {
      DispatchPack<128>();
    } else if (nByte >= 192 || nByte % 64 == 0) {
      DispatchPack<64>();
    } else {
      DispatchPack<32>();
    }
  }

  template <int MEM_COHERENCE>
  void DispatchPack() {
    int nByte = n * sizeof(DT);
    if (nByte % 16 == 0) {
      DispatchKernel<MEM_COHERENCE, 16 / sizeof(DT)>();
    } else if (nByte % 8 == 0) {
      DispatchKernel<MEM_COHERENCE, 8 / sizeof(DT)>();
    } else {
      DispatchKernel<32 * sizeof(DT), 1>();
    }
  }

  template <int MEM_COHERENCE, int PACK>
  void DispatchKernel() {
    static_assert(MEM_COHERENCE <= GEMV_CTA * PACK * sizeof(DT), "");
    static_assert(MEM_COHERENCE <= REDUCE_CTA * PACK * sizeof(DT), "");

    const int CTA_X = MEM_COHERENCE / (PACK * sizeof(DT));
    const int GEMV_CTA_Y = GEMV_CTA / CTA_X;
    const int REDUCE_CTA_Y = REDUCE_CTA / CTA_X;

    const int TILE_N = CTA_X * PACK;
    const int GEMV_TILE_K = GEMV_CTA_Y * UNROLL;

    gridX = (n + TILE_N - 1) / TILE_N;
    gemvGridY = (k + GEMV_TILE_K - 1) / GEMV_TILE_K;

    if (gemvGridY > 1) {
      gemvKernel =
          BatchedGEMVKernel<CTA_X, GEMV_CTA_Y, PACK, UNROLL, true, CT, DT>;
      wsSize = n * gemvGridY * batch * sizeof(DT);
    } else {
      gemvKernel =
          BatchedGEMVKernel<CTA_X, GEMV_CTA_Y, PACK, UNROLL, false, CT, DT>;
      wsSize = 0;
    }

    reduceKernel =
        BatchedGEMVReduceKernel<CTA_X, REDUCE_CTA_Y, PACK, UNROLL, CT, DT>;
  }

  int Launch(const DT* const* xArray, const DT* const* matrixArray,
             int ldMatrix, DT* const* yArray, void* ws, size_t wsSize,
             cudaStream_t stream) {
    if (wsSize < wsSize) {
      return -1;
    }
    DT* wsDT = static_cast<DT*>(ws);

    int gemvGrid = gridX * gemvGridY * batch;
    gemvKernel<<<gemvGrid, GEMV_CTA, 0, stream>>>(
        xArray, matrixArray, yArray, wsDT, k, n, ldMatrix, U32DivMod(gridX),
        U32DivMod(gemvGridY));

    if (gemvGridY > 1) {
      int reduceGrid = gridX * batch;
      reduceKernel<<<reduceGrid, REDUCE_CTA, 0, stream>>>(
          wsDT, yArray, gemvGridY, n, U32DivMod(gridX));
    }

    if (cudaGetLastError() != cudaSuccess) {
      return -1;
    } else {
      return 0;
    }
  }

  void (*gemvKernel)(const DT* const*, const DT* const*, DT* const*, DT*, int,
                     int, int, U32DivMod, U32DivMod) = nullptr;
  void (*reduceKernel)(const DT*, DT* const*, int, int, U32DivMod) = nullptr;

  // kernel configuration
  static const int GEMV_CTA = 512;
  static const int REDUCE_CTA = 256;
  static const int UNROLL = 8;

  int gridX = 0;
  int gemvGridY = 0;

  int k = 0;
  int n = 0;
  int batch = 0;

  size_t wsSize = 0;
};

template <typename CT, typename DT>
size_t GetBatchedGEMVWorkspaceSize(int k, int n, int batchSize) {
  KernelLauncher<CT, DT> launcher(k, n, batchSize);
  return launcher.wsSize;
}

/**
 * CT: type (or precision) for computation
 * DT: type of the input vector, matrix and output vector
 *
 * k: number of rows of matrix, and the size of input vector
 * n: number of columns of matrix, and the size of output vector
 * xArray: array of pointers to the input vector
 * matrixArray: array of pointers to the matrix
 * ldMatrix: leading dimension of the matrix (row-major)
 * yArray: array of pointers to the output array
 */
template <typename CT, typename DT>
int BatchedGEMV(int k, int n, int batchSize, const DT* const* xArray,
                const DT* const* matrixArray, int ldMatrix, DT* const* yArray,
                void* ws, size_t wsSize, cudaStream_t stream) {
  KernelLauncher<CT, DT> launcher(k, n, batchSize);
  return launcher.Launch(xArray, matrixArray, ldMatrix, yArray, ws, wsSize,
                         stream);
}

template size_t GetBatchedGEMVWorkspaceSize<float, float>(int k, int n,
                                                          int batchSize);
#ifdef ENABLE_FP16
template size_t GetBatchedGEMVWorkspaceSize<half, half>(int k, int n,
                                                        int batchSize);
#endif
#ifdef ENABLE_BF16
template size_t GetBatchedGEMVWorkspaceSize<hie::bfloat16, hie::bfloat16>(
    int k, int n, int batchSize);
#endif
template int BatchedGEMV<float, float>(int k, int n, int batchSize,
                                       const float* const* xArray,
                                       const float* const* matrixArray,
                                       int ldMatrix, float* const* yArray,
                                       void* ws, size_t wsSize,
                                       cudaStream_t stream);
#ifdef ENABLE_FP16
template int BatchedGEMV<half, half>(int k, int n, int batchSize,
                                     const half* const* xArray,
                                     const half* const* matrixArray,
                                     int ldMatrix, half* const* yArray,
                                     void* ws, size_t wsSize,
                                     cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template int BatchedGEMV<hie::bfloat16, hie::bfloat16>(
    int k, int n, int batchSize, const hie::bfloat16* const* xArray,
    const hie::bfloat16* const* matrixArray, int ldMatrix,
    hie::bfloat16* const* yArray, void* ws, size_t wsSize, cudaStream_t stream);
#endif
}  // namespace cuda
}  // namespace allspark
