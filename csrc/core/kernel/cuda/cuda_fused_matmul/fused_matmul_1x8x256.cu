/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_1x8x256.cu
 */

#include <utility>

#include "cuda/hie/cuda_activation.hpp"
#include "cuda/hie/cuda_intdivider.hpp"
#include "fused_matmul_1x1x1.hpp"
#include "fused_matmul_1x8x256.hpp"

namespace hie {

namespace dynamic_quant_matmul_fused {

using std::int32_t;
using std::int8_t;
using std::uint32_t;

template <int32_t ALIGN, typename TYPE>
struct alignas(ALIGN * sizeof(TYPE)) packed_data {
  TYPE data[ALIGN];
};

/*
 * for 256 threads/
 *      - [32]. tile-k
 *          [8]. split-n
 * for single threads
 *      - [8]. align-n
 */
template <typename TYPE, template <class> class ACT, int32_t NPROC,
          int32_t ALIGN, int32_t BLOCK>
__global__ void gemm_nn_1x8x256_activation_fused_kernel(
    const int8_t* gptr_a, const int8_t* zero_a, const int32_t* reduce_a,
    const float* scale_a, const int8_t* gptr_b, const int8_t* zero_b,
    const int32_t* reduce_b, const float* scale_b, const TYPE* tbias,
    TYPE* gptr_c, TYPE alpha, TYPE beta, int32_t gemm_m, int32_t gemm_n,
    int32_t gemm_k, int32_t loop_k, int32_t grid_n) {
  constexpr int32_t NTID =
      NPROC / ALIGN;  // n-dim thread cost for single n-proc.
  constexpr int32_t KBLK = BLOCK / NTID;
  __shared__ int32_t part_kxn[KBLK][NPROC];
  using i8pd_t = packed_data<ALIGN, int8_t>;
  int32_t i32n[ALIGN];  // partial gemm in regs. n-pack.

  // block level indexing for m, n, k
  int32_t midx = blockIdx.x / grid_n;
  int32_t ngrd = blockIdx.x % grid_n;
  int32_t kblk = threadIdx.x / NTID;  // k inner block offset for load b.
  int32_t nres = threadIdx.x % NTID;  // n inner block offset.

  if (loop_k > 0) {
    int32_t nblk = ngrd * NPROC + nres * ALIGN;
    const int8_t* k0ptr_a = gptr_a + midx * gemm_k + 0;  // [m, k], k=0
    const int8_t* k0ptr_b = gptr_b + 0 * gemm_n + nblk;  // [k, n], k=0

// init local gemm[ALIGN]
#pragma unroll
    for (int32_t ur = 0; ur < ALIGN; ur++) {
      i32n[ur] = 0;
    }

    // k - loop
    int32_t nmax = nblk + ALIGN;  // every thread process ALIGN n.
    if (nmax <= gemm_n) {
      // if local thread packed-n is all legal, load a and b, and cal partial
      // gemm. k - dim  [loop, kblk, align]. if blocksize = 256, we have
      // KBLK(a.k.a 32) part per block. for each part, loop load k from kblk
      // index with KBLK stride. see ldb indexing part.
      for (int32_t loop = 0; loop < loop_k; loop++) {
        // load a with vector k.
        // inner loop basic process 1x8x8 (mxnxk) basic block. when align=8.
        // for logic simplization, use same align=8 for lda(1x8) and
        // ldb(unroll=8 1x8).
        int32_t klpx = loop * KBLK + kblk;
        i8pd_t lda = reinterpret_cast<const i8pd_t*>(k0ptr_a)[klpx];

        // load b with vector n. align-n(same as align-k currently) times. k is
        // continuous per thread.
        i8pd_t ldb[ALIGN];
#pragma unroll
        for (int32_t kur = 0; kur < ALIGN; kur++) {
          int32_t kidx = loop * KBLK * ALIGN + kblk * ALIGN + kur;
          ldb[kur] =
              reinterpret_cast<const i8pd_t*>(k0ptr_b + kidx * gemm_n)[0];
        }

// sum part gemm
// nvcc may vectorize to idp.4a.s8
#pragma unroll
        for (int32_t kur = 0; kur < ALIGN; kur++) {
#pragma unroll
          for (int32_t nur = 0; nur < ALIGN; nur++) {
            i32n[nur] += static_cast<int32_t>(lda.data[kur]) *
                         static_cast<int32_t>(ldb[kur].data[nur]);
          }
        }
      }
    }

// store to shared.
#pragma unroll
    for (int32_t nur = 0; nur < ALIGN; nur++) {
      part_kxn[kblk][nres * ALIGN + nur] = i32n[nur];
    }

    __syncthreads();
  }

  // cause current kernel is aimming to maximize grid level parallel, and
  // minimize single block cycle usage. stg c, dequant and activation is not the
  // bottleneck, we may not utilize all thread here.
  int32_t nidx = ngrd * NPROC + threadIdx.x;
  if (threadIdx.x < NPROC) {
    int32_t gemm = 0;

    // load partial and sum.
    if (loop_k > 0) {
#pragma unroll
      for (int32_t ur = 0; ur < KBLK; ur++) {
        gemm += part_kxn[ur][threadIdx.x];
      }
    }

    // k - res.
    // cause we use block size k=8x32, for some k(rare of course),
    // we may have off 256-align k for accumulation.
    if (nidx < gemm_n) {
      const int8_t* k0ptr_a = gptr_a + midx * gemm_k + 0;
      const int8_t* k0ptr_b = gptr_b + 0 * gemm_n + nidx;
      for (int32_t kidx = loop_k * KBLK * ALIGN; kidx < gemm_k; kidx++) {
        gemm += static_cast<int32_t>(k0ptr_a[kidx]) *
                static_cast<int32_t>(k0ptr_b[kidx * gemm_n]);
      }
    }

    // dequant and act
    if (nidx < gemm_n) {
      int32_t qz_a = static_cast<int32_t>(zero_a[midx]);
      int32_t qr_a = reduce_a[midx];
      float qs_a = scale_a[midx];
      int32_t qz_b = static_cast<int32_t>(zero_b[nidx]);
      int32_t qr_b = reduce_b[nidx];
      float qs_b = scale_b[nidx];
      float bias = tbias != nullptr ? static_cast<float>(tbias[nidx]) : 0.f;

      int32_t idqt = gemm_k * qz_a * qz_b - qz_a * qr_b - qr_a * qz_b;
      float fsum = bias + qs_a * qs_b * static_cast<float>(gemm + idqt);
      float fact = ACT<float>::Op(fsum);

      // store back. streaming, no l1 cache.
      TYPE tact = static_cast<TYPE>(fact);
      gptr_c[midx * gemm_n + nidx] = tact;
    }
  }
}

template <typename TYPE>
struct gemm_nn_1x1x1_activation_fused_impl {
  static int calws(std::string act_string, int m, int n, int k) { return 0; }

  static int gridn(int m, int n, int k) {
    return n % gemm_nn_1x1x1_nproc ? n / gemm_nn_1x1x1_nproc + 1
                                   : n / gemm_nn_1x1x1_nproc;
  }

  template <template <class> class ACT>
  struct gemm_nn_1x1x1_activation_fused_inner {
    void operator()(cudaStream_t stream, int m, int n, int k, float alpha,
                    float beta, const int8_t* aquant, const int8_t* azero,
                    const int32_t* areduce, const float* ascale,
                    const int8_t* bquant, const int8_t* bzero,
                    const int32_t* breduce, const float* bscale,
                    const void* bias, void* c) {
      const TYPE* tbias = static_cast<const TYPE*>(bias);
      TYPE* tc = static_cast<TYPE*>(c);

      int32_t grid_n = gridn(m, n, k);
      int32_t grid_number = grid_n * m;
      gemm_nn_1x8x256_activation_fused_kernel<TYPE, ACT, gemm_nn_1x1x1_nproc,
                                              gemm_nn_1x1x1_align,
                                              gemm_nn_1x1x1_block>
          <<<grid_number, gemm_nn_1x1x1_block, 0, stream>>>(
              aquant, azero, areduce, ascale, bquant, bzero, breduce, bscale,
              tbias, tc, static_cast<TYPE>(alpha), static_cast<TYPE>(beta), m,
              n, k, 0, grid_n);
    }
  };

  void operator()(cudaStream_t stream, std::string act_string,
                  const int8_t* aquant, const int8_t* azero,
                  const int32_t* areduce, const float* ascale,
                  const int8_t* bquant, const int8_t* bzero,
                  const int32_t* breduce, const float* bscale, const void* bias,
                  int m, int n, int k, float alpha, float beta, void* c) {
    if (act_string == "NONE") {
      gemm_nn_1x1x1_activation_fused_inner<activation::Identity>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "RELU") {
      gemm_nn_1x1x1_activation_fused_inner<activation::Relu>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "TANH") {
      gemm_nn_1x1x1_activation_fused_inner<activation::Tanh>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "GELU") {
      gemm_nn_1x1x1_activation_fused_inner<activation::Gelu>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "GELU_TANH") {
      gemm_nn_1x1x1_activation_fused_inner<activation::GeluTanh>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else {
      LOG(ERROR) << "ACTIVE FAIL. current act = " << act_string;
    }
  }
};

// 1x8x256
template <typename TYPE>
struct gemm_nn_1x8x256_activation_fused_impl {
  static int calws(std::string act_string, int m, int n, int k) {
    int32_t loop_k = k / (gemm_nn_1x8x256_k_num * gemm_nn_1x8x256_align);
    if (loop_k >= 1) {
      if (n % gemm_nn_1x8x256_align != 0 || k % gemm_nn_1x8x256_align != 0)
        return -1;
    }
    return 0;
  }

  static int gridn(int m, int n, int k) {
    return n % gemm_nn_1x8x256_nproc ? n / gemm_nn_1x8x256_nproc + 1
                                     : n / gemm_nn_1x8x256_nproc;
  }

  template <template <class> class ACT>
  struct gemm_nn_1x8x256_activation_fused_inner {
    void operator()(cudaStream_t stream, int m, int n, int k, float alpha,
                    float beta, const int8_t* aquant, const int8_t* azero,
                    const int32_t* areduce, const float* ascale,
                    const int8_t* bquant, const int8_t* bzero,
                    const int32_t* breduce, const float* bscale,
                    const void* bias, void* c) {
      const TYPE* tbias = static_cast<const TYPE*>(bias);
      TYPE* tc = static_cast<TYPE*>(c);

      int32_t loop_k = k / (gemm_nn_1x8x256_k_num * gemm_nn_1x8x256_align);
      int32_t grid_n = gridn(m, n, k);
      int32_t grid_number = grid_n * m;
      gemm_nn_1x8x256_activation_fused_kernel<TYPE, ACT, gemm_nn_1x8x256_nproc,
                                              gemm_nn_1x8x256_align,
                                              gemm_nn_1x8x256_block>
          <<<grid_number, gemm_nn_1x8x256_block, 0, stream>>>(
              aquant, azero, areduce, ascale, bquant, bzero, breduce, bscale,
              tbias, tc, static_cast<TYPE>(alpha), static_cast<TYPE>(beta), m,
              n, k, loop_k, grid_n);
    }
  };

  void operator()(cudaStream_t stream, std::string act_string,
                  const int8_t* aquant, const int8_t* azero,
                  const int32_t* areduce, const float* ascale,
                  const int8_t* bquant, const int8_t* bzero,
                  const int32_t* breduce, const float* bscale, const void* bias,
                  int m, int n, int k, float alpha, float beta, void* c) {
    if (act_string == "NONE") {
      gemm_nn_1x8x256_activation_fused_inner<activation::Identity>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "RELU") {
      gemm_nn_1x8x256_activation_fused_inner<activation::Relu>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "TANH") {
      gemm_nn_1x8x256_activation_fused_inner<activation::Tanh>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "GELU") {
      gemm_nn_1x8x256_activation_fused_inner<activation::Gelu>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else if (act_string == "GELU_TANH") {
      gemm_nn_1x8x256_activation_fused_inner<activation::GeluTanh>()(
          stream, m, n, k, alpha, beta, aquant, azero, areduce, ascale, bquant,
          bzero, breduce, bscale, bias, c);
    } else {
      LOG(ERROR) << "ACTIVE FAIL. current act = " << act_string;
    }
  }
};

}  // namespace dynamic_quant_matmul_fused

int64_t dynamicQuantMatMulActivationFused1x8x256WorkSpace(hie::DataType dtype,
                                                          std::string act,
                                                          int sm_ver,
                                                          int sm_cnt, int m,
                                                          int n, int k) {
  if (dtype == hie::DataType::FLOAT) {
    return static_cast<int64_t>(
        dynamic_quant_matmul_fused::gemm_nn_1x8x256_activation_fused_impl<
            float>::calws(act, m, n, k));
  }

#ifdef ENABLE_FP16
  if (dtype == hie::DataType::FLOAT16) {
    return static_cast<int64_t>(
        dynamic_quant_matmul_fused::gemm_nn_1x8x256_activation_fused_impl<
            half>::calws(act, m, n, k));
  }
#endif

  return -1;
}

void dynamicQuantMatMulActivationFused1x8x256Launch(
    cudaStream_t stream, hie::DataType dtype, std::string act, int sm_ver,
    int sm_cnt, int m, int n, int k, float alpha, float beta,
    const int8_t* aquant, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bquant, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const void* bias, void* c) {
  if (dtype == hie::DataType::FLOAT) {
    dynamic_quant_matmul_fused::gemm_nn_1x8x256_activation_fused_impl<float>()(
        stream, act, aquant, azero, areduce, ascale, bquant, bzero, breduce,
        bscale, bias, m, n, k, alpha, beta, c);
    return;
  }

#ifdef ENABLE_FP16
  if (dtype == hie::DataType::FLOAT16) {
    dynamic_quant_matmul_fused::gemm_nn_1x8x256_activation_fused_impl<half>()(
        stream, act, aquant, azero, areduce, ascale, bquant, bzero, breduce,
        bscale, bias, m, n, k, alpha, beta, c);
    return;
  }
#endif

  return;
}

int64_t dynamicQuantMatMulActivationFused1x1x1WorkSpace(hie::DataType dtype,
                                                        std::string act,
                                                        int sm_ver, int sm_cnt,
                                                        int m, int n, int k) {
  if (dtype == hie::DataType::FLOAT) {
    return static_cast<int64_t>(
        dynamic_quant_matmul_fused::gemm_nn_1x1x1_activation_fused_impl<
            float>::calws(act, m, n, k));
  }

#ifdef ENABLE_FP16
  if (dtype == hie::DataType::FLOAT16) {
    return static_cast<int64_t>(
        dynamic_quant_matmul_fused::gemm_nn_1x1x1_activation_fused_impl<
            half>::calws(act, m, n, k));
  }
#endif

  return -1;
}

void dynamicQuantMatMulActivationFused1x1x1Launch(
    cudaStream_t stream, hie::DataType dtype, std::string act, int sm_ver,
    int sm_cnt, int m, int n, int k, float alpha, float beta,
    const int8_t* aquant, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bquant, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const void* bias, void* c) {
  if (dtype == hie::DataType::FLOAT) {
    dynamic_quant_matmul_fused::gemm_nn_1x1x1_activation_fused_impl<float>()(
        stream, act, aquant, azero, areduce, ascale, bquant, bzero, breduce,
        bscale, bias, m, n, k, alpha, beta, c);
    return;
  }

#ifdef ENABLE_FP16
  if (dtype == hie::DataType::FLOAT16) {
    dynamic_quant_matmul_fused::gemm_nn_1x1x1_activation_fused_impl<half>()(
        stream, act, aquant, azero, areduce, ascale, bquant, bzero, breduce,
        bscale, bias, m, n, k, alpha, beta, c);
    return;
  }
#endif

  return;
}

}  // namespace hie
