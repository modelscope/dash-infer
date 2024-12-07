/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_inplace_softmax.cu
 */

#include "../attention/softmax.hpp"
#include "attn_kv_cache.hpp"
#include "kernel_utils.cuh"

// single block for [batch, nhead]
// score            [batch, nhead, xseql]

namespace allspark {
namespace cuda {
namespace mha_quant_cache {
namespace softmax {
constexpr int32_t basic_warp_num = 8;
template <typename FT, int32_t LOOP>
__global__ void inplace_softmax_kernel(FT* score, int32_t batch, int32_t nhead,
                                       int32_t xseql) {
  __shared__ float warp_max[basic_warp_num];
  __shared__ float warp_sum[basic_warp_num];
  constexpr float init_value = static_cast<float>(-INFINITY);
  int32_t warp = threadIdx.x / utils::warp_size;
  int32_t lane = threadIdx.x % utils::warp_size;
  float regs[LOOP];
  float tid_max = init_value;
#pragma unroll
  for (int32_t loop = 0; loop < LOOP; loop++) {
    int32_t xidx = loop * blockDim.x + threadIdx.x;
    regs[loop] = xidx < xseql
                     ? static_cast<float>(score[blockIdx.x * xseql + xidx])
                     : init_value;
    tid_max = max(tid_max, regs[loop]);
  }
  tid_max = utils::ReduceThread<utils::MaxOp, float>(tid_max);
  if (lane == 0) warp_max[warp] = tid_max;
  __syncthreads();
#pragma unroll
  for (int32_t wpid = 0; wpid < basic_warp_num; wpid++) {
    tid_max = max(tid_max, warp_max[wpid]);
  }

  float tid_exp_sum = 0;
#pragma unroll
  for (int32_t loop = 0; loop < LOOP; loop++) {
    int32_t xidx = loop * blockDim.x + threadIdx.x;
    if (xidx < xseql) {
      regs[loop] = __expf(regs[loop] - tid_max);
      tid_exp_sum += regs[loop];
    }
  }
  tid_exp_sum = utils::ReduceThread<utils::SumOp, float>(tid_exp_sum);
  if (lane == 0) warp_sum[warp] = tid_exp_sum;
  __syncthreads();
  float block_exp_sum = 1e-12;
#pragma unroll
  for (int32_t wpid = 0; wpid < basic_warp_num; wpid++) {
    block_exp_sum += warp_sum[wpid];
  }
  float tid_exp_div = __fdividef(1.f, block_exp_sum);

#pragma unroll
  for (int32_t loop = 0; loop < LOOP; loop++) {
    int32_t xidx = loop * blockDim.x + threadIdx.x;
    if (xidx < xseql) {
      score[blockIdx.x * xseql + xidx] =
          static_cast<FT>(regs[loop] * tid_exp_div);
    }
  }
}

template <typename TYPE>
struct inplace_softmax_basic_impl {
  // score        [batch, nhead, xseql]
  void operator()(cudaStream_t stream, void* score, int32_t batch,
                  int32_t nhead, int32_t xseql) {
    TYPE* tscore = reinterpret_cast<TYPE*>(score);
    constexpr int32_t kBlock = basic_warp_num * utils::warp_size;
    int32_t kGrid = batch * nhead;
    int32_t kLoop = utils::cal_ceil(xseql, kBlock);

#define LOOP_DISPATCH(LOOP)                                              \
  if (kLoop <= LOOP) {                                                   \
    /*printf("inplace_softmax_kernel<%d><<<%d, %d>>>(batch=%d, nhead=%d, \
       xseql=%d)\n", LOOP, kGrid, kBlock, batch, nhead, xseql);*/        \
    inplace_softmax_kernel<TYPE, LOOP>                                   \
        <<<kGrid, kBlock, 0, stream>>>(tscore, batch, nhead, xseql);     \
    return;                                                              \
  }

    LOOP_DISPATCH(1);
    LOOP_DISPATCH(2);
    LOOP_DISPATCH(3);
    LOOP_DISPATCH(4);
    LOOP_DISPATCH(6);
    LOOP_DISPATCH(8);   // 2048
    LOOP_DISPATCH(16);  // 4096
    LOOP_DISPATCH(32);  // 8192
    // assert(kLoop <= 32);
    if (kLoop >= 32) {
      softmax_4d_fallback(
          stream, allspark::cuda::attention::toDataType<TYPE>::cuda_type, score,
          score, nullptr, 1.f, batch, 1, nhead, xseql, 1, 1, 1, 1, false, false,
          false, false, 0, 0);
      // LOG(ERROR) << "Not-support softmax alignment.";
    }
#undef LOOP_DISPATCH
  }
};
}  // namespace softmax

#define INPLACE_SOFTMAX_IMPL(FT)                                           \
  template <>                                                              \
  void inplace_softmax<FT>(cudaStream_t stream, FT * score, int32_t batch, \
                           int32_t nhead, int32_t xseql) {                 \
    softmax::inplace_softmax_basic_impl<FT>()(stream, score, batch, nhead, \
                                              xseql);                      \
  }

INPLACE_SOFTMAX_IMPL(float);
#if ENABLE_FP16
INPLACE_SOFTMAX_IMPL(half_t);
#endif  // ENABLE_FP16
#if ENABLE_BF16
INPLACE_SOFTMAX_IMPL(__hie_buildin::bfloat16)
#endif  // ENABLE_BF16

}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark
