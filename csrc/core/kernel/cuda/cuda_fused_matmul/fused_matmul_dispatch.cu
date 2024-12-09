/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_dispatch.cu
 */

#include <tuple>

#include "fused_matmul_1x1x1.hpp"
#include "fused_matmul_1x8x256.hpp"
#include "fused_matmul_256x128x64.hpp"
#include "fused_matmul_interface.hpp"
namespace hie {

namespace dynamic_quant_matmul_fused {

// lhs x rhs
// sm_ver, m, n, k
const std::vector<std::tuple<int, int, int, int>>
    perf_map_k1x8x256_k256x128x64 = {
        {0x0705, 32, 1024, 2048}, {0x0705, 32, 512, 65536},
        {0x0705, 12, 2048, 2048}, {0x0705, 12, 1024, 32768},
        {0x0705, 4, 2048, 16384}, {0x0705, 2, 4096, 4096},
};

const std::vector<std::tuple<int, int, int, int>> perf_map_k1x1x1_k1x8x256 = {
    // always worse.
};

const std::vector<std::tuple<int, int, int, int>> perf_map_k1x1x1_k256x128x64 =
    {
        {0x0705, 32, 256, 512},  {0x0705, 32, 64, 2048},
        {0x0705, 32, 2048, 32},  {0x0705, 4, 1024, 1024},
        {0x0705, 4, 128, 16384},
};

/**     K
 *      ^     .
 *      | false.
 *      |-------x     return false
 *      |-------|-x   rhs is better
 *      | true  | |  .
 *      |-------|-|------x
 *      | lhs is| |      |    .
 *      | better| | true | false        . different kernels meet same
 * performance
 *      + - - - - - - - - - - - > N       at given M, N, K, (m for curves),
 * lhs always has more restrict constrain.
 */
bool selectLHSKernelByPerf(
    const std::vector<std::tuple<int, int, int, int>> boundary, int sm_ver,
    int m, int n, int k) {
  for (auto btuple : boundary) {
    if (sm_ver == std::get<0>(btuple)) {
      if (m <= std::get<1>(btuple) && n <= std::get<2>(btuple) &&
          k <= std::get<3>(btuple))
        return true;
    }
  }
  return false;
}

bool selectRHSKernelByPerf(
    const std::vector<std::tuple<int, int, int, int>> boundary, int sm_ver,
    int m, int n, int k) {
  return !selectLHSKernelByPerf(boundary, sm_ver, m, n, k);
}

}  // namespace dynamic_quant_matmul_fused

dqmm_fused_kernel dynamicQuantizeMatMulFusedFindKernel(hie::DataType dtype,
                                                       std::string act,
                                                       int sm_ver, int sm_cnt,
                                                       int m, int n, int k) {
  bool kernel_256x128x64_valid =
      dynamicQuantizeMatMulWorkSpace(dqmm_fused_kernel::DQMM_FUSED_256x128x64,
                                     dtype, act, sm_ver, sm_cnt, m, n, k) >= 0;
  bool kernel_1x8x256_valid =
      dynamicQuantizeMatMulWorkSpace(dqmm_fused_kernel::DQMM_FUSED_1x8x256,
                                     dtype, act, sm_ver, sm_cnt, m, n, k) >= 0;
  bool kernel_1x1x1_valid =
      dynamicQuantizeMatMulWorkSpace(dqmm_fused_kernel::DQMM_FUSED_1x1x1, dtype,
                                     act, sm_ver, sm_cnt, m, n, k) >= 0;
  // assert(kernel_1x1x1_valid);  // always true. default.

  if (m > 32) {
    if (kernel_256x128x64_valid)
      return dqmm_fused_kernel::DQMM_FUSED_256x128x64;
    else if (kernel_1x8x256_valid)
      return dqmm_fused_kernel::DQMM_FUSED_1x8x256;
    else if (kernel_1x1x1_valid)
      return dqmm_fused_kernel::DQMM_FUSED_1x1x1;
    else
      return dqmm_fused_kernel::DQMM_FUSED_NONE;
  }

  if (m <= 32) {
    if (kernel_1x8x256_valid && kernel_256x128x64_valid) {
      if (dynamic_quant_matmul_fused::selectLHSKernelByPerf(
              dynamic_quant_matmul_fused::perf_map_k1x8x256_k256x128x64, sm_ver,
              m, n, k)) {
        return dqmm_fused_kernel::DQMM_FUSED_1x8x256;
      } else {
        return dqmm_fused_kernel::DQMM_FUSED_256x128x64;
      }
    } else if (kernel_1x1x1_valid && kernel_256x128x64_valid) {
      if (dynamic_quant_matmul_fused::selectLHSKernelByPerf(
              dynamic_quant_matmul_fused::perf_map_k1x1x1_k256x128x64, sm_ver,
              m, n, k)) {
        return dqmm_fused_kernel::DQMM_FUSED_1x1x1;
      } else {
        return dqmm_fused_kernel::DQMM_FUSED_256x128x64;
      }
      // } else if (kernel_256x128x64_valid) {
      //     // equals to kernel_1x1x1_valid && kernel_256x128x64_valid
      //     return dqmm_fused_kernel::DQMM_FUSED_256x128x64;
    } else if (kernel_1x1x1_valid && kernel_1x8x256_valid) {
      if (dynamic_quant_matmul_fused::selectLHSKernelByPerf(
              dynamic_quant_matmul_fused::perf_map_k1x1x1_k1x8x256, sm_ver, m,
              n, k)) {
        return dqmm_fused_kernel::DQMM_FUSED_1x1x1;
      } else {
        return dqmm_fused_kernel::DQMM_FUSED_1x8x256;
      }
    } else if (kernel_1x8x256_valid) {
      return dqmm_fused_kernel::DQMM_FUSED_1x8x256;
    } else {
      return dqmm_fused_kernel::DQMM_FUSED_1x1x1;
    }
  }

  return dqmm_fused_kernel::DQMM_FUSED_1x1x1;
}

// if invalid return -1.
int64_t dynamicQuantizeMatMulWorkSpace(dqmm_fused_kernel kernel,
                                       hie::DataType dtype, std::string act,
                                       int sm_ver, int sm_cnt, int m, int n,
                                       int k) {
  switch (kernel) {
    case dqmm_fused_kernel::DQMM_FUSED_1x1x1:
      return dynamicQuantMatMulActivationFused1x1x1WorkSpace(dtype, act, sm_ver,
                                                             sm_cnt, m, n, k);
    case dqmm_fused_kernel::DQMM_FUSED_1x8x256:
      return dynamicQuantMatMulActivationFused1x8x256WorkSpace(
          dtype, act, sm_ver, sm_cnt, m, n, k);
    case dqmm_fused_kernel::DQMM_FUSED_256x128x64:
      return dynamicQuantMatMulActivationFused256x128x64WorkSpace(
          dtype, act, sm_ver, sm_cnt, m, n, k);
    case dqmm_fused_kernel::DQMM_FUSED_NONE:
    default:
      return -1;
  }
}

void dynamicQuantizeMatMulLaunch(cudaStream_t stream, dqmm_fused_kernel kernel,
                                 hie::DataType dtype, std::string act,
                                 int sm_ver, int sm_cnt, int m, int n, int k,
                                 float alpha, float beta, const int8_t* aquant,
                                 const int8_t* azero, const int32_t* areduce,
                                 const float* ascale, const int8_t* bquant,
                                 const int8_t* bzero, const int32_t* breduce,
                                 const float* bscale, const void* bias,
                                 void* c) {
  if (dynamicQuantizeMatMulWorkSpace(kernel, dtype, act, sm_ver, sm_cnt, m, n,
                                     k) < 0) {
    return;
  }

  switch (kernel) {
    case dqmm_fused_kernel::DQMM_FUSED_1x1x1:
      return dynamicQuantMatMulActivationFused1x1x1Launch(
          stream, dtype, act, sm_ver, sm_cnt, m, n, k, alpha, beta, aquant,
          azero, areduce, ascale, bquant, bzero, breduce, bscale, bias, c);
    case dqmm_fused_kernel::DQMM_FUSED_1x8x256:
      return dynamicQuantMatMulActivationFused1x8x256Launch(
          stream, dtype, act, sm_ver, sm_cnt, m, n, k, alpha, beta, aquant,
          azero, areduce, ascale, bquant, bzero, breduce, bscale, bias, c);
    case dqmm_fused_kernel::DQMM_FUSED_256x128x64:
      return dynamicQuantMatMulActivationFused256x128x64Launch(
          stream, dtype, act, sm_ver, sm_cnt, m, n, k, alpha, beta, aquant,
          azero, areduce, ascale, bquant, bzero, breduce, bscale, bias, c);
    case dqmm_fused_kernel::DQMM_FUSED_NONE:
    default:
      return;
  }
}

}  // namespace hie
