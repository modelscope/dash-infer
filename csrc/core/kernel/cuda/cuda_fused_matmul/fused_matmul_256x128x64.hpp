/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fused_matmul_256x128x64.hpp
 */

#ifndef __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_256x128x64_HPP__
#define __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_256x128x64_HPP__

#include <cstdint>
#include <cstdlib>

#include "cuda/cuda_common.h"
#include "cuda/hie/cuda_intdivider.hpp"

namespace hie {

namespace dynamic_quant_matmul_fused {

// activation
template <typename FT, template <class> class ACT>
__global__
    __launch_bounds__(256) void gemm_nn_256x128x64_activation_fused_kernel(
        const int8_t* aqptr, const int8_t* azero, const int32_t* areduce,
        const float* ascale, const int8_t* bqptr, const int8_t* bzero,
        const int32_t* breduce, const float* bscale, int32_t gemm_m,
        int32_t gemm_n, int32_t gemm_k, float alpha, float beta,
        uint32_t full_wave_blocks, uint32_t wave_y,
        hie::internal::IntDivModer<uint32_t> wave_size_divmod,
        hie::internal::IntDivModer<uint32_t> wave_y_divmod,
        hie::internal::IntDivModer<uint32_t> last_wave_y_divmod, const FT* bias,
        FT* cgptr);

// bias only
template <typename FT>
__global__ __launch_bounds__(256) void gemm_nn_256x128x64_fused_kernel(
    const int8_t* aqptr, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bqptr, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const FT* bias, FT* cgptr,
    int32_t gemm_m, int32_t gemm_n, int32_t gemm_k, float alpha, float beta,
    uint32_t full_wave_blocks, uint32_t wave_y,
    hie::internal::IntDivModer<uint32_t> wave_size_divmod,
    hie::internal::IntDivModer<uint32_t> wave_y_divmod,
    hie::internal::IntDivModer<uint32_t> last_wave_y_divmod);

}  // namespace dynamic_quant_matmul_fused

int64_t dynamicQuantMatMulActivationFused256x128x64WorkSpace(
    hie::DataType dtype, std::string act, int sm_ver, int sm_cnt, int m, int n,
    int k);

void dynamicQuantMatMulActivationFused256x128x64Launch(
    cudaStream_t stream, hie::DataType dtype, std::string act, int sm_ver,
    int sm_cnt, int m, int n, int k, float alpha, float beta,
    const int8_t* aquant, const int8_t* azero, const int32_t* areduce,
    const float* ascale, const int8_t* bquant, const int8_t* bzero,
    const int32_t* breduce, const float* bscale, const void* bias, void* c);

}  // namespace hie

#endif  // __DYNAMIC_QUANTIZE_MATMUL_CUDA_FUSED_MATMUL_FUSED_MATMUL_256x128x64_HPP__
