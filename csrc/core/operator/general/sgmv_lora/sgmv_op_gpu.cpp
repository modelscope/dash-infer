/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sgmv_op_gpu.cpp
 */

#ifdef ENABLE_CUDA
#include <common/common.h>
#include <core/kernel/cuda/cuda_common.h>
#include <core/kernel/kernel.h>
#include <core/operator/operator.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#include <core/kernel/cuda/hie/cuda_activation.hpp>
#ifdef ENABLE_FP16
#ifdef ENABLE_CUDA
#include <cuda_fp16.h>
#else
#include <common/float16.h>
#endif
#endif
#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

namespace allspark {
AsStatus sgmv_cutlass(DataType dtype, void* out, const void* in,
                      const AsTensor* weight_ptrs, const AsTensor* segments,
                      const AsTensor* ranks, void* buf, int d_in, int d_out,
                      bool is_k_tensor, bool is_n_tensor, int num_problems,
                      bool unsplit, int unsplit_n, int max_rank, int CC,
                      const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(in);
    const T** typed_weight_ptrs =
        static_cast<const T**>(weight_ptrs->GetDataPtr());
    const int32_t* typed_segments =
        static_cast<const int32_t*>(segments->GetDataPtr());
    const int32_t* typed_ranks =
        static_cast<const int32_t*>(ranks->GetDataPtr());
    cuda::SgmvCutlass<T>(typed_out, typed_input, typed_weight_ptrs,
                         typed_segments, typed_ranks, buf, is_k_tensor,
                         is_n_tensor, num_problems, d_in, d_out, unsplit,
                         unsplit_n, max_rank, CC, cu_stream);
  };
  // dispatch
  switch (dtype) {
#ifdef ENABLE_FP16
    case DataType::FLOAT16: {
      std::forward<decltype(functor)>(functor).template operator()<half>();
      break;
    }
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16: {
      std::forward<decltype(functor)>(functor)
          .template operator()<hie::bfloat16>();
      break;
    }
#endif
    default: {
      LOG(ERROR) << "unsupported datatype " << DataType_Name(dtype)
                 << " for SgmvCutlass";
      throw AsException("ALLSPARK_RUNTIME_ERROR");
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus sgmv_split_qkv(DataType dtype, AsTensor* out_ptrs, const void* in,
                        const AsTensor* segments, const AsTensor* lora_B_ranks,
                        int max_rank, int num_problems,
                        const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  auto functor = [&]<typename T>() {
    T** typed_out_ptrs = static_cast<T**>(out_ptrs->GetDataPtr());
    const T* typed_input = static_cast<const T*>(in);
    const int32_t* typed_segments =
        static_cast<const int32_t*>(segments->GetDataPtr());
    const int32_t* typed_lora_B_ranks =
        static_cast<const int32_t*>(lora_B_ranks->GetDataPtr());
    cuda::SgmvSplitQKV<T>(typed_out_ptrs, typed_input, typed_segments,
                          typed_lora_B_ranks, max_rank, num_problems,
                          cu_stream);
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
}  // namespace allspark
#endif
