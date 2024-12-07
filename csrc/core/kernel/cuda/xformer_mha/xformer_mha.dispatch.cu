/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    xformer_mha.dispatch.cu
 */

#include "xformer_mha.h"

#ifdef ENABLE_CUDA
// CUDA use standard cutlass
#include "xformer_kernel.cuh"
namespace allspark {
namespace cuda {

size_t xformer_prefill_attention_workspace_inbytes(const xformer_t& param) {
#define DispatchXformerWSS(CType, CArch)                                    \
  if (param.dtype == xformer_utils::CutlassType2DataType<CType>::dt &&      \
      param.sm_version >= xformer_utils::CutlassArch2Int<CArch>::version) { \
    return xformer_prefill_attention_workspace_inbytes_dispatch<            \
        CType, CArch, kBlockQuery, kBlockKey, kMaxK>(param);                \
  }

  //  CUDA
  if (param.phead <= 64) {
    constexpr int kBlockQuery = 64;
    constexpr int kBlockKey = 64;
    constexpr int kMaxK = 64;
    DispatchXformerWSS(float, cutlass::arch::Sm80);
    DispatchXformerWSS(float, cutlass::arch::Sm75);
    DispatchXformerWSS(float, cutlass::arch::Sm70);
#ifdef ENABLE_FP16
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm80);
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm75);
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm70);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    DispatchXformerWSS(cutlass::bfloat16_t, cutlass::arch::Sm80);
    DispatchXformerWSS(cutlass::bfloat16_t, cutlass::arch::Sm75);
    /* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
#endif  // ENABLE_BF16
  } else if (param.phead <= 128) {
    constexpr int kBlockQuery = 32;
    constexpr int kBlockKey = 128;
    constexpr int kMaxK = 128;
    DispatchXformerWSS(float, cutlass::arch::Sm80);
    DispatchXformerWSS(float, cutlass::arch::Sm75);
    DispatchXformerWSS(float, cutlass::arch::Sm70);
#ifdef ENABLE_FP16
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm80);
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm75);
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm70);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    DispatchXformerWSS(cutlass::bfloat16_t, cutlass::arch::Sm80);
    DispatchXformerWSS(cutlass::bfloat16_t, cutlass::arch::Sm75);
    /* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
#endif  // ENABLE_BF16
  } else {
    constexpr int kBlockQuery = 32;
    constexpr int kBlockKey = 128;
    constexpr int kMaxK = 65536;
    DispatchXformerWSS(float, cutlass::arch::Sm80);
    DispatchXformerWSS(float, cutlass::arch::Sm75);
    DispatchXformerWSS(float, cutlass::arch::Sm70);
#ifdef ENABLE_FP16
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm80);
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm75);
    DispatchXformerWSS(cutlass::half_t, cutlass::arch::Sm70);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    DispatchXformerWSS(cutlass::bfloat16_t, cutlass::arch::Sm80);
    DispatchXformerWSS(cutlass::bfloat16_t, cutlass::arch::Sm75);
    /* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
#endif  // ENABLE_BF16
  }

#undef DispatchXformerWSS
  return 0;
}

// fixme: slow compiling, fixme later
AsStatus xformer_prefill_attention(const xformer_t& param, void* qptr,
                                   void* kptr, void* vptr, void* output,
                                   void* workspace, float alpha,
                                   const cudaStream_t& stream) {
#define DispatchXformer(CType, CArch)                                          \
  if (param.dtype == xformer_utils::CutlassType2DataType<CType>::dt &&         \
      param.sm_version >= xformer_utils::CutlassArch2Int<CArch>::version) {    \
    return xformer_prefill_attention_dispatch<CType, CArch, kBlockQuery,       \
                                              kBlockKey, kMaxK>(               \
        param, reinterpret_cast<CType*>(qptr), reinterpret_cast<CType*>(kptr), \
        reinterpret_cast<CType*>(vptr), reinterpret_cast<CType*>(output),      \
        workspace, alpha, stream);                                             \
  }

  //  CUDA
  if (param.phead <= 64) {
    constexpr int kBlockQuery = 64;
    constexpr int kBlockKey = 64;
    constexpr int kMaxK = 64;
    DispatchXformer(float, cutlass::arch::Sm80);
    DispatchXformer(float, cutlass::arch::Sm75);
    DispatchXformer(float, cutlass::arch::Sm70);
#ifdef ENABLE_FP16
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm80);
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm75);
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm70);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    DispatchXformer(cutlass::bfloat16_t, cutlass::arch::Sm80);
    DispatchXformer(cutlass::bfloat16_t, cutlass::arch::Sm75);
    /* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
#endif  // ENABLE_BF16
  } else if (param.phead <= 128) {
    constexpr int kBlockQuery = 32;
    constexpr int kBlockKey = 128;
    constexpr int kMaxK = 128;
    DispatchXformer(float, cutlass::arch::Sm80);
    DispatchXformer(float, cutlass::arch::Sm75);
    DispatchXformer(float, cutlass::arch::Sm70);
#ifdef ENABLE_FP16
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm80);
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm75);
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm70);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    DispatchXformer(cutlass::bfloat16_t, cutlass::arch::Sm80);
    DispatchXformer(cutlass::bfloat16_t, cutlass::arch::Sm75);
    /* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
#endif  // ENABLE_BF16
  } else {
    constexpr int kBlockQuery = 32;
    constexpr int kBlockKey = 128;
    constexpr int kMaxK = 65536;
    DispatchXformer(float, cutlass::arch::Sm80);
    DispatchXformer(float, cutlass::arch::Sm75);
    DispatchXformer(float, cutlass::arch::Sm70);
#ifdef ENABLE_FP16
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm80);
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm75);
    DispatchXformer(cutlass::half_t, cutlass::arch::Sm70);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
    DispatchXformer(cutlass::bfloat16_t, cutlass::arch::Sm80);
    DispatchXformer(cutlass::bfloat16_t, cutlass::arch::Sm75);
    /* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
#endif  // ENABLE_BF16
  }

#undef DispatchXformer
  return AsStatus::ALLSPARK_RUNTIME_ERROR;
}

}  // namespace cuda
}  // namespace allspark
#endif  // ENABLE_CUDA
