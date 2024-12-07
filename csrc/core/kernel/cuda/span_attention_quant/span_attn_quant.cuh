/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn_quant.cuh
 */

#pragma once

#include <stdexcept>

#include "../cache_quant/qcache.cuh"
#include "impl.cuh"

namespace allspark {
namespace cuda {

template <QuantMode QMODE, typename T>
class SpanAttentionQuant {
  using QT = typename qcache::QCacheConfig<QMODE, T>::QuantT;
  using CPT = typename qcache::QCacheConfig<QMODE, T>::ComputeT;

  span_attention_quant::QuantizedSpanAttn<QMODE, T, QT, CPT, CPT> obj;

 public:
  SpanAttentionQuant(int headSize, int nHead, int seqLength, int chunkSize,
                     int nChunk, int batch, int deviceId)
      : obj(headSize, nHead, seqLength, chunkSize, nChunk, batch, deviceId) {}

  size_t GetWorkspaceSize() const { return obj.GetWorkspaceSize(); }

  /**
   *  Q: device memory
   *  KCachePtrs, VCachePtrs: host memory
   *  O: device memory
   * ws: device memory
   *
   * QKScale: scaling factor for QK (the common value is 1 / sqrt(headSize))
   *
   * return value:
   *     0: success
   *    -1: program error
   *    -2: cuda runtime error
   */
  int Run(const T* Q, const QT* const* KCachePtrs, const QT* const* VCachePtrs,
          T* O, CPT QKScale, void* ws, size_t wsSize,
          cudaStream_t stream) const {
    return obj.Run(Q, KCachePtrs, VCachePtrs, O, ws, wsSize, QKScale, stream);
  }
};

/**
 * @brief Create a SpanAttentionQuant object.
 *
 * @tparam T
 * @param obj The address of the pointer to the object to be created.
 * @param batchSize Batch size.
 * @param nHeads Number of MHA heads.
 * @param headSize Size of each MHA head, must be power of 2; support 64 and
 * 129 for now.
 * @param spanLen Length (number of tokens) of each cache span; support 16
 * and 32 for now.
 * @param nSpansPerBatch Number of cache spans for each request in the
 * batch.
 * @param seqLen Length (number of tokens) of the cached sequence, including
 * the newly generated token(s).
 */
template <QuantMode QMODE, typename T>
void SpanAttentionQuantCreate(SpanAttentionQuant<QMODE, T>** obj, int batchSize,
                              int nHeads, int headSize, int spanLen,
                              int nSpansPerBatch, int seqLen, int deviceId) {
  *obj = new SpanAttentionQuant<QMODE, T>(headSize, nHeads, seqLen, spanLen,
                                          nSpansPerBatch, batchSize, deviceId);
  return;
}

/**
 * @brief Destroy the SpanAttentionQuant object.
 *
 * @tparam T
 * @param obj The pointer to the SpanAttentionQuant object created via
 * #SpanAttentionQuantCreate.
 */
template <QuantMode QMODE, typename T>
void SpanAttentionQuantDestroy(SpanAttentionQuant<QMODE, T>* obj) {
  delete obj;
  return;
}

/**
 * @brief Store the workspace size in bytes in wsInBytes.
 *
 * @tparam T
 * @param obj SpanAttentionQuant object.
 * @param wsInBytes The pointer to the variable to store the workspace size in
 * bytes.
 */
template <QuantMode QMODE, typename T>
void SpanAttentionQuantGetWorkspaceSize(const SpanAttentionQuant<QMODE, T>* obj,
                                        size_t* wsInBytes) {
  size_t ret = obj->GetWorkspaceSize();
  *wsInBytes = ret;
  return;
}

/**
 * @brief Compute span attention.
 *
 * Shape info:
 *
 * output: [batchSize, (1), nHeads, headSize]
 * query: [batchSize, (1), nHeads, headSize]
 * kSpanArray/vSpanArray: [batchSize, nSpans]
 *                                       |---> [nHeads, spanLen, headSize]
 *
 * @tparam T
 * @param obj SpanAttentionQuant object.
 * @param output The device pointer to the output tensor.
 * @param query The device pointer to the input query (Q) tensor.
 * @param kSpanArray The device pointer to the key (K) cache span array.
 * @param vSpanArray The device pointer to the value (V) cache span array.
 * @param workspace The device pointer to the workspace.
 * @param wsSizeInBytes Workspace size in bytes.
 * @param stream CUDA stream.
 */
template <QuantMode QMODE, typename T>
void SpanAttentionQuantLauncher(const SpanAttentionQuant<QMODE, T>* obj,
                                T* output, const T* query,
                                const void* const* kSpanArray,
                                const void* const* vSpanArray, float QKScale,
                                void* workspace, size_t wsSizeInBytes,
                                cudaStream_t stream) {
  using QT = typename qcache::QCacheConfig<QMODE, T>::QuantT;
  using CPT = typename qcache::QCacheConfig<QMODE, T>::ComputeT;

  const QT* const* kSpanArrayPtr =
      reinterpret_cast<const QT* const*>(kSpanArray);
  const QT* const* vSpanArrayPtr =
      reinterpret_cast<const QT* const*>(vSpanArray);

  int ret =
      obj->Run(query, kSpanArrayPtr, vSpanArrayPtr, output,
               static_cast<CPT>(QKScale), workspace, wsSizeInBytes, stream);
  if (ret != span_attention_quant::RetConfig::Success) {
    throw std::runtime_error(
        "SpanAttentionQuantLauncher failed with error code " +
        std::to_string(ret));
  }
  return;
}

}  // namespace cuda
}  // namespace allspark
