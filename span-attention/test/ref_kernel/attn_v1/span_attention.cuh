/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attention.cuh
 */

#pragma once

#include <stdexcept>

namespace allspark {
namespace cuda {

namespace span_attention {

struct RetConfig {
  static const int Success = 0;
  static const int ProgramError = -1;
  static const int CudaRuntimeError = -2;
};

}  // namespace span_attention

template <typename T>
class SpanAttention {
 public:
  SpanAttention(int headSize, int nHead, int nGroups, int seqLength,
                int chunkSize, int nChunk, int batch, int deviceId) {
    throw std::runtime_error("NOT IMPLEMENTED");
  }
  SpanAttention(const SpanAttention&) = delete;
  SpanAttention(SpanAttention&&) = delete;

  int Run(const T* Q, const T* const* KCachePtrs, const T* const* VCachePtrs,
          T* O, float QKScale, void* ws, size_t wsSize,
          cudaStream_t stream) const {
    throw std::runtime_error("NOT IMPLEMENTED");
  }
  size_t GetWorkspaceSize() const {
    throw std::runtime_error("NOT IMPLEMENTED");
  }
};

/**
 * @brief Create a SpanAttention object.
 *
 * @tparam T
 * @param obj The address of the pointer to the object to be created.
 * @param batchSize Batch size.
 * @param nHeads Number of query heads.
 * @param nGroups Number of key/value heads.
 * @param headSize Size of each MHA head, must be power of 2; support 64 and 129
 * for now.
 * @param spanLen Length (number of tokens) of each cache span; support 16 and
 * 32 for now.
 * @param nSpansPerBatch Number of cache spans for each request in the batch.
 * @param seqLen Length (number of tokens) of the cached sequence, including the
 * newly generated token(s).
 */
template <typename T>
void SpanAttentionCreate(SpanAttention<T>** obj, int batchSize, int nHeads,
                         int nGroups, int headSize, int spanLen,
                         int nSpansPerBatch, int seqLen, int deviceId) {
  *obj = new SpanAttention<T>(headSize, nHeads, nGroups, seqLen, spanLen,
                              nSpansPerBatch, batchSize, deviceId);
  return;
}

/**
 * @brief Destroy the SpanAttention object.
 *
 * @tparam T
 * @param obj The pointer to the SpanAttention object created via
 * #SpanAttentionCreate.
 */
template <typename T>
void SpanAttentionDestroy(SpanAttention<T>* obj) {
  delete obj;
  return;
}

/**
 * @brief Store the workspace size in bytes in wsInBytes.
 *
 * @tparam T
 * @param obj SpanAttention object.
 * @param wsInBytes The pointer to the variable to store the workspace size in
 * bytes.
 */
template <typename T>
void SpanAttentionGetWorkspaceSize(const SpanAttention<T>* obj,
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
 * @param obj SpanAttention object.
 * @param output The device pointer to the output tensor.
 * @param query The device pointer to the input query (Q) tensor.
 * @param kSpanArray The device pointer to the key (K) cache span array.
 * @param vSpanArray The device pointer to the value (V) cache span array.
 * @param workspace The device pointer to the workspace.
 * @param wsSizeInBytes Workspace size in bytes.
 * @param stream CUDA stream.
 */
template <typename T>
void SpanAttentionLauncher(const SpanAttention<T>* obj, T* output,
                           const T* query, const void* const* kSpanArray,
                           const void* const* vSpanArray, float QKScale,
                           void* workspace, size_t wsSizeInBytes,
                           cudaStream_t stream) {
  const T* const* kSpanArrayPtr = reinterpret_cast<const T* const*>(kSpanArray);
  const T* const* vSpanArrayPtr = reinterpret_cast<const T* const*>(vSpanArray);

  int ret = obj->Run(query, kSpanArrayPtr, vSpanArrayPtr, output, QKScale,
                     workspace, wsSizeInBytes, stream);
  if (ret != span_attention::RetConfig::Success) {
    throw std::runtime_error("SpanAttentionLauncher failed with error code " +
                             std::to_string(ret));
  }
  return;
}

}  // namespace cuda
}  // namespace allspark
