/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn.h
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace span {

/**
 * @brief SpanAttention handle type.
 */
typedef struct SpanAttnHandle* SpanAttnHandle_t;

/**
 * @brief Input and output data type.
 *
 * Supported types: FP32, FP16, BF16.
 */
enum class DataType : int {
  /// @brief 32-bit single precision floating point.
  FP32,
  /// @brief 16-bit half precision floating point.
  FP16,
  /// @brief 16-bit bfloat16 precision floating point.
  BF16,
};

/**
 * @brief KV cache quantization mode.
 *
 * Supported modes: NONE, I8, U4.
 */
enum class QuantMode : int {
  /// @brief No quantization.
  NONE,
  /// @brief 8-bit signed integer quantization.
  I8,
  /// @brief 4-bit unsigned integer quantization.
  U4,
};

/**
 * @brief SpanAttention status code.
 */
enum class SaStatus : int {
  /// @brief Success.
  SUCCESS = 0,
  /// @brief CUDA runtime API error.
  CUDA_ERROR = 1,
  /// @brief Runtime error.
  RUNTIME_ERROR = 2,
  /// @brief Invalid parameter.
  PARAM_ERROR = 3,
  /// @brief Parameter exceeding numerical limits.
  EXCEED_LIMIT_ERROR = 4,
  /// @brief Internal error.
  INTERNAL_ERROR = 5,
  /// @brief Unrecognized error.
  UNKNOWN_ERROR = 127,
};

/**
 * @brief Get the string representation of a SpanAttention status code enum
 * name.
 *
 * @param code SpanAttention status code.
 * @return @c char* pointer to a NULL-terminated string.
 */
const char* GetErrorName(SaStatus code);

/**
 * @brief Get the description of a SpanAttention status code.
 *
 * @param code SpanAttention status code.
 * @return @c char* pointer to a NULL-terminated string.
 */
const char* GetErrorString(SaStatus code);

/**
 * @brief Create a SpanAttention handle.
 *
 * @param handle The pointer to the handle to be created.
 * @param dataType The data type of input and output.
 * @param kvQuantMode KV cache quantization mode.
 * @param batchSize Batch size.
 * @param nHeads Number of query heads, must be a multiple of nGroups.
 * @param nGroups Number of key/value heads; requiring nHeads / nGroups <=
 * 32 for now.
 * @param headSize Size of each head, must be a power of 2; supporting 128
 * for now.
 * @param spanLen Length (number of tokens) of each cache span; supporting
 * 16, 32, 64, and 128 for now.
 * @param nSpansPerRequest Max number of cache spans for a request in the
 * batch.
 * @param seqLen Host pointer to the array of the lengths (number of tokens)
 * of the cached sequence, @b *including* the newly generated token(s).
 * @param deviceProp CUDA device properties, obtained by calling CUDA runtime
 * API @c cudaGetDeviceProperties.
 *
 * @return @c SaStatus status code.
 */
SaStatus CreateHandle(SpanAttnHandle_t* handle, DataType dataType,
                      QuantMode kvQuantMode, int batchSize, int nHeads,
                      int nGroups, int headSize, int spanLen,
                      int nSpansPerRequest, const int* seqLen,
                      const cudaDeviceProp& deviceProp);

/**
 * @brief Destroy a SpanAttention handle.
 *
 * @param handle The handle to be destroyed.
 *
 * @return @c SaStatus status code.
 */
SaStatus DestroyHandle(SpanAttnHandle_t handle);

/**
 * @brief Store the host workspace size in bytes in wsInBytes.
 *
 * @param wsInBytes The pointer to the variable to store the host workspace size
 * in bytes.
 * @param handle SpanAttention handle.
 *
 * @return @c SaStatus status code.
 */
SaStatus GetHostWorkspaceSize(size_t* wsInBytes, SpanAttnHandle_t handle);

/**
 * @brief Store the device workspace size in bytes in wsInBytes.
 *
 * @param wsInBytes The pointer to the variable to store the device workspace
 * size in bytes.
 * @param handle SpanAttention handle.
 *
 * @return @c SaStatus status code.
 */
SaStatus GetDeviceWorkspaceSize(size_t* wsInBytes, SpanAttnHandle_t handle);

/**
 * @brief Compute span attention.
 *
 * Shape info:
 *
 * output: [batchSize, (1,) nHeads, headSize]
 *
 * query: [batchSize, (1,) nHeads, headSize]
 *
 * kSpanArray / vSpanArray: [batchSize, nSpansPerRequest]
 *                                       |---> [nGroups, spanLen, headSize]
 *
 * @param output The device pointer to the output tensor.
 * @param query The device pointer to the input query (Q) tensor.
 * @param kSpanArray The device pointer to the key (K) cache span array.
 * @param vSpanArray The device pointer to the value (V) cache span array.
 * @param deviceWorkspace The device pointer to the device workspace.
 * @param deviceWsSizeInBytes Device workspace size in bytes.
 * @param hostWorkspace The host pointer to the host workspace.
 * @param hostWsSizeInBytes Host workspace size in bytes.
 * @param QKScale Scaling factor for QK computation.
 * @param handle SpanAttention handle.
 * @param stream CUDA stream.
 *
 * @return @c SaStatus status code.
 */
SaStatus Run(void* output, const void* query, const void* const* kSpanArray,
             const void* const* vSpanArray, void* deviceWorkspace,
             size_t deviceWsSizeInBytes, void* hostWorkspace,
             size_t hostWsSizeInBytes, float QKScale, SpanAttnHandle_t handle,
             cudaStream_t stream);

}  // namespace span
