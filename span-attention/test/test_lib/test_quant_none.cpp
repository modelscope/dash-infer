/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    test_quant_none.cpp
 */

#include <span_attn.h>
#include <span_attn_v1.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <sstream>

#include "common.hpp"
#include "framework_utils.hpp"
#include "test_common.h"

using namespace allspark;

// #define KERNEL_SPAN_ATTN_TEST_VERBOSE
#define KERNEL_SPAN_ATTN_TEST_WARMUP
#define KERNEL_SPAN_ATTN_TEST_REF
// #define KERNEL_SPAN_ATTN_TEST_CHECK_PER_CHANNEL

namespace {

enum class BaselineMode {
  LOOP,
  BATCH,
};

template <typename T>
span::DataType GetDataType() {
  return span::DataType::FP32;
}

#ifdef ENABLE_FP16
template <>
span::DataType GetDataType<half>() {
  return span::DataType::FP16;
}
#endif

#ifdef ENABLE_BF16
template <>
span::DataType GetDataType<hie::bfloat16>() {
  return span::DataType::BF16;
}
#endif

uint32_t udiv_up(uint32_t seq_len, uint32_t span_len) {
  return (seq_len + span_len - 1) / span_len;
}

// if a % b == 0, reserve 1 additional span
uint32_t udiv_up_reserve(uint32_t seq_len, uint32_t span_len) {
  return (seq_len + span_len) / span_len;
}

[[maybe_unused]] void reset_sstream(std::stringstream& ss) {
  std::stringstream tmp;
  ss.swap(tmp);
  return;
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
  for (auto& e : v) {
    o << e << " ";
  }
  return o;
}

span::QuantMode toQuantMode(AsCacheMode cache_mode) {
  switch (cache_mode) {
    case AsCacheMode::AsCacheDefault:
      return span::QuantMode::NONE;
    case AsCacheMode::AsCacheQuantI8:
      return span::QuantMode::I8;
    case AsCacheMode::AsCacheQuantU4:
      return span::QuantMode::U4;
    default:
      std::cerr << "Bad cache mode" << std::endl;
      auto panic = []() { ASSERT_TRUE(0); };
      panic();
      return span::QuantMode::NONE;
  }
}

struct SpanAttnParam {
  SpanAttnParam(const std::vector<int>& seq_lengths, int num_heads,
                int num_groups, int size_per_head, int span_size,
                int max_length, int device_id, cudaDeviceProp device_prop)
      : seq_lengths(seq_lengths),
        batch_size(seq_lengths.size()),
        num_heads(num_heads),
        num_groups(num_groups),
        size_per_head(size_per_head),
        hidden_size(num_heads * size_per_head),
        span_size(span_size),
        max_length(max_length),
        max_num_spans(udiv_up(max_length, span_size)),
        device_id(device_id),
        device_prop(std::move(device_prop)) {}

  const std::vector<int>& seq_lengths;
  const int batch_size;
  const int num_heads;
  const int num_groups;
  const int size_per_head;
  const int hidden_size;
  const int span_size;
  const int max_length;
  const int max_num_spans;
  const int device_id;
  const cudaDeviceProp device_prop;
};

std::ostream& operator<<(std::ostream& o, const SpanAttnParam& param) {
  o << "num_heads:\t" << param.num_heads << std::endl
    << "num_groups:\t" << param.num_groups << std::endl
    << "size_per_head:\t" << param.size_per_head << std::endl
    << "hidden_size:\t" << param.hidden_size << std::endl
    << "span_size:\t" << param.span_size << std::endl
    << "max_num_spans:\t" << param.max_num_spans << std::endl
    << "batch_size:\t" << param.batch_size << std::endl
    << "seq_lengths:\t" << param.seq_lengths;
  return o;
}

// ---------------------------------
// Baseline launcher
// ---------------------------------
template <BaselineMode MODE, typename T>
struct SpanAttnBaseline {
  static void Create(std::vector<allspark::cuda::SpanAttention<T>*>& out,
                     const SpanAttnParam& param) {
    for (auto seq_len : param.seq_lengths) {
      constexpr int single_batch = 1;
      allspark::cuda::SpanAttention<T>* kernelObj{nullptr};
      allspark::cuda::SpanAttentionCreate(
          &kernelObj, single_batch, param.num_heads, param.num_groups,
          param.size_per_head, param.span_size, param.max_num_spans, seq_len,
          param.device_id);
      ASSERT_NE(kernelObj, nullptr);
      out.push_back(kernelObj);
    }
  }

  static void Destroy(
      const std::vector<allspark::cuda::SpanAttention<T>*>& kernelObjs) {
    for (auto kernelObj : kernelObjs) {
      allspark::cuda::SpanAttentionDestroy(kernelObj);
    }
  }

  static void GetWorkspaceBytes(
      const std::vector<allspark::cuda::SpanAttention<T>*>& kernelObjs,
      size_t& ws_bytes) {
    for (auto kernelObj : kernelObjs) {
      size_t ws_bytes_task{0};
      allspark::cuda::SpanAttentionGetWorkspaceSize(kernelObj, &ws_bytes_task);
      ws_bytes = std::max(ws_bytes, ws_bytes_task);
    }
  }

  static void Run(
      const std::vector<allspark::cuda::SpanAttention<T>*>& kernelObjs,
      const SpanAttnParam& param, void* device_out, void* device_q,
      void* device_k_spans, void* device_v_spans, void* workspace,
      size_t workspace_bytes, float alpha, cudaStream_t cuda_stream) {
    for (int batch_idx = 0; batch_idx < param.batch_size; ++batch_idx) {
      constexpr int query_len = 1;
      [[maybe_unused]] const int seq_len = param.seq_lengths[batch_idx];
      const int batch_stride = query_len * param.hidden_size;
      cuda::SpanAttentionLauncher(
          kernelObjs[batch_idx],
          static_cast<T*>(device_out) + batch_idx * batch_stride,
          static_cast<const T*>(device_q) + batch_idx * batch_stride,
          static_cast<const void* const*>(device_k_spans) +
              batch_idx * param.max_num_spans,
          static_cast<const void* const*>(device_v_spans) +
              batch_idx * param.max_num_spans,
          alpha, workspace, workspace_bytes, cuda_stream);
    }
  }
};

template <typename T>
struct SpanAttnBaseline<BaselineMode::BATCH, T> {
  static void GetWorkspaceBytes(const SpanAttnParam& param, size_t& ws_bytes) {
    // check all sequences share the same length
    ASSERT_EQ(
        std::adjacent_find(param.seq_lengths.begin(), param.seq_lengths.end(),
                           std::not_equal_to<>()),
        param.seq_lengths.end());

    const int seq_len = param.seq_lengths[0];

    allspark::cuda::SpanAttention<T>* kernelObj{nullptr};
    allspark::cuda::SpanAttentionCreate(
        &kernelObj, param.batch_size, param.num_heads, param.num_groups,
        param.size_per_head, param.span_size, param.max_num_spans, seq_len,
        param.device_id);
    ASSERT_NE(kernelObj, nullptr);
    allspark::cuda::SpanAttentionGetWorkspaceSize(kernelObj, &ws_bytes);
    allspark::cuda::SpanAttentionDestroy(kernelObj);
  }

  static void Run(const SpanAttnParam& param, void* device_out, void* device_q,
                  void* device_k_spans, void* device_v_spans, void* workspace,
                  size_t workspace_bytes, float alpha,
                  cudaStream_t cuda_stream) {
    // check all sequences share the same length
    ASSERT_EQ(
        std::adjacent_find(param.seq_lengths.begin(), param.seq_lengths.end(),
                           std::not_equal_to<>()),
        param.seq_lengths.end());

    constexpr int query_len = 1;
    const int seq_len = param.seq_lengths[0];

    allspark::cuda::SpanAttention<T>* kernelObj{nullptr};
    allspark::cuda::SpanAttentionCreate(
        &kernelObj, param.batch_size, param.num_heads, param.num_groups,
        param.size_per_head, param.span_size, param.max_num_spans, seq_len,
        param.device_id);
    ASSERT_NE(kernelObj, nullptr);
    cuda::SpanAttentionLauncher(kernelObj, static_cast<T*>(device_out),
                                static_cast<const T*>(device_q),
                                static_cast<const void* const*>(device_k_spans),
                                static_cast<const void* const*>(device_v_spans),
                                alpha, workspace, workspace_bytes, cuda_stream);
    cuda::SpanAttentionDestroy(kernelObj);
  }
};

template <typename T>
struct SpanAttnV2 {
  static void Create(span::SpanAttnHandle_t* kernelObjPtr,
                     const SpanAttnParam& param) {
#if 0
    // check all sequences share the same length
    ASSERT_EQ(
        std::adjacent_find(param.seq_lengths.begin(), param.seq_lengths.end(),
                           std::not_equal_to<>()),
        param.seq_lengths.end());

    const int seq_len = param.seq_lengths[0];
#endif

    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::CreateHandle(kernelObjPtr, GetDataType<T>(),
                                 toQuantMode(AsCacheMode::AsCacheDefault),
                                 param.batch_size, param.num_heads,
                                 param.num_groups, param.size_per_head,
                                 param.span_size, param.max_num_spans,
                                 param.seq_lengths.data(), param.device_prop));
    ASSERT_NE(*kernelObjPtr, nullptr);
  }

  static void Destroy(span::SpanAttnHandle_t kernelObj) {
    ASSERT_EQ(span::SaStatus::SUCCESS, span::DestroyHandle(kernelObj));
  }

  static void GetHostWorkspaceBytes(const span::SpanAttnHandle_t kernelObj,
                                    size_t& ws_bytes) {
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetHostWorkspaceSize(&ws_bytes, kernelObj));
  }

  static void GetWorkspaceBytes(const span::SpanAttnHandle_t kernelObj,
                                size_t& ws_bytes) {
#if 0
    // check all sequences share the same length
    ASSERT_EQ(
        std::adjacent_find(param.seq_lengths.begin(), param.seq_lengths.end(),
                           std::not_equal_to<>()),
        param.seq_lengths.end());

    const int seq_len = param.seq_lengths[0];
#endif

    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetDeviceWorkspaceSize(&ws_bytes, kernelObj));
  }

  static void Run(const span::SpanAttnHandle_t kernelObj,
                  const SpanAttnParam& param, void* device_out, void* device_q,
                  void* device_k_spans, void* device_v_spans, void* workspace,
                  size_t workspace_bytes, void* host_workspace,
                  size_t host_workspace_bytes, float alpha,
                  cudaStream_t cuda_stream) {
#if 0
    // check all sequences share the same length
    ASSERT_EQ(
        std::adjacent_find(param.seq_lengths.begin(), param.seq_lengths.end(),
                           std::not_equal_to<>()),
        param.seq_lengths.end());

    const int seq_len = param.seq_lengths[0];
#endif

    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::Run(device_out, device_q,
                        static_cast<const void* const*>(device_k_spans),
                        static_cast<const void* const*>(device_v_spans),
                        workspace, workspace_bytes, host_workspace,
                        host_workspace_bytes, alpha, kernelObj, cuda_stream));
  }
};

// ---------------------------------
// Test
// ---------------------------------
class TestQuantNone : public ::testing::Test {
 public:
  template <typename T>
  void test_span_attn(int batch_size, int seq_length, int num_heads,
                      int num_groups, int size_per_head, int span_size,
                      int max_length, float feps = 1e-3,
                      bool check_init = false) {
    test_span_attn<T>(std::vector<int>(batch_size, seq_length), num_heads,
                      num_groups, size_per_head, span_size, max_length, feps,
                      check_init);
    return;
  }

  template <typename T>
  void test_span_attn(const std::vector<int>& seq_lengths, int num_heads,
                      int num_groups, int size_per_head, int span_size,
                      int max_length, float feps = 1e-3,
                      bool check_init = false) {
    ASSERT_GT(size_per_head, 0);
    ASSERT_GT(num_heads, 0);
    ASSERT_GT(num_groups, 0);
    ASSERT_GT(span_size, 0);
    ASSERT_GT(max_length, 0);
    ASSERT_EQ(num_heads % num_groups, 0);
    ASSERT_GT(seq_lengths.size(), 0);
    ASSERT_LE(seq_lengths.size(), std::numeric_limits<int>::max());

    const auto& cuda_stream =
        dynamic_cast<const CUDAContext*>(device.get())->GetStream();
    const auto& device_id =
        dynamic_cast<const CUDAContext*>(device.get())->GetDeviceId();

    SpanAttnParam param(seq_lengths, num_heads, num_groups, size_per_head,
                        span_size, max_length, device_id, device_prop);

    const int batch_size = param.batch_size;
    const int max_num_spans = param.max_num_spans;
    const int hidden_size = param.hidden_size;

    int num_spans = 0;
    std::vector<int> request_num_spans(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      auto len = seq_lengths[i];
      int request_num_span = udiv_up_reserve(len, span_size);
      request_num_spans[i] = request_num_span;
      num_spans += request_num_span;
    }
    ASSERT_TRUE(num_spans <= max_num_spans * batch_size);

    const int max_request_num_span =
        *std::max_element(request_num_spans.begin(), request_num_spans.end());

    // TODO: quant
    const size_t span_bytes =
        num_groups * span_size * size_per_head * sizeof(T);
    const float alpha = 1.0f / std::sqrt(size_per_head * 1.0f);

    // tensor map
    allspark::TensorMap tensors;

    std::string span_pool_name = "__span_pool";
    // 2: K and V
    ASSERT_LE(2 * num_spans * span_bytes, std::numeric_limits<dim_t>::max());
    std::vector<allspark::dim_t> shape_span_pool = {
        static_cast<dim_t>(2 * num_spans * span_bytes)};
    common::AddTensor(tensors, span_pool_name, asINT8);

    std::string out_name = "out_tensor";
    std::vector<allspark::dim_t> shape_out = {batch_size, hidden_size};
    common::AddTensor(tensors, out_name, common::toDataType<T>::dt);

#ifdef KERNEL_SPAN_ATTN_TEST_REF
    std::string ref_name = "ref_tensor";
    std::vector<allspark::dim_t> shape_ref = {batch_size, hidden_size};
    common::AddTensor(tensors, ref_name, common::toDataType<T>::dt);
#endif

    std::string q_name = "decoder_q_tensor";
    std::vector<allspark::dim_t> shape_q = {batch_size, hidden_size};
    common::AddTensor(tensors, q_name, common::toDataType<T>::dt);

    std::string k_span_name = "k_span_array_tensor_device";
    std::vector<allspark::dim_t> shape_k_span = {batch_size, max_num_spans};
    common::AddTensor(tensors, k_span_name, asPTR);

    std::string v_span_name = "v_span_array_tensor_device";
    std::vector<allspark::dim_t> shape_v_span = {batch_size, max_num_spans};
    common::AddTensor(tensors, v_span_name, asPTR);

    // workspace
    span::SpanAttnHandle_t outObj;
    auto timeBeforeCreate = std::chrono::steady_clock::now();
    SpanAttnV2<T>::Create(&outObj, param);
    auto timeAfterCreate = std::chrono::steady_clock::now();
    std::cout << "SpanAttnV2 create time:\t"
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     timeAfterCreate - timeBeforeCreate)
                     .count()
              << " us" << std::endl;

    size_t host_ws_bytes{0};
    void* host_ws{nullptr};
    SpanAttnV2<T>::GetHostWorkspaceBytes(outObj, host_ws_bytes);
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&host_ws, host_ws_bytes));

    size_t ws_bytes{0};
    SpanAttnV2<T>::GetWorkspaceBytes(outObj, ws_bytes);
#ifdef KERNEL_SPAN_ATTN_TEST_REF
    size_t ref_ws_bytes{0};
    std::vector<cuda::SpanAttention<T>*> refObjs;
    SpanAttnBaseline<BaselineMode::LOOP, T>::Create(refObjs, param);
    SpanAttnBaseline<BaselineMode::LOOP, T>::GetWorkspaceBytes(refObjs,
                                                               ref_ws_bytes);
    ws_bytes = std::max(ws_bytes, ref_ws_bytes);
#endif
    std::string ws_name = "workspace";
    std::vector<allspark::dim_t> shape_ws = {dim_t(ws_bytes)};
    common::AddTensor(tensors, ws_name, asINT8);

    // reshape
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(span_pool_name)->SetShape(Shape(shape_span_pool)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(q_name)->SetShape(Shape(shape_q)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(k_span_name)->SetShape(Shape(shape_k_span)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(v_span_name)->SetShape(Shape(shape_v_span)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(out_name)->SetShape(Shape(shape_out)));
#ifdef KERNEL_SPAN_ATTN_TEST_REF
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(ref_name)->SetShape(Shape(shape_ref)));
#endif
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(ws_name)->SetShape(Shape(shape_ws)));

    device->Synchronize();

    std::cout << param << std::endl
              << "span_byte:\t" << span_bytes << " (0x" << std::hex
              << span_bytes << std::dec << ")" << std::endl
              << "span_pool_base:\t" << tensors.at(span_pool_name)->GetDataPtr()
              << std::endl
              << "span_pool_byte:\t"
              << tensors.at(span_pool_name)->GetSizeInByte() << " (0x"
              << std::hex << tensors.at(span_pool_name)->GetSizeInByte()
              << std::dec << ")" << std::endl
              << "workspace_byte:\t" << tensors.at(ws_name)->GetSizeInByte()
              << " (0x" << std::hex << tensors.at(ws_name)->GetSizeInByte()
              << std::dec << ")" << std::endl;

    // set data
    // TODO: quant
    std::vector<T> span_values = common::rand_normal_float<T>(
        2ULL * num_spans * num_groups * span_size * size_per_head, 50.f, 10.f,
        0.f, 64);
    std::vector<T> q_data = common::rand_normal_float<T>(
        batch_size * hidden_size, 50.f, 10.f, 0.f, batch_size);

    std::vector<void*> k_span_ptrs(batch_size * max_num_spans, 0);
    std::vector<void*> v_span_ptrs(batch_size * max_num_spans, 0);
    /// NOTE: strided span allocation, to simulate non-contiguous spans
    int alloc_span_idx = 0;
    char* span_base =
        static_cast<char*>(tensors.at(span_pool_name)->GetDataPtr());
    for (int span_idx = 0; span_idx < max_request_num_span; ++span_idx) {
      for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (span_idx >= request_num_spans[batch_idx]) {
          continue;
        }
        k_span_ptrs[batch_idx * max_num_spans + span_idx] =
            span_base + (alloc_span_idx++) * span_bytes;
      }
    }
    alloc_span_idx = 0;
    span_base += num_spans * span_bytes;
    for (int span_idx = 0; span_idx < max_request_num_span; ++span_idx) {
      for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (span_idx >= request_num_spans[batch_idx]) {
          continue;
        }
        v_span_ptrs[batch_idx * max_num_spans + span_idx] =
            span_base + (alloc_span_idx++) * span_bytes;
      }
    }

#ifdef KERNEL_SPAN_ATTN_TEST_VERBOSE
    for (int i = 0; i < batch_size; ++i) {
      std::cout << "request " << i << ":\t" << request_num_spans[i] << " spans"
                << std::endl;
      std::cout << "\tk:\t";
      for (int j = 0; j < udiv_up_reserve(seq_lengths[i], span_size); ++j) {
        std::cout << k_span_ptrs[i * max_num_spans + j] << "\t";
      }
      std::cout << std::endl;

      std::cout << "\tv:\t";
      for (int j = 0; j < udiv_up_reserve(seq_lengths[i], span_size); ++j) {
        std::cout << v_span_ptrs[i * max_num_spans + j] << "\t";
      }
      std::cout << std::endl;
    }
#endif

    ASSERT_TRUE(common::AsyncH2D(span_values.data(),
                                 tensors.at(span_pool_name).get(),
                                 span_values.size() * sizeof(T), cuda_stream));
    ASSERT_TRUE(common::AsyncH2D(q_data.data(), tensors.at(q_name).get(),
                                 q_data.size() * sizeof(T), cuda_stream));
    ASSERT_TRUE(
        common::AsyncH2D(k_span_ptrs.data(), tensors.at(k_span_name).get(),
                         k_span_ptrs.size() * sizeof(void*), cuda_stream));
    ASSERT_TRUE(
        common::AsyncH2D(v_span_ptrs.data(), tensors.at(v_span_name).get(),
                         v_span_ptrs.size() * sizeof(void*), cuda_stream));

    cudaEvent_t ref_start, ref_end;
    cudaEvent_t out_start, out_end;
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&ref_start));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&ref_end));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&out_start));
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&out_end));

    float ref_time{0}, out_time{0};
#ifdef KERNEL_SPAN_ATTN_TEST_REF
    // ref
#ifdef KERNEL_SPAN_ATTN_TEST_WARMUP
    // warmup
    SpanAttnBaseline<BaselineMode::LOOP, T>::Run(
        refObjs, param, tensors.at(ref_name)->GetDataPtr(),
        tensors.at(q_name)->GetDataPtr(), tensors.at(k_span_name)->GetDataPtr(),
        tensors.at(v_span_name)->GetDataPtr(),
        tensors.at(ws_name)->GetDataPtr(), tensors.at(ws_name)->GetSizeInByte(),
        alpha, cuda_stream);
#endif
    ASSERT_EQ(cudaSuccess, cudaEventRecord(ref_start, cuda_stream));
    SpanAttnBaseline<BaselineMode::LOOP, T>::Run(
        refObjs, param, tensors.at(ref_name)->GetDataPtr(),
        tensors.at(q_name)->GetDataPtr(), tensors.at(k_span_name)->GetDataPtr(),
        tensors.at(v_span_name)->GetDataPtr(),
        tensors.at(ws_name)->GetDataPtr(), tensors.at(ws_name)->GetSizeInByte(),
        alpha, cuda_stream);
    ASSERT_EQ(cudaSuccess, cudaEventRecord(ref_end, cuda_stream));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(ref_end));
    ASSERT_EQ(cudaSuccess, cudaEventElapsedTime(&ref_time, ref_start, ref_end));
    std::cout << "Ref time: " << ref_time << " ms" << std::endl;
#endif
    // run
#ifdef KERNEL_SPAN_ATTN_TEST_WARMUP
    // warmup
    SpanAttnV2<T>::Run(
        outObj, param, tensors.at(out_name)->GetDataPtr(),
        tensors.at(q_name)->GetDataPtr(), tensors.at(k_span_name)->GetDataPtr(),
        tensors.at(v_span_name)->GetDataPtr(),
        tensors.at(ws_name)->GetDataPtr(), tensors.at(ws_name)->GetSizeInByte(),
        host_ws, host_ws_bytes, alpha, cuda_stream);
#endif
    ASSERT_EQ(cudaSuccess, cudaEventRecord(out_start, cuda_stream));
    SpanAttnV2<T>::Run(
        outObj, param, tensors.at(out_name)->GetDataPtr(),
        tensors.at(q_name)->GetDataPtr(), tensors.at(k_span_name)->GetDataPtr(),
        tensors.at(v_span_name)->GetDataPtr(),
        tensors.at(ws_name)->GetDataPtr(), tensors.at(ws_name)->GetSizeInByte(),
        host_ws, host_ws_bytes, alpha, cuda_stream);
    ASSERT_EQ(cudaSuccess, cudaEventRecord(out_end, cuda_stream));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(out_end));
    ASSERT_EQ(cudaSuccess, cudaEventElapsedTime(&out_time, out_start, out_end));
    std::cout << "Out time: " << out_time << " ms" << std::endl;

    std::vector<T> host_out(batch_size * hidden_size);
    ASSERT_TRUE(common::AsyncD2H(tensors.at(out_name).get(), host_out.data(),
                                 host_out.size() * sizeof(T), cuda_stream));
#ifdef KERNEL_SPAN_ATTN_TEST_REF
    std::vector<T> host_ref(batch_size * hidden_size);
    ASSERT_TRUE(common::AsyncD2H(tensors.at(ref_name).get(), host_ref.data(),
                                 host_ref.size() * sizeof(T), cuda_stream));
#endif
    device->Synchronize();

#ifdef KERNEL_SPAN_ATTN_TEST_REF
    // check
#ifndef KERNEL_SPAN_ATTN_TEST_CHECK_PER_CHANNEL
    float diff = check_equal<T>(host_ref.data(), host_out.data(),
                                batch_size * hidden_size, false, 64);
    printf("[DIFF] max diff = %f\n", diff);
    EXPECT_LT(diff, feps);
#else
    for (int i = 0; i < batch_size; ++i) {
      std::cout << "Request " << i << " seq len " << seq_lengths[i]
                << std::endl;
      for (int j = 0; j < num_heads; ++j) {
        std::cout << "++Head " << j << std::endl;
        float diff = check_equal<T>(
            host_ref.data() + i * hidden_size + j * size_per_head,
            host_out.data() + i * hidden_size + j * size_per_head,
            size_per_head);
        printf("[DIFF] max diff = %f\n", diff);
        EXPECT_LT(diff, feps);
      }
    }
#endif
#endif

#ifdef KERNEL_SPAN_ATTN_TEST_REF
    SpanAttnBaseline<BaselineMode::LOOP, T>::Destroy(refObjs);
    SpanAttnV2<T>::Destroy(outObj);
#endif

    ASSERT_EQ(cudaSuccess, cudaFreeHost(host_ws));

    ASSERT_EQ(cudaSuccess, cudaEventDestroy(ref_start));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(ref_end));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(out_start));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(out_end));

    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(span_pool_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(out_name)->Free());
#ifdef KERNEL_SPAN_ATTN_TEST_REF
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(ref_name)->Free());
#endif
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(q_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(k_span_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(v_span_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(ws_name)->Free());
    return;
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&device_prop, 0));
    return;
  }
  void TearDown() override {}

 protected:
  // context
  std::shared_ptr<allspark::DeviceContext> device;
  cudaDeviceProp device_prop;
};

}  // namespace

// qwen2-72b: qheads=64, kvheads=8, maxlen=8k, 2cards

#ifdef ENABLE_FP16
TEST_F(TestQuantNone, fp16_mha_q2g2s16) {
  test_span_attn<half>(2, 15, 2, 2, 128, 16, 64);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b16l2k) {
  test_span_attn<half>(16, 2047, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b16l4k) {
  test_span_attn<half>(16, 4095, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b16l8k) {
  test_span_attn<half>(16, 8191, 32, 32, 128, 32, 8192);
}

//

TEST_F(TestQuantNone, fp16_mha_q32g32s32b32l2k) {
  test_span_attn<half>(32, 2047, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b32l4k) {
  test_span_attn<half>(32, 4095, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b32l8k) {
  test_span_attn<half>(32, 8191, 32, 32, 128, 32, 8192);
}

//

TEST_F(TestQuantNone, fp16_mha_q32g32s32b48l2k) {
  test_span_attn<half>(48, 2047, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b48l4k) {
  test_span_attn<half>(48, 4095, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b48l8k) {
  test_span_attn<half>(48, 8191, 32, 32, 128, 32, 8192);
}

//
TEST_F(TestQuantNone, fp16_mha_q32g32s32b64l2k) {
  test_span_attn<half>(64, 2047, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b64l4k) {
  test_span_attn<half>(64, 4095, 32, 32, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_mha_q32g32s32b64l8k) {
  test_span_attn<half>(64, 8191, 32, 32, 128, 32, 8192);
}

// gqa

// for debugging
#if 0
TEST_F(TestQuantNone, fp16_gqa_small_1) {
  test_span_attn<half>({15}, 1, 1, 128, 16, 16);
}

TEST_F(TestQuantNone, fp16_gqa_small_2) {
  test_span_attn<half>({33}, 2, 1, 128, 32, 64);
}

TEST_F(TestQuantNone, fp16_gqa_short_1) {
  test_span_attn<half>({15, 16}, 2, 1, 128, 32, 32);
}

TEST_F(TestQuantNone, fp16_gqa_short_2) {
  test_span_attn<half>({17, 31}, 2, 1, 128, 32, 32);
}

TEST_F(TestQuantNone, fp16_gqa_short_3) {
  test_span_attn<half>({31, 17}, 2, 1, 128, 32, 32);
}

TEST_F(TestQuantNone, fp16_gqa_short_4) {
  test_span_attn<half>({17}, 7, 1, 128, 16, 32);
}

TEST_F(TestQuantNone, fp16_gqa_short_5) {
  test_span_attn<half>({3}, 4, 2, 128, 16, 32);
}

TEST_F(TestQuantNone, fp16_gqa_short_6) {
  test_span_attn<half>({17, 15}, 16, 2, 128, 16, 32);
}

TEST_F(TestQuantNone, fp16_gqa_large_1) {
  test_span_attn<half>({81, 99, 133, 255}, 16, 2, 128, 16, 512);
}

TEST_F(TestQuantNone, fp16_gqa_large_2) {
  test_span_attn<half>(14, 999, 16, 2, 128, 16, 999);
}
#endif

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b16l2k) {
  test_span_attn<half>(16, 2047, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b16l4k) {
  test_span_attn<half>(16, 4095, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b16l8k) {
  test_span_attn<half>(16, 8191, 32, 4, 128, 32, 8192);
}

//

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b32l2k) {
  test_span_attn<half>(32, 2047, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b32l4k) {
  test_span_attn<half>(32, 4095, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b32l8k) {
  test_span_attn<half>(32, 8191, 32, 4, 128, 32, 8192);
}

//

// representative
TEST_F(TestQuantNone, fp16_gqa_q32g4s32b48l2k) {
  test_span_attn<half>(48, 2047, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b48l4k) {
  test_span_attn<half>(48, 4095, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b48l8k) {
  test_span_attn<half>(48, 8191, 32, 4, 128, 32, 8192);
}

//
TEST_F(TestQuantNone, fp16_gqa_q32g4s32b64l2k) {
  test_span_attn<half>(64, 2047, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b64l4k) {
  test_span_attn<half>(64, 4095, 32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b64l8k) {
  test_span_attn<half>(64, 8191, 32, 4, 128, 32, 8192);
}

//
TEST_F(TestQuantNone, fp16_gqa_q32g4s32b5long) {
  test_span_attn<half>({9000, 9000, 9000, 9000, 9000}, 32, 4, 128, 32, 16384);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b50lmix) {
  test_span_attn<half>(
      {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
       1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000,
       2000, 2000, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000,
       3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000, 4000,
       5000, 5000, 5000, 5000, 5000, 5000, 6000, 6000, 7000, 8000},
      32, 4, 128, 32, 8192);
}

TEST_F(TestQuantNone, fp16_gqa_q32g4s32b50lmixlong) {
  test_span_attn<half>(
      {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,  1000,
       1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000,  2000,
       2000, 2000, 2000, 2000, 2000, 3000, 3000, 3000, 3000,  3000,
       3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000,  4000,
       5000, 5000, 5000, 5000, 5000, 5000, 6000, 6000, 16000, 32000},
      32, 4, 128, 32, 32768);
}
#endif

#ifdef ENABLE_BF16
TEST_F(TestQuantNone, bf16_mha_q2g2s16) {
  test_span_attn<hie::bfloat16>({31, 47}, 2, 2, 128, 16, 64);
}

TEST_F(TestQuantNone, bf16_mha_q32g32s32) {
  test_span_attn<hie::bfloat16>({2047, 4095}, 32, 32, 128, 32, 8192, 1e-2);
}

TEST_F(TestQuantNone, bf16_gqa_q32g4s32) {
  test_span_attn<hie::bfloat16>({2047, 1000}, 32, 4, 128, 32, 8192, 1e-2);
}

// representative
TEST_F(TestQuantNone, bf16_gqa_q40g4s32) {
  test_span_attn<hie::bfloat16>(16, 8191, 40, 4, 128, 32, 8192, 1e-2);
}

TEST_F(TestQuantNone, bf16_gqa_q32g4s32b50lmix) {
  test_span_attn<hie::bfloat16>(
      {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
       1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000,
       2000, 2000, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000,
       3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000, 4000,
       5000, 5000, 5000, 5000, 5000, 5000, 6000, 6000, 7000, 8000},
      32, 4, 128, 32, 8192, 1e-2);
}
#endif

//* NOTE: no reference kernel for FP32
#if 0
TEST_F(TestQuantNone, fp32_mha_q2g2s16) {
  test_span_attn<float>({31, 47}, 2, 2, 128, 16, 64);
}

TEST_F(TestQuantNone, fp32_mha_q32g32s32) {
  test_span_attn<float>({2047, 4095}, 32, 32, 128, 32, 8192, 1e-2);
}

TEST_F(TestQuantNone, fp32_gqa_q32g4s32) {
  test_span_attn<float>({2047, 1000}, 32, 4, 128, 32, 8192, 1e-2);
}

// representative
TEST_F(TestQuantNone, fp32_gqa_q40g4s32) {
  test_span_attn<float>(16, 8191, 40, 4, 128, 32, 8192, 1e-2);
}

TEST_F(TestQuantNone, fp32_gqa_q32g4s32b50lmix) {
  test_span_attn<float>(
      {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
       1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000,
       2000, 2000, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000,
       3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000, 4000,
       5000, 5000, 5000, 5000, 5000, 5000, 6000, 6000, 7000, 8000},
      32, 4, 128, 32, 8192, 1e-2);
}
#endif

// large batch
#ifdef ENABLE_FP16
TEST_F(TestQuantNone, fp16_largebs_q32g4s32b500l2k) {
  test_span_attn<half>(500, 2047, 32, 4, 128, 32, 4096);
}

TEST_F(TestQuantNone, fp16_largebs_q32g4s32b1000l2k) {
  test_span_attn<half>(1000, 2047, 32, 4, 128, 32, 4096);
}
#endif

// long context

#ifdef ENABLE_BF16
TEST_F(TestQuantNone, bf16_longcontext_q32g4s32b2l32k) {
  test_span_attn<hie::bfloat16>({16385, 31}, 32, 4, 128, 32, 32768, 1e-2);
}

TEST_F(TestQuantNone, bf16_longcontext_q32g4s32b50l32kmix) {
  test_span_attn<hie::bfloat16>(
      {11000, 11000, 1000,  1000,  5000,  1000,  1000,  7000,  1000,  1000,
       1000,  1000,  1000,  1000,  12000, 2000,  12000, 2000,  2000,  22000,
       2000,  2000,  2000,  32000, 2000,  23000, 3000,  13000, 3000,  31000,
       3000,  3000,  30000, 4000,  14000, 4000,  4000,  14000, 4000,  24000,
       25000, 5000,  15000, 5000,  15000, 5000,  26000, 6000,  27000, 18000},
      32, 4, 128, 32, 32768, 1e-2);
}

TEST_F(TestQuantNone, bf16_longcontext_q32g4s32b2l128k) {
  test_span_attn<hie::bfloat16>({128000, 1000}, 32, 4, 128, 32, 131072, 1e-2);
}

TEST_F(TestQuantNone, bf16_longcontext_q32g4s32b10l128kmix) {
  test_span_attn<hie::bfloat16>(
      {15000, 25000, 26000, 128000, 4000, 14000, 40000, 100000, 27000, 18000},
      32, 4, 128, 32, 131072, 1e-2);
}
#endif
