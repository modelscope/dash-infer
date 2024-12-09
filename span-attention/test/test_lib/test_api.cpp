/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    test_api.cpp
 */

#include <cuda_runtime.h>
#include <span_attn.h>

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "common.hpp"
#include "framework_utils.hpp"
#include "test_common.h"

namespace {

uint32_t udiv_up(uint32_t seq_len, uint32_t span_len) {
  return (seq_len + span_len - 1) / span_len;
}

void fake_run(void* deviceWorkspace, size_t deviceWsSizeInBytes,
              void* hostWorkspace, size_t hostWsSizeInBytes) {
  EXPECT_LE(hostWsSizeInBytes, deviceWsSizeInBytes);
  EXPECT_EQ(cudaSuccess, cudaMemcpy(deviceWorkspace, hostWorkspace,
                                    hostWsSizeInBytes, cudaMemcpyHostToDevice));
  return;
}

}  // namespace

struct SpanAttnParam {
  SpanAttnParam(std::vector<int> seq_lengths_, int num_heads_, int num_groups_,
                int size_per_head_, int span_size_, int max_length_,
                int device_id_, cudaDeviceProp device_prop_)
      : seq_lengths(std::move(seq_lengths_)),
        batch_size(seq_lengths.size()),
        num_heads(num_heads_),
        num_groups(num_groups_),
        size_per_head(size_per_head_),
        hidden_size(num_heads_ * size_per_head_),
        span_size(span_size_),
        max_length(max_length_),
        max_num_spans(udiv_up(max_length_, span_size_)),
        device_id(device_id_),
        device_prop(std::move(device_prop_)) {}

  const std::vector<int> seq_lengths;
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

class TestApi : public ::testing::Test {
 public:
  void test_api(span::DataType dtype, span::QuantMode qmode, int batch_size,
                int seq_length, int num_heads, int num_groups,
                int size_per_head, int span_size, int max_length) {
// check status code
#define SA_CHECK(expr)                          \
  do {                                          \
    ASSERT_EQ(span::SaStatus::SUCCESS, (expr)); \
  } while (0)

    SpanAttnParam param(std::vector<int>(batch_size, seq_length), num_heads,
                        num_groups, size_per_head, span_size, max_length, 0,
                        device_prop);

    // first, create a handle
    span::SpanAttnHandle_t handle{nullptr};
    SA_CHECK(span::CreateHandle(
        &handle, dtype, qmode, param.batch_size, param.num_heads,
        param.num_groups, param.size_per_head, param.span_size,
        param.max_num_spans, param.seq_lengths.data(), param.device_prop));
    ASSERT_NE(nullptr, handle);

    // next, alloc workspace for span attention kernel
    void* device_workspace{nullptr};
    void* host_workspace{nullptr};
    size_t device_ws_size{0};
    size_t host_ws_size{0};
    SA_CHECK(span::GetDeviceWorkspaceSize(&device_ws_size, handle));
    SA_CHECK(span::GetHostWorkspaceSize(&host_ws_size, handle));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&device_workspace, device_ws_size));
    host_workspace = std::malloc(host_ws_size);
    ASSERT_NE(nullptr, host_workspace);

    // then, run the kernel
    /* SA_CHECK(span::Run(...)); */

    // run a fake kernel to prevent compiler optimizations
    fake_run(device_workspace, device_ws_size, host_workspace, host_ws_size);

    // finally, destroy the handle and free workspace
    SA_CHECK(span::DestroyHandle(handle));
    ASSERT_EQ(cudaSuccess, cudaFree(device_workspace));
    std::free(host_workspace);
    return;

#undef SA_CHECK
  }

  void test_api_fail(span::SaStatus fail_status, span::DataType dtype,
                     span::QuantMode qmode, int batch_size, int seq_length,
                     int num_heads, int num_groups, int size_per_head,
                     int span_size, int max_length) {
    SpanAttnParam param(std::vector<int>(batch_size, seq_length), num_heads,
                        num_groups, size_per_head, span_size, max_length, 0,
                        device_prop);

    // first, create a handle
    span::SpanAttnHandle_t handle{nullptr};
    ASSERT_EQ(fail_status,
              span::CreateHandle(&handle, dtype, qmode, param.batch_size,
                                 param.num_heads, param.num_groups,
                                 param.size_per_head, param.span_size,
                                 param.max_num_spans, param.seq_lengths.data(),
                                 param.device_prop));
    return;
  }

  void test_api_run_fail(span::SaStatus fail_status, span::DataType dtype,
                         span::QuantMode qmode, int batch_size, int seq_length,
                         int num_heads, int num_groups, int size_per_head,
                         int span_size, int max_length) {
    SpanAttnParam param(std::vector<int>(batch_size, seq_length), num_heads,
                        num_groups, size_per_head, span_size, max_length, 0,
                        device_prop);

    // first, create a handle
    span::SpanAttnHandle_t handle{nullptr};
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::CreateHandle(&handle, dtype, qmode, param.batch_size,
                                 param.num_heads, param.num_groups,
                                 param.size_per_head, param.span_size,
                                 param.max_num_spans, param.seq_lengths.data(),
                                 param.device_prop));
    ASSERT_NE(nullptr, handle);

    void* device_workspace{nullptr};
    void* host_workspace{nullptr};
    size_t device_ws_size{0};
    size_t host_ws_size{0};
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetDeviceWorkspaceSize(&device_ws_size, handle));
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetHostWorkspaceSize(&host_ws_size, handle));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&device_workspace, device_ws_size));
    host_workspace = std::malloc(host_ws_size);
    ASSERT_NE(nullptr, host_workspace);

    void* output = reinterpret_cast<void*>(0xdeadbeef);
    void* query = reinterpret_cast<void*>(0xdeadbeef);
    void* kspans = reinterpret_cast<void*>(0xdeadbeef);
    void* vspans = reinterpret_cast<void*>(0xdeadbeef);

    ASSERT_EQ(fail_status,
              span::Run(output, query, static_cast<const void* const*>(kspans),
                        static_cast<const void* const*>(vspans),
                        device_workspace, device_ws_size, host_workspace,
                        host_ws_size, 0.1f, handle, 0));

    ASSERT_EQ(span::SaStatus::SUCCESS, span::DestroyHandle(handle));
    ASSERT_EQ(cudaSuccess, cudaFree(device_workspace));
    std::free(host_workspace);
    return;
  }

  void test_api_null_fail(span::SaStatus fail_status, span::DataType dtype,
                          span::QuantMode qmode, int batch_size, int seq_length,
                          int num_heads, int num_groups, int size_per_head,
                          int span_size, int max_length) {
    SpanAttnParam param(std::vector<int>(batch_size, seq_length), num_heads,
                        num_groups, size_per_head, span_size, max_length, 0,
                        device_prop);

    // first, create a handle
    span::SpanAttnHandle_t handle{nullptr};
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::CreateHandle(&handle, dtype, qmode, param.batch_size,
                                 param.num_heads, param.num_groups,
                                 param.size_per_head, param.span_size,
                                 param.max_num_spans, param.seq_lengths.data(),
                                 param.device_prop));
    ASSERT_NE(nullptr, handle);

    void* device_workspace{nullptr};
    void* host_workspace{nullptr};
    size_t device_ws_size{0};
    size_t host_ws_size{0};
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetDeviceWorkspaceSize(&device_ws_size, handle));
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetHostWorkspaceSize(&host_ws_size, handle));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&device_workspace, device_ws_size));
    host_workspace = std::malloc(host_ws_size);
    ASSERT_NE(nullptr, host_workspace);

    void* output{nullptr};
    void* query{nullptr};
    void* kspans = reinterpret_cast<void*>(0xdeadbeef);
    void* vspans = reinterpret_cast<void*>(0xdeadbeef);

    ASSERT_EQ(fail_status,
              span::Run(output, query, static_cast<const void* const*>(kspans),
                        static_cast<const void* const*>(vspans),
                        device_workspace, device_ws_size, host_workspace,
                        host_ws_size, 0.1f, handle, 0));

    ASSERT_EQ(span::SaStatus::SUCCESS, span::DestroyHandle(handle));
    ASSERT_EQ(cudaSuccess, cudaFree(device_workspace));
    std::free(host_workspace);
    return;
  }

  void test_api_workspace_fail(span::SaStatus fail_status, span::DataType dtype,
                               span::QuantMode qmode, int batch_size,
                               int seq_length, int num_heads, int num_groups,
                               int size_per_head, int span_size,
                               int max_length) {
    SpanAttnParam param(std::vector<int>(batch_size, seq_length), num_heads,
                        num_groups, size_per_head, span_size, max_length, 0,
                        device_prop);

    // first, create a handle
    span::SpanAttnHandle_t handle{nullptr};
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::CreateHandle(&handle, dtype, qmode, param.batch_size,
                                 param.num_heads, param.num_groups,
                                 param.size_per_head, param.span_size,
                                 param.max_num_spans, param.seq_lengths.data(),
                                 param.device_prop));
    ASSERT_NE(nullptr, handle);

    void* device_workspace{nullptr};
    void* host_workspace{nullptr};
    size_t device_ws_size{0};
    size_t host_ws_size{0};
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetDeviceWorkspaceSize(&device_ws_size, handle));
    ASSERT_EQ(span::SaStatus::SUCCESS,
              span::GetHostWorkspaceSize(&host_ws_size, handle));

    void* output = reinterpret_cast<void*>(0xdeadbeef);
    void* query = reinterpret_cast<void*>(0xdeadbeef);
    void* kspans = reinterpret_cast<void*>(0xdeadbeef);
    void* vspans = reinterpret_cast<void*>(0xdeadbeef);

    ASSERT_EQ(fail_status,
              span::Run(output, query, static_cast<const void* const*>(kspans),
                        static_cast<const void* const*>(vspans),
                        device_workspace, device_ws_size, host_workspace,
                        host_ws_size, 0.1f, handle, 0));

    ASSERT_EQ(span::SaStatus::SUCCESS, span::DestroyHandle(handle));
    return;
  }

 protected:
  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&device_prop, 0));
    return;
  }
  void TearDown() override {}

 protected:
  cudaDeviceProp device_prop;
};

TEST_F(TestApi, fp32_mha) {
  test_api(span::DataType::FP32, span::QuantMode::NONE, 2, 31, 2, 2, 128, 16,
           64);
}

TEST_F(TestApi, fp32_gqa) {
  test_api(span::DataType::FP32, span::QuantMode::NONE, 14, 999, 16, 2, 128, 16,
           999);
}

TEST_F(TestApi, fp32_gqa_i8) {
  test_api(span::DataType::FP32, span::QuantMode::I8, 14, 999, 16, 2, 128, 16,
           999);
}

TEST_F(TestApi, fp32_gqa_u4) {
  test_api(span::DataType::FP32, span::QuantMode::U4, 14, 999, 16, 2, 128, 16,
           999);
}

#ifdef ENABLE_FP16
TEST_F(TestApi, fp16_mha) {
  test_api(span::DataType::FP16, span::QuantMode::NONE, 2, 31, 2, 2, 128, 16,
           64);
}

TEST_F(TestApi, fp16_gqa) {
  test_api(span::DataType::FP16, span::QuantMode::NONE, 14, 999, 16, 2, 128, 16,
           999);
}

TEST_F(TestApi, fp16_gqa_i8) {
  test_api(span::DataType::FP16, span::QuantMode::I8, 14, 999, 16, 2, 128, 16,
           999);
}

TEST_F(TestApi, fp16_gqa_u4) {
  test_api(span::DataType::FP16, span::QuantMode::U4, 14, 999, 16, 2, 128, 16,
           999);
}
#endif

#ifdef ENABLE_BF16
TEST_F(TestApi, bf16_mha) {
  test_api(span::DataType::BF16, span::QuantMode::NONE, 2, 31, 2, 2, 128, 16,
           64);
}

TEST_F(TestApi, bf16_gqa) {
  test_api(span::DataType::BF16, span::QuantMode::NONE, 14, 999, 16, 2, 128, 16,
           999);
}

TEST_F(TestApi, bf16_gqa_i8) {
  test_api(span::DataType::BF16, span::QuantMode::I8, 14, 999, 16, 2, 128, 16,
           999);
}

TEST_F(TestApi, bf16_gqa_u4) {
  test_api(span::DataType::BF16, span::QuantMode::U4, 14, 999, 16, 2, 128, 16,
           999);
}
#endif

// fail tests

TEST_F(TestApi, fail_fp32_gqa) {
  test_api_fail(span::SaStatus::EXCEED_LIMIT_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 181239920, 16, 2, 128, 16,
                181239922);
}

#ifndef ENABLE_FP16
TEST_F(TestApi, fail_fp16_type) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP16,
                span::QuantMode::NONE, 14, 999, 16, 2, 128, 16, 999);
}
#endif

#ifndef ENABLE_BF16
TEST_F(TestApi, fail_bf16_type) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::BF16,
                span::QuantMode::NONE, 14, 999, 16, 2, 128, 16, 999);
}
#endif

TEST_F(TestApi, fail_fp32_heads_1) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, 15, 2, 128, 16, 1024);
}

TEST_F(TestApi, fail_fp32_heads_2) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, -8, 2, 128, 512, 1024);
}

TEST_F(TestApi, fail_fp32_heads_3) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, 8, -1, 128, 512, 1024);
}

TEST_F(TestApi, fail_fp32_head_size) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, 8, 2, 512, 512, 1024);
}

TEST_F(TestApi, fail_fp32_zero) {
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 0, 1024, 8, 2, 128, 32, 1024);
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, 0, 2, 128, 32, 1024);
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, 8, 0, 128, 32, 1024);
  test_api_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                span::QuantMode::NONE, 14, 1024, 8, 2, 0, 32, 1024);
}

TEST_F(TestApi, fail_fp32_head_too_many) {
  test_api_run_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                    span::QuantMode::NONE, 14, 1024, 68, 2, 128, 16, 1024);
}

TEST_F(TestApi, fail_fp32_chunk_invalid) {
  test_api_run_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                    span::QuantMode::NONE, 14, 1024, 8, 2, 128, 512, 1024);
}

TEST_F(TestApi, fail_fp32_null_input) {
  test_api_null_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                     span::QuantMode::NONE, 14, 1024, 8, 2, 128, 16, 1024);
}

TEST_F(TestApi, fail_fp32_null_workspace) {
  test_api_workspace_fail(span::SaStatus::PARAM_ERROR, span::DataType::FP32,
                          span::QuantMode::NONE, 14, 1024, 8, 2, 128, 16, 1024);
}
