/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_softmax_test.cpp
 */

#include <test_common.h>

#include <cmath>
#include <common.hpp>
#include <vector>

#include "core/kernel/cuda/attention/softmax.hpp"
#define ENABLE_SoftmaxBasicTest 1
#define ENABLE_SoftmaxWithMaskTest 1
#define ENABLE_SoftmaxLognWithMaskTest 1
#define ENABLE_SoftmaxFuseLogTest 1
#define ENABLE_SoftmaxDecoderTest 1
namespace {

class SoftmaxBasicTest : public ::testing::Test {
 public:
  template <typename FT>
  std::vector<FT> reference(const std::vector<FT>& batch_align, int batch,
                            int align) {
    // assert(batch_align.size() == batch * align);
    std::vector<FT> softmax(batch * align, static_cast<FT>(0.f));
    for (int bidx = 0; bidx < batch; bidx++) {
      std::vector<float> data_align(align, 0.f);
      float expf_sum = 0.f;
      for (int aidx = 0; aidx < align; aidx++) {
        data_align[aidx] =
            exp(static_cast<float>(batch_align[bidx * align + aidx]));
        expf_sum += data_align[aidx];
      }
      for (int aidx = 0; aidx < align; aidx++) {
        softmax[bidx * align + aidx] =
            static_cast<FT>(data_align[aidx] / expf_sum);  // [0, 1]
      }
    }
    return softmax;
  }

  template <typename FT>
  void run_test(int batch, int align, float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, input0, FT, batch, align);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * align, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("input0").get(),
                     data_in0.size() * sizeof(FT), stream);

    // run.
    allspark::cuda::SoftmaxKernelLauncher<FT>(
        static_cast<FT*>(tm.at("input0")->GetDataPtr()), nullptr, batch, 1, 1,
        1, align, stream);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("input0").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref = reference<FT>(data_in0, batch, align);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_fallback(int batch, int align, float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputf, FT, batch, align);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * align, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputf").get(),
                     data_in0.size() * sizeof(FT), stream);

    // run.
    allspark::cuda::softmax_4d_fallback(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()), nullptr, 1.f, batch, 1,
        1, align, 1, 1, 1, 1,
        false,         // decoder mask
        false,         // mask with 1/0
        false,         // fuse log
        false, 0, 0);  // attn logn
    std::vector<FT> data_rfb = data_in0;
    common::AsyncD2H(tm.at("inputf").get(), data_rfb.data(),
                     data_rfb.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref = reference<FT>(data_in0, batch, align);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_rfb, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);
    return;
  }

  template <typename FT>
  void run_unroll(int batch, int align, float feps = 1e-2) {
    // test only valid cases.
    if (!allspark::cuda::softmax_unroll_nomask_valid(
            common::toDataType<FT>::cuda_t, 1.f, batch, 1, align, false, 0,
            false))
      return;
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputr, FT, batch, align);
    CREATE_TENSOR(tm, inputu, FT, batch, align);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * align, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputr").get(),
                     data_in0.size() * sizeof(FT), stream);
    common::AsyncH2D(data_in0.data(), tm.at("inputu").get(),
                     data_in0.size() * sizeof(FT), stream);

    // run.
    allspark::cuda::softmax_4d_test_only(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()), nullptr, 1.f, batch, 1,
        1, align, 1, 1, 1, 1,
        false,         // decoder mask
        false,         // mask with 1/0
        false,         // fuse log
        false, 0, 0);  // attn logn
    allspark::cuda::softmax_unroll_nomask(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()), 1.f, batch, 1,
        align,     // [batch, align]
        false, 0,  // attn logn,
        false);    // fuse log
    std::vector<FT> data_ref = data_in0;
    common::AsyncD2H(tm.at("inputr").get(), data_ref.data(),
                     data_ref.size() * sizeof(FT), stream);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("inputu").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);
    return;
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    return;
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<allspark::DeviceContext> device;
};  // SoftmaxBasicTest

class SoftmaxWithMaskTest : public ::testing::Test {
  /* in our softmax definition, mask layout should be [batch / beams, xseql,
   * cstep] however, in the current implementation, mask not support beam
   * search. and require beams == 1. our input mask require datatye = float,
   * with 1/0 only.
   */
 public:
  template <typename FT>
  std::vector<FT> reference(
      const std::vector<FT>& data,     // [batch,  xseql, nhead, cstep]
      const std::vector<float>& mask,  // [batch / beams, xseql, cstep]
      int batch, int beams, int nhead, int xseql, int cstep) {
    // assert(batch_align.size() == batch * align);
    std::vector<FT> softmax = data;
    for (int bidx = 0; bidx < batch; bidx++) {
      int32_t midx = bidx / beams;
      for (int xidx = 0; xidx < xseql; xidx++) {
        for (int nidx = 0; nidx < nhead; nidx++) {
          std::vector<float> data_cstep(cstep, 0.f);
          float expf_sum = 0.f;
#if 0
                    for (int cidx = 0; cidx < cstep; cidx++) {
                        int data_index = 
                                bidx * xseql * nhead * cstep +
                                        xidx * nhead * cstep +
                                                nidx * cstep +
                                                        cidx;
                        int mask_index =
                                midx * xseql * cstep +
                                        xidx * cstep +
                                                cidx;
                        data_cstep[cidx] = exp(
                            static_cast<float>(data[data_index]) + (1 - mask[mask_index]) * -1e10);
                        expf_sum += data_cstep[cidx];
                    }
                    for (int cidx = 0; cidx < cstep; cidx++) {
                        int data_index = 
                                bidx * xseql * nhead * cstep +
                                        xidx * nhead * cstep +
                                                nidx * cstep +
                                                        cidx;
                        softmax[data_index] = static_cast<FT>(data_cstep[cidx] / expf_sum);
                    }
#else
          float data_max = -INFINITY;
          for (int cidx = 0; cidx < cstep; cidx++) {
            int data_index = bidx * xseql * nhead * cstep +
                             xidx * nhead * cstep + nidx * cstep + cidx;
            int mask_index = midx * xseql * cstep + xidx * cstep + cidx;
            data_cstep[cidx] = static_cast<float>(data[data_index]) +
                               (1 - mask[mask_index]) * -1e10;
            data_max = std::max(data_max, data_cstep[cidx]);
          }
          for (int cidx = 0; cidx < cstep; cidx++) {
            float ori = data_cstep[cidx];
            data_cstep[cidx] = exp(data_cstep[cidx] - data_max);
            expf_sum += data_cstep[cidx];
          }
          for (int cidx = 0; cidx < cstep; cidx++) {
            int data_index = bidx * xseql * nhead * cstep +
                             xidx * nhead * cstep + nidx * cstep + cidx;
            softmax[data_index] = static_cast<FT>(data_cstep[cidx] / expf_sum);
          }
#endif
        }
      }
    }
    return softmax;
  }

  template <typename FT>
  void run_test(int batch, int beams, int nhead, int xseql, int cstep,
                bool mask, float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, input0, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch / beams, xseql, cstep);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * xseql * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("input0").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch / beams * xseql * cstep, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = mask && data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // run.
    allspark::cuda::SoftmaxKernelLauncher<FT>(
        static_cast<FT*>(tm.at("input0")->GetDataPtr()),
        mask ? static_cast<float*>(tm.at("inputm")->GetDataPtr()) : nullptr,
        batch, beams, nhead, xseql, cstep, stream);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("input0").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref =
        reference<FT>(data_in0, data_mask, batch, beams, nhead, xseql, cstep);

    // for (int i = 0; i < data_ref.size(); i++) {
    //     float eps =
    //         std::min(std::fabs((float)(data_ref[i] - data_inf[i])),
    //                  std::fabs((float)(data_ref[i] - data_inf[i]) /
    //                  (float)(data_inf[i])));
    //     printf("[%d] ref = %f, inf = %f, eps = %f\n", i, data_ref[i],
    //     data_inf[i], eps);
    // }

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_fallback(int batch, int beams, int nhead, int xseql, int cstep,
                    bool mask, float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputf, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch / beams, xseql, cstep);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * xseql * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputf").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch / beams * xseql * cstep, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = mask && data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // ref.
    allspark::cuda::softmax_4d_fallback(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        mask ? static_cast<float*>(tm.at("inputm")->GetDataPtr()) : nullptr,
        1.f, batch, xseql, nhead, cstep,  // [batch,  xseql, nhead, cstep]
        batch / beams, xseql, 1, cstep,   // [batch / beams, xseql, cstep]
        false,                            // decoder mask
        true,                             // mask with 10
        false,                            // log fuse
        false, 0, 0);                     // logn attn
    std::vector<FT> data_rfb = data_in0;
    common::AsyncD2H(tm.at("inputf").get(), data_rfb.data(),
                     data_rfb.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref =
        reference<FT>(data_in0, data_mask, batch, beams, nhead, xseql, cstep);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_rfb, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_unroll(int batch, int beams, int nhead, int xseql, int cstep,
                  bool mask, float feps = 1e-2) {
    if (!allspark::cuda::softmax_unroll_3dmask_valid(
            common::toDataType<FT>::cuda_t, 1.f, batch, xseql, nhead, cstep,
            batch, xseql, cstep, false, true, false, 0)) {
      printf("current case is not suitable for this unroll kernel.\n");
      return;
    }
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputu, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputr, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch / beams, xseql, cstep);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * xseql * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputu").get(),
                     data_in0.size() * sizeof(FT), stream);
    common::AsyncH2D(data_in0.data(), tm.at("inputr").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch / beams * xseql * cstep, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = mask && data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // run.
    allspark::cuda::softmax_unroll_3dmask(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        mask ? static_cast<float*>(tm.at("inputm")->GetDataPtr()) : nullptr,
        1.f, batch, xseql, nhead, cstep,  // [batch,  xseql, nhead, cstep]
        batch / beams, xseql, cstep,      // [batch / beams, xseql, cstep]
        false, true, false, 0);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("inputu").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    allspark::cuda::softmax_4d_fallback(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        mask ? static_cast<float*>(tm.at("inputm")->GetDataPtr()) : nullptr,
        1.f, batch, xseql, nhead, cstep,  // [batch,  xseql, nhead, cstep]
        batch / beams, xseql, 1, cstep,   // [batch / beams, xseql, cstep]
        false,                            // decoder mask
        true,                             // mask with 10
        false,                            // log fuse
        false, 0, 0);                     // logn attn
    std::vector<FT> data_ref = data_in0;
    common::AsyncD2H(tm.at("inputr").get(), data_ref.data(),
                     data_ref.size() * sizeof(FT), stream);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    return;
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<allspark::DeviceContext> device;
};  // SoftmaxWithMaskTest

class SoftmaxLognWithMaskTest : public ::testing::Test {
  /* in our logn-softmax definition, mask layout should be [batch, xseql, cstep]
   * for logn-softmax logic, if sequence longer than xlogn,
   * we will use the following formula to calculate the scale factor:
   *   scale = alpha * logf(xidx, xlogn)
   * otherwise we set it as 1.
   * if logn == 0, assume disable logn logic.
   */
 public:
  template <typename FT>
  std::vector<FT> reference(
      const std::vector<FT>& data,     // [batch, xseql, nhead, cstep]
      const std::vector<float>& mask,  // [batch, xseql, cstep]
      float alpha, int batch, int nhead, int xseql, int cstep, int xlogn) {
    std::vector<FT> softmax = data;
    for (int bidx = 0; bidx < batch; bidx++) {
      for (int xidx = 0; xidx < xseql; xidx++) {
        float scale = xidx > xlogn && xlogn ? logf(xidx) / logf(xlogn) : 1.f;
        for (int nidx = 0; nidx < nhead; nidx++) {
          std::vector<float> data_cstep(cstep, 0.f);
          std::vector<float> mask_cstep(cstep, 0.f);
          std::vector<float> diff_cstep(cstep, 0.f);
          float expf_sum = 0.f;
          float data_max = -INFINITY;
          for (int cidx = 0; cidx < cstep; cidx++) {
            int data_index = bidx * xseql * nhead * cstep +
                             xidx * nhead * cstep + nidx * cstep + cidx;
            int mask_index = bidx * xseql * cstep + xidx * cstep + cidx;
            data_cstep[cidx] = static_cast<float>(data[data_index]);
            mask_cstep[cidx] = (1 - mask[mask_index]) * -1e15;
            diff_cstep[cidx] =
                alpha * scale * data_cstep[cidx] + mask_cstep[cidx];
            data_max = std::max(data_max, diff_cstep[cidx]);
          }
          std::vector<float> expf_cstep(cstep, 0.f);
          for (int cidx = 0; cidx < cstep; cidx++) {
            float ori = diff_cstep[cidx];
            expf_cstep[cidx] = exp(diff_cstep[cidx] - data_max);
            expf_sum += expf_cstep[cidx];
          }
          for (int cidx = 0; cidx < cstep; cidx++) {
            int data_index = bidx * xseql * nhead * cstep +
                             xidx * nhead * cstep + nidx * cstep + cidx;
            softmax[data_index] = static_cast<FT>(expf_cstep[cidx] / expf_sum);
          }
        }
      }
    }
    return softmax;
  }

  template <typename FT>
  void run_test(int batch, int nhead, int xseql, int cstep, int xlogn,
                float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, input0, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch, xseql, cstep);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * xseql * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("input0").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch * xseql * cstep, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // run.
    allspark::cuda::LognSoftmaxKernelLauncher<FT>(
        static_cast<FT*>(tm.at("input0")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), batch, nhead, xseql,
        cstep, xlogn, stream);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("input0").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref = reference<FT>(data_in0, data_mask, 1.f, batch,
                                             nhead, xseql, cstep, xlogn);

    // for (int i = 0; i < data_ref.size(); i++) {
    //     float eps =
    //         std::min(std::fabs((float)(data_ref[i] - data_inf[i])),
    //                  std::fabs((float)(data_ref[i] - data_inf[i]) /
    //                  (float)(data_inf[i])));
    //     printf("[%d] ref = %f, inf = %f, eps = %f\n", i, data_ref[i],
    //     data_inf[i], eps);
    // }

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_fallback(int batch, int nhead, int xseql, int cstep, int xlogn,
                    float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputf, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch, xseql, cstep);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * xseql * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputf").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch * xseql * cstep, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // ref.
    allspark::cuda::softmax_4d_fallback(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), 1.f, batch, xseql,
        nhead, cstep,            // [batch, xseql, nhead, cstep]
        batch, xseql, 1, cstep,  // [batch, xseql,     1, cstep]
        false,                   // decoder mask
        true,                    // mask with 10
        false,                   // log fuse
        true, xlogn, 0);         // logn attn
    std::vector<FT> data_rfb = data_in0;
    common::AsyncD2H(tm.at("inputf").get(), data_rfb.data(),
                     data_rfb.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref = reference<FT>(data_in0, data_mask, 1.f, batch,
                                             nhead, xseql, cstep, xlogn);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_rfb, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_unroll(int batch, int nhead, int xseql, int cstep, int xlogn,
                  float feps = 1e-2) {
    if (!allspark::cuda::softmax_unroll_3dmask_valid(
            common::toDataType<FT>::cuda_t, 1.f, batch, xseql, nhead, cstep,
            batch, xseql, cstep, false, true, true, xlogn)) {
      printf("current case is not suitable for this unroll kernel.\n");
      return;
    }
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputu, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputr, FT, batch, xseql, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch, xseql, cstep);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * xseql * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputu").get(),
                     data_in0.size() * sizeof(FT), stream);
    common::AsyncH2D(data_in0.data(), tm.at("inputr").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch * xseql * cstep, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // run,
    allspark::cuda::softmax_unroll_3dmask(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), 1.f, batch, xseql,
        nhead, cstep,         // [batch, xseql, nhead, cstep]
        batch, xseql, cstep,  // [batch, xseql,        cstep]
        false,                // decoder mask
        true,                 // mask with 10
        true, xlogn);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("inputu").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    allspark::cuda::softmax_4d_test_only(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), 1.f, batch, xseql,
        nhead, cstep,            // [batch, xseql, nhead, cstep]
        batch, xseql, 1, cstep,  // [batch, xseql,     1, cstep]
        false,                   // decoder mask
        true,                    // mask with 10
        false,                   // log fuse
        true, xlogn, 0);         // logn attn
    std::vector<FT> data_ref = data_in0;
    common::AsyncD2H(tm.at("inputr").get(), data_ref.data(),
                     data_ref.size() * sizeof(FT), stream);

    // for (int i = 0; i < data_ref.size(); i++) {
    //     float eps =
    //         std::min(std::fabs((float)(data_ref[i] - data_inf[i])),
    //                  std::fabs((float)(data_ref[i] - data_inf[i]) /
    //                  (float)(data_inf[i])));
    //     printf("[%d] ref = %f, inf = %f, eps = %f\n", i, data_ref[i],
    //     data_inf[i], eps);
    // }

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_inf, data_ref, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    return;
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<allspark::DeviceContext> device;
};  // SoftmaxLognWithMaskTest

class SoftmaxFuseLogTest : public ::testing::Test {
  /* softmax fuse logn
   */
 public:
  template <typename FT>
  std::vector<FT> reference(const std::vector<FT>& data,  // [batch, align]
                            int batch, int align) {
    std::vector<FT> softmax = data;
    for (int bidx = 0; bidx < batch; bidx++) {
      std::vector<float> data_align(align, 0.f);
      float expf_sum = 0.f;
#if 0
            for (int aidx = 0; aidx < align; aidx++) {
                int data_index = bidx * align + aidx;
                data_align[aidx] = exp(static_cast<float>(data[data_index]));
                expf_sum += data_align[aidx];
            }
            for (int aidx = 0; aidx < align; aidx++) {
                int data_index = bidx * align + aidx;
                softmax[data_index] = static_cast<FT>(logf(data_align[aidx] / expf_sum));
            }
#else
      float data_max = -INFINITY;
      for (int aidx = 0; aidx < align; aidx++) {
        int data_index = bidx * align + aidx;
        data_align[aidx] = static_cast<float>(data[data_index]);
        data_max = data_max > data_align[aidx] ? data_max : data_align[aidx];
      }
      for (int aidx = 0; aidx < align; aidx++) {
        expf_sum += expf(data_align[aidx] - data_max);
      }
      float logf_sum = logf(expf_sum + 1e-12);
      for (int aidx = 0; aidx < align; aidx++) {
        int data_index = bidx * align + aidx;
        softmax[data_index] =
            static_cast<FT>(data_align[aidx] - data_max - logf_sum);
      }
#endif
    }
    return softmax;
  }

  template <typename FT>
  void run_test(int batch, int align, float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, input0, FT, batch, align);
    CREATE_TENSOR(tm, output, FT, batch, align);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * align, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("input0").get(),
                     data_in0.size() * sizeof(FT), stream);

    // run.
    allspark::cuda::LogSoftmaxKernelLauncher<FT>(
        static_cast<FT*>(tm.at("input0")->GetDataPtr()),
        static_cast<FT*>(tm.at("output")->GetDataPtr()), batch, align, stream);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("output").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref = reference<FT>(data_in0, batch, align);

    // for (int i = 0; i < data_ref.size(); i++) {
    //     float eps =
    //         std::min(std::fabs((float)(data_ref[i] - data_inf[i])),
    //                  std::fabs((float)(data_ref[i] - data_inf[i]) /
    //                  (float)(data_inf[i])));
    //     printf("[%d] ref = %f, inf = %f, eps = %f\n", i, data_ref[i],
    //     data_inf[i], eps);
    // }

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_fallback(int batch, int align, float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputf, FT, batch, align);
    CREATE_TENSOR(tm, output, FT, batch, align);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * align, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputf").get(),
                     data_in0.size() * sizeof(FT), stream);

    // ref.
    allspark::cuda::softmax_4d_fallback(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()), nullptr, 1.f, batch, 1,
        1, align,      // [batch, align]
        1, 1, 1, 1,    // []
        false,         // decoder mask
        false,         // mask with 10
        true,          // log fuse
        false, 0, 0);  // logn attn
    std::vector<FT> data_rfb = data_in0;
    common::AsyncD2H(tm.at("inputf").get(), data_rfb.data(),
                     data_rfb.size() * sizeof(FT), stream);

    std::vector<FT> data_ref = reference<FT>(data_in0, batch, align);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_rfb, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_unroll(int batch, int align, float feps = 1e-2) {
    if (!allspark::cuda::softmax_unroll_nomask_valid(
            common::toDataType<FT>::cuda_t, 1.f, batch, 1, align, false, 0,
            true)) {
      printf("current case is not suitable for this unroll kernel.\n");
      return;
    }
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputr, FT, batch, align);
    CREATE_TENSOR(tm, inputu, FT, batch, align);
    CREATE_TENSOR(tm, output, FT, batch, align);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * align, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputr").get(),
                     data_in0.size() * sizeof(FT), stream);
    common::AsyncH2D(data_in0.data(), tm.at("inputu").get(),
                     data_in0.size() * sizeof(FT), stream);

    // ref.
    allspark::cuda::softmax_4d_test_only(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()), nullptr, 1.f, batch, 1,
        1, align,      // [batch, align]
        1, 1, 1, 1,    // []
        false,         // decoder mask
        false,         // mask with 10
        true,          // log fuse
        false, 0, 0);  // logn attn
    std::vector<FT> data_ref = data_in0;
    common::AsyncD2H(tm.at("inputr").get(), data_ref.data(),
                     data_ref.size() * sizeof(FT), stream);

    allspark::cuda::softmax_unroll_nomask(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()), 1.f, batch, 1, align,
        false, 0,  // logn attn
        true);     // log fuse
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("inputu").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    return;
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<allspark::DeviceContext> device;
};  // SoftmaxFuseLogTest

class SoftmaxDecoderTest : public ::testing::Test {
  /* softmax fuse logn
   */
 public:
  template <typename FT>
  std::vector<FT> reference(
      const std::vector<FT>& data,     // [batch,             1, nhead, cstep]
      const std::vector<float>& mask,  // [batch / beams, inl-1,     1, inlen]
                                       //                 inlen         - stride
      int batch, int beams, int nhead, int cstep, int inlen) {
    std::vector<FT> softmax = data;
    for (int bidx = 0; bidx < batch; bidx++) {
      int midx = bidx / beams;
      for (int nidx = 0; nidx < nhead; nidx++) {
        std::vector<float> data_cstep(cstep, 0.f);
        float expf_sum = 0.f;
        float data_max = -INFINITY;
        for (int cidx = 0; cidx < cstep; cidx++) {
          int data_index = bidx * nhead * cstep + nidx * cstep + cidx;
          int mask_index = midx * inlen * inlen + (inlen - 1) * inlen + cidx;
          float data_val = static_cast<float>(data[data_index]);
          float mask_val = cidx < inlen ? mask[mask_index] : 1.f;
          data_cstep[cidx] = data_val + (1 - mask_val) * -1e10;
          data_max = data_max > data_cstep[cidx] ? data_max : data_cstep[cidx];
        }
        for (int cidx = 0; cidx < cstep; cidx++) {
          data_cstep[cidx] = exp(data_cstep[cidx] - data_max);
          expf_sum += data_cstep[cidx];
        }
        for (int cidx = 0; cidx < cstep; cidx++) {
          int data_index = bidx * nhead * cstep + nidx * cstep + cidx;
          softmax[data_index] = static_cast<FT>(data_cstep[cidx] / expf_sum);
        }
      }
    }
    return softmax;
  }

  template <typename FT>
  void run_test(int batch, int beams, int nhead, int cstep, int inlen,
                float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, input0, FT, batch, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch / beams, inlen, inlen);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("input0").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch / beams * inlen * inlen, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // run.
    allspark::cuda::DecoderSoftmaxKernelLauncher<FT>(
        static_cast<FT*>(tm.at("input0")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), batch, beams, nhead,
        1, cstep, inlen, stream);
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("input0").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref =
        reference<FT>(data_in0, data_mask, batch, beams, nhead, cstep, inlen);

    // for (int i = 0; i < data_ref.size(); i++) {
    //     float eps =
    //         std::min(std::fabs((float)(data_ref[i] - data_inf[i])),
    //                  std::fabs((float)(data_ref[i] - data_inf[i]) /
    //                  (float)(data_inf[i])));
    //     printf("[%d] ref = %f, inf = %f, eps = %f\n", i, data_ref[i],
    //     data_inf[i], eps);
    // }

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_fallback(int batch, int beams, int nhead, int cstep, int inlen,
                    float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputf, FT, batch, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch / beams, inlen, inlen);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputf").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch / beams * inlen * inlen, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // ref.
    allspark::cuda::softmax_4d_fallback(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputf")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), 1.f, batch, 1,
        nhead, cstep,  // [batch,      1, nhead, cstep]
        batch / beams, inlen, 1,
        inlen,         // [batch / beams,     1, inlen]
                       //                 inlen stride, inlen - 1 as value
        true,          // decoder mask
        true,          // mask with 10
        false,         // log fuse
        false, 0, 0);  // logn attn
    std::vector<FT> data_rfb = data_in0;
    common::AsyncD2H(tm.at("inputf").get(), data_rfb.data(),
                     data_rfb.size() * sizeof(FT), stream);

    // ref.
    std::vector<FT> data_ref =
        reference<FT>(data_in0, data_mask, batch, beams, nhead, cstep, inlen);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_rfb, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);

    return;
  }

  template <typename FT>
  void run_unroll(int batch, int beams, int nhead, int cstep, int inlen,
                  float feps = 1e-2) {
    if (!allspark::cuda::softmax_unroll_3dmask_valid(
            common::toDataType<FT>::cuda_t, 1.f, batch, 1, nhead, cstep,
            batch / beams, inlen, inlen, true, true, false, 0)) {
      printf("current case not suitable for this unroll kernel.\n");
      return;
    }

    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // tensor
    CREATE_TENSOR(tm, inputu, FT, batch, nhead, cstep);
    CREATE_TENSOR(tm, inputr, FT, batch, nhead, cstep);
    CREATE_TENSOR(tm, inputm, float, batch / beams, inlen, inlen);
    CREATE_WORKSPACE(tm, 0);

    // sync
    device->Synchronize();

    // rand
    std::vector<FT> data_in0 =
        common::rand_normal_float<FT>(batch * nhead * cstep, 2.5f);
    common::AsyncH2D(data_in0.data(), tm.at("inputu").get(),
                     data_in0.size() * sizeof(FT), stream);
    common::AsyncH2D(data_in0.data(), tm.at("inputr").get(),
                     data_in0.size() * sizeof(FT), stream);
    std::vector<float> data_mask =
        common::rand_normal_float<float>(batch / beams * inlen * inlen, 2.5f);
    for (int midx = 0; midx < data_mask.size(); midx++)
      data_mask[midx] = data_mask[midx] > 0.5f ? 0.f : 1.f;
    common::AsyncH2D(data_mask.data(), tm.at("inputm").get(),
                     data_mask.size() * sizeof(float), stream);

    // run.
    allspark::cuda::softmax_unroll_3dmask(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputu")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), 1.f, batch, 1,
        nhead, cstep,                 // [batch,             1, nhead, cstep]
        batch / beams, inlen, inlen,  // [batch / beams, inlen,        inlen]
        true,                         // deocoder
        true,                         // with10
        false, 0);                    // logn attn
    std::vector<FT> data_inf = data_in0;
    common::AsyncD2H(tm.at("inputu").get(), data_inf.data(),
                     data_inf.size() * sizeof(FT), stream);

    // ref.
    allspark::cuda::softmax_4d_test_only(
        stream, common::toDataType<FT>::cuda_t,
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<FT*>(tm.at("inputr")->GetDataPtr()),
        static_cast<float*>(tm.at("inputm")->GetDataPtr()), 1.f, batch, 1,
        nhead, cstep,  // [batch,      1, nhead, cstep]
        batch / beams, inlen, 1,
        inlen,         // [batch / beams,     1, inlen]
                       //          inlen stride, inlen - 1 as value
        true,          // decoder mask
        true,          // mask with 10
        false,         // log fuse
        false, 0, 0);  // logn attn
    std::vector<FT> data_ref = data_in0;
    common::AsyncD2H(tm.at("inputr").get(), data_ref.data(),
                     data_ref.size() * sizeof(FT), stream);

    // check
    float max_diff;
    size_t max_idx = -1;
    int num_exc = 0, num_nan = 0;
    common::DiffWithMaxIndex(data_ref, data_inf, feps, max_diff, max_idx,
                             num_exc, num_nan);
    EXPECT_EQ(num_exc == 0, true);
    EXPECT_EQ(num_nan == 0, true);
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    return;
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<allspark::DeviceContext> device;
};  // SoftmaxDecoder

}  // namespace

using bf16 = hie::bfloat16;

#if ENABLE_SoftmaxBasicTest
#define TestSoftmaxBasic(DTYPE, BATCH, ALIGN, EPS)                \
  TEST_F(SoftmaxBasicTest, test##DTYPE##B##BATCH##A##ALIGN) {     \
    run_test<DTYPE>(BATCH, ALIGN, EPS);                           \
  }                                                               \
  TEST_F(SoftmaxBasicTest, fallback##DTYPE##B##BATCH##A##ALIGN) { \
    run_fallback<DTYPE>(BATCH, ALIGN, EPS);                       \
  }                                                               \
  TEST_F(SoftmaxBasicTest, unroll##DTYPE##B##BATCH##A##ALIGN) {   \
    run_unroll<DTYPE>(BATCH, ALIGN, EPS);                         \
  }
// clang-format off
//                  DTYPE,  BATCH,  ALIGN,  EPS
TestSoftmaxBasic(   float,  1,      63,     1e-6);
TestSoftmaxBasic(   float,  2,      1234,   1e-6);
TestSoftmaxBasic(   float,  3,      8191,   1e-6);
TestSoftmaxBasic(   float,  12,     34,     1e-6);
TestSoftmaxBasic(   float,  34,     910,    1e-6);
TestSoftmaxBasic(   float,  16384,  26,     1e-6);
TestSoftmaxBasic(   float,  11,     8193,   1e-4);
TestSoftmaxBasic(   float,  123,    12345,  1e-4);
TestSoftmaxBasic(   float,  7,      16384,  1e-4);
TestSoftmaxBasic(   float,  15,     32768,  1e-4);
TestSoftmaxBasic(   float,  1,      543210, 1e-4);
#ifdef ENABLE_FP16
TestSoftmaxBasic(   half,   1,      7,      3e-4);
TestSoftmaxBasic(   half,   12,     34,     3e-4);
TestSoftmaxBasic(   half,   34,     910,    3e-4);
TestSoftmaxBasic(   half,   3,      2048,   3e-4);
TestSoftmaxBasic(   half,   3,      2051,   3e-4);
TestSoftmaxBasic(   half,   16384,  26,     3e-4);
TestSoftmaxBasic(   half,   26,     16384,  3e-4);
TestSoftmaxBasic(   half,   123,    12345,  3e-4);
TestSoftmaxBasic(   half,   1,      543210, 3e-4);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
TestSoftmaxBasic(   bf16,   1,      7,      3e-4);
TestSoftmaxBasic(   bf16,   12,     34,     3e-4);
TestSoftmaxBasic(   bf16,   34,     910,    3e-4);
TestSoftmaxBasic(   bf16,   3,      2048,   3e-4);
TestSoftmaxBasic(   bf16,   3,      2051,   3e-4);
TestSoftmaxBasic(   bf16,   16384,  26,     3e-4);
TestSoftmaxBasic(   bf16,   26,     16384,  3e-4);
TestSoftmaxBasic(   bf16,   123,    12345,  3e-4);
TestSoftmaxBasic(   bf16,   1,      543210, 3e-4);
#endif  // ENABLE_BF16
#undef TestSoftmaxBasic
// clang-format on
#endif

// beams not support in current kernel
// allspark::cuda::SoftmaxKernelLauncher
#if ENABLE_SoftmaxWithMaskTest
#define TestSoftmaxWithMask(DTYPE, BATCH, BEAMS, NHEAD, XSEQL, CSTEP, EPS)    \
  TEST_F(SoftmaxWithMaskTest,                                                 \
         test##DTYPE##B##BATCH##B##BEAMS##N##NHEAD##X##XSEQL##C##CSTEP) {     \
    run_test<DTYPE>(BATCH, BEAMS, NHEAD, XSEQL, CSTEP, true, EPS);            \
  }                                                                           \
  TEST_F(SoftmaxWithMaskTest,                                                 \
         fallback##DTYPE##B##BATCH##B##BEAMS##N##NHEAD##X##XSEQL##C##CSTEP) { \
    run_fallback<DTYPE>(BATCH, BEAMS, NHEAD, XSEQL, CSTEP, true, EPS);        \
  }                                                                           \
  TEST_F(SoftmaxWithMaskTest,                                                 \
         unroll##DTYPE##B##BATCH##B##BEAMS##N##NHEAD##X##XSEQL##C##CSTEP) {   \
    run_unroll<DTYPE>(BATCH, BEAMS, NHEAD, XSEQL, CSTEP, true, EPS);          \
  }
// clang-format off
//                  DTYPE,  BATCH,  BEAMS,  NHEAD,  XSEQL,  CSTEP,  EPS
TestSoftmaxWithMask(float,  1,      1,      1,      1,      8,      1e-4);
TestSoftmaxWithMask(float,  3,      1,      3,      1,      8192,   1e-4);
TestSoftmaxWithMask(float,  1,      1,      3,      2,      8101,   1e-4);
TestSoftmaxWithMask(float,  2,      1,      3,      12,     32,     1e-4);
TestSoftmaxWithMask(float,  4,      1,      9,      12,     16,     1e-4);
TestSoftmaxWithMask(float,  2,      1,      4,      1234,   1234,   1e-4);
TestSoftmaxWithMask(float,  8,      1,      1,      12,     32,     1e-4);
TestSoftmaxWithMask(float,  3,      1,      4,      3,      7,      1e-4);
TestSoftmaxWithMask(float,  3,      1,      1,      128,    9216,   3e-4);
TestSoftmaxWithMask(float,  1,      1,      12,     32,     16384,  3e-4);
TestSoftmaxWithMask(float,  2,      1,      5,      112,    12345,  3e-4);
TestSoftmaxWithMask(float,  3,      1,      8,      16,     32768,  3e-4);
#ifdef ENABLE_FP16
TestSoftmaxWithMask(half,   1,      1,      1,      16,     16,     3e-3);
TestSoftmaxWithMask(half,   2,      1,      3,      7,      11,     3e-3);
TestSoftmaxWithMask(half,   2,      1,      4,      1234,   1234,   3e-3);
TestSoftmaxWithMask(half,   3,      1,      1,      128,    9216,   3e-3);
TestSoftmaxWithMask(half,   3,      1,      8,      16,     32768,  3e-3);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
TestSoftmaxWithMask(bf16,   1,      1,      1,      16,     16,     3e-3);
TestSoftmaxWithMask(bf16,   2,      1,      3,      7,      11,     3e-3);
TestSoftmaxWithMask(bf16,   2,      1,      4,      1234,   1234,   3e-3);
TestSoftmaxWithMask(bf16,   3,      1,      1,      128,    9216,   3e-3);
TestSoftmaxWithMask(bf16,   3,      1,      8,      16,     32768,  3e-3);
#endif  // ENABLE_BF16
#undef TestSoftmaxWithMask
// clang-format on
#endif

#if ENABLE_SoftmaxLognWithMaskTest
#define TestSoftmaxLogn(DTYPE, BATCH, NHEAD, XSEQL, CSTEP, XLOGN, EPS)        \
  TEST_F(SoftmaxLognWithMaskTest,                                             \
         test##DTYPE##B##BATCH##N##NHEAD##X##XSEQL##C##CSTEP##L##XLOGN) {     \
    run_test<DTYPE>(BATCH, NHEAD, XSEQL, CSTEP, XLOGN, EPS);                  \
  }                                                                           \
  TEST_F(SoftmaxLognWithMaskTest,                                             \
         fallback##DTYPE##B##BATCH##N##NHEAD##X##XSEQL##C##CSTEP##L##XLOGN) { \
    run_fallback<DTYPE>(BATCH, NHEAD, XSEQL, CSTEP, XLOGN, EPS);              \
  }                                                                           \
  TEST_F(SoftmaxLognWithMaskTest,                                             \
         unroll##DTYPE##B##BATCH##N##NHEAD##X##XSEQL##C##CSTEP##L##XLOGN) {   \
    run_unroll<DTYPE>(BATCH, NHEAD, XSEQL, CSTEP, XLOGN, EPS);                \
  }
// clang-format off
//              DTYPE,  BATCH,  NHEAD,  XSEQL,  CSTEP,  XLOGN,  EPS
TestSoftmaxLogn(float,  1,      1,      4,      8,      4,      1e-5);
TestSoftmaxLogn(float,  3,      1,      32,     8192,   32,     1e-5);
TestSoftmaxLogn(float,  3,      3,      8,      16384,  4,      3e-3);
TestSoftmaxLogn(float,  1,      1,      256,    128,    128,    1e-5);
TestSoftmaxLogn(float,  2,      1,      32,     16,     16,     1e-5);
TestSoftmaxLogn(float,  4,      1,      16,     16,     32,     1e-5);
TestSoftmaxLogn(float,  8,      1,      32,     16,     32,     1e-5);
TestSoftmaxLogn(float,  2,      3,      16,     32,     0,      1e-5);
TestSoftmaxLogn(float,  3,      1,      4,      7,      0,      1e-5);
TestSoftmaxLogn(float,  1,      1,      3,      65504,  32768,  3e-3);
TestSoftmaxLogn(float,  2,      3,      32,     8192,   8192,   3e-3);
TestSoftmaxLogn(float,  2,      3,      12,     12345,  4096,   3e-3);
#ifdef ENABLE_FP16
TestSoftmaxLogn(half,   3,      3,      52,     27,     0,      3e-3);
TestSoftmaxLogn(half,   3,      3,      52,     27,     34,     3e-3);
TestSoftmaxLogn(half,   3,      3,      52,     27,     72,     3e-3);
TestSoftmaxLogn(half,   1,      1,      3,      65504,  32768,  3e-3);
TestSoftmaxLogn(half,   2,      3,      12,     12345,  4096,   3e-3);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
TestSoftmaxLogn(bf16,   3,      3,      52,     27,     0,      3e-3);
TestSoftmaxLogn(bf16,   3,      3,      52,     27,     34,     3e-3);
TestSoftmaxLogn(bf16,   3,      3,      52,     27,     72,     3e-3);
TestSoftmaxLogn(bf16,   1,      1,      3,      65504,  32768,  3e-3);
TestSoftmaxLogn(bf16,   2,      3,      12,     12345,  4096,   3e-3);
#endif  // ENABLE_BF16
#undef TestSoftmaxLogn
// clang-format on
#endif

#if ENABLE_SoftmaxFuseLogTest
#define TestSoftmaxFuseLog(DTYPE, BATCH, ALIGN, EPS)                \
  TEST_F(SoftmaxFuseLogTest, test##DTYPE##B##BATCH##A##ALIGN) {     \
    run_test<DTYPE>(BATCH, ALIGN, EPS);                             \
  }                                                                 \
  TEST_F(SoftmaxFuseLogTest, fallback##DTYPE##B##BATCH##A##ALIGN) { \
    run_fallback<DTYPE>(BATCH, ALIGN, EPS);                         \
  }                                                                 \
  TEST_F(SoftmaxFuseLogTest, unroll##DTYPE##B##BATCH##A##ALIGN) {   \
    run_unroll<DTYPE>(BATCH, ALIGN, EPS);                           \
  }
// clang-format off
//                  DTYPE,  BATCH,  Align,  EPS
TestSoftmaxFuseLog( float,  1,      8,      1e-5);
TestSoftmaxFuseLog( float,  32,     256,    1e-5);
TestSoftmaxFuseLog( float,  2,      1048,   1e-5);
TestSoftmaxFuseLog( float,  1234,   1234,   1e-5);
TestSoftmaxFuseLog( float,  3,      16384,  3e-4);
TestSoftmaxFuseLog( float,  13,     54321,  3e-4);
TestSoftmaxFuseLog( float,  11,     65504,  3e-4);
#ifdef ENABLE_FP16
TestSoftmaxFuseLog( half,   3,      16,     3e-4);
TestSoftmaxFuseLog( half,   3,      4099,   3e-4);
TestSoftmaxFuseLog( half,   13,     54321,  3e-4);
TestSoftmaxFuseLog( half,   11,     65504,  3e-4);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
TestSoftmaxFuseLog( bf16,   3,      16,     3e-4);
TestSoftmaxFuseLog( bf16,   3,      4099,   3e-4);
TestSoftmaxFuseLog( bf16,   13,     54321,  3e-4);
TestSoftmaxFuseLog( bf16,   11,     65504,  3e-4);
#endif  // ENABLE_BF16
#undef TestSoftmaxFuseLog
// clang-format on
#endif

#if ENABLE_SoftmaxDecoderTest
#define TestSoftmaxDecoder(DTYPE, BATCH, BEAMS, NHEAD, CSTEP, INLEN, EPS)     \
  TEST_F(SoftmaxDecoderTest,                                                  \
         test##DTYPE##B##BATCH##B##BEAMS##N##NHEAD##C##CSTEP##I##INLEN) {     \
    run_test<DTYPE>(BATCH, BEAMS, NHEAD, CSTEP, INLEN, EPS);                  \
  }                                                                           \
  TEST_F(SoftmaxDecoderTest,                                                  \
         fallback##DTYPE##B##BATCH##B##BEAMS##N##NHEAD##C##CSTEP##I##INLEN) { \
    run_fallback<DTYPE>(BATCH, BEAMS, NHEAD, CSTEP, INLEN, EPS);              \
  }                                                                           \
  TEST_F(SoftmaxDecoderTest,                                                  \
         unroll##DTYPE##B##BATCH##B##BEAMS##N##NHEAD##C##CSTEP##I##INLEN) {   \
    run_unroll<DTYPE>(BATCH, BEAMS, NHEAD, CSTEP, INLEN, EPS);                \
  }
// clang-format off
//                  DTYPE,  BATCH,  BEAMS,  NHEAD,  CSTEP,  INLEN,  EPS
TestSoftmaxDecoder( float,  3,      3,      2,      7,      4,      1e-5);
TestSoftmaxDecoder( float,  4,      2,      1,      8,      4,      1e-5);
TestSoftmaxDecoder( float,  4,      2,      3,      16,     6,      1e-5);
TestSoftmaxDecoder( float,  2,      1,      1,      16,     6,      1e-5);
TestSoftmaxDecoder( float,  8,      2,      5,      32,     40,     1e-5);
TestSoftmaxDecoder( float,  7,      1,      3,      37,     33,     1e-5);
TestSoftmaxDecoder( float,  1,      1,      1,      8192,   2048,   1e-5);
TestSoftmaxDecoder( float,  1,      1,      1,      8192,   8192,   1e-5);
TestSoftmaxDecoder( float,  1,      1,      1,      65504,  1024,   3e-4);
TestSoftmaxDecoder( float,  8,      4,      3,      12345,  512,    3e-4);
#ifdef ENABLE_FP16
TestSoftmaxDecoder( half,   4,      2,      3,      1024,   256,    3e-4);
TestSoftmaxDecoder( half,   4,      2,      3,      1024,   1024,   3e-4);
TestSoftmaxDecoder( half,   4,      2,      3,      1024,   2048,   3e-4);
TestSoftmaxDecoder( half,   1,      1,      1,      65504,  1024,   3e-4);
TestSoftmaxDecoder( half,   8,      4,      3,      12345,  512,    3e-4);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
TestSoftmaxDecoder( bf16,   4,      2,      3,      1024,   256,    3e-4);
TestSoftmaxDecoder( bf16,   4,      2,      3,      1024,   1024,   3e-4);
TestSoftmaxDecoder( bf16,   4,      2,      3,      1024,   2048,   3e-4);
TestSoftmaxDecoder( bf16,   1,      1,      1,      65504,  1024,   3e-4);
TestSoftmaxDecoder( bf16,   8,      4,      3,      12345,  512,    3e-4);
#endif  // ENABLE_BF16
#undef TestSoftmaxDecoder
// clang-format on
#endif
