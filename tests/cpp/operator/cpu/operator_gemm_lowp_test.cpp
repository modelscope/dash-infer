/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator_gemm_lowp_test.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include <core/operator/general/gemm_lowp/gemm_a16w8_arm.h>

#include "../test_operator_utils.h"
namespace AS_UTEST {

template <typename FT, typename QT>
void CPU_SubC_Ref(std::vector<FT>& Ahost, std::vector<QT>& Bhost,
                  std::vector<FT>& BShost, std::vector<FT>& BZhost,
                  std::vector<FT>& ChostRef, const uint32_t M, const uint32_t N,
                  const uint32_t K, const uint32_t GroupSize) {
#pragma omp parallel for collapse(2)
  for (uint32_t mi = 0; mi < M; ++mi) {
    for (uint32_t ni = 0; ni < N; ++ni) {
      float sum_t = float(0);
      for (uint32_t ki = 0; ki < K; ++ki) {
        float tmp = (float(Bhost[ki * N + ni]) -
                     float(BZhost[ki / GroupSize * N + ni])) *
                    float(BShost[ki / GroupSize * N + ni]);
        sum_t += float(Ahost[mi * K + ki]) * tmp;
      }
      ChostRef[mi * N + ni] = FT(sum_t);
    }
  }
}

template <typename FT, typename QT>
void CPU_PerC_Ref(std::vector<FT>& Ahost, std::vector<QT>& Bhost,
                  std::vector<FT>& BShost, std::vector<FT>& BZhost,
                  std::vector<FT>& ChostRef, const uint32_t M, const uint32_t N,
                  const uint32_t K) {
#pragma omp parallel for collapse(2)
  for (uint32_t mi = 0; mi < M; ++mi) {
    for (uint32_t ni = 0; ni < N; ++ni) {
      float sum_t = float(0);
      for (uint32_t ki = 0; ki < K; ++ki) {
        float tmp =
            (float(Bhost[ki * N + ni]) - float(BZhost[ni])) * float(BShost[ni]);
        sum_t += float(Ahost[mi * K + ki]) * tmp;
      }
      ChostRef[mi * N + ni] = FT(sum_t);
    }
  }
}

template <typename FT, typename QT>
void TestGemmA16W8(const int BS, const int M, const int N, const int K,
                   const int GroupSize, const float EPS = 1e-1) {
  const int NumGroup = GroupSize == -1 ? 1 : (K + GroupSize - 1) / GroupSize;

  std::vector<FT> Ahost(BS * M * K);
  std::vector<QT> Bhost(N * K);
  std::vector<FT> BShost(N * NumGroup);
  std::vector<FT> BZhost(N * NumGroup);
  std::vector<FT> Chost(BS * M * N);
  std::vector<FT> ChostRef(BS * M * N);

  const float UPPER = 1.0f;
  const float LOWER = -1.0f;
  const float SRANGE = (UPPER - LOWER) / 256;

  generate_random_data<FT>(Ahost, BS * M * K, FT(LOWER), FT(UPPER));
  generate_random_data<QT>(Bhost, N * K, QT(LOWER), QT(UPPER));
  generate_random_data<FT>(BShost, N * NumGroup, SRANGE * 0.9f, SRANGE * 1.1f);
  generate_random_data<FT>(BZhost, N * NumGroup, -10.0f, 10.0f);

  std::vector<int64_t> AShape = {BS, M, K};
  std::vector<int64_t> BShape = {K, N};
  std::vector<int64_t> BSShape = {NumGroup, N};
  std::vector<int64_t> BZShape = {NumGroup, N};
  std::vector<int64_t> CShape = {BS, M, N};
  if (GroupSize == -1) {
    CPU_PerC_Ref<FT, QT>(Ahost, Bhost, BShost, BZhost, ChostRef, BS * M, N, K);
  } else {
    CPU_SubC_Ref<FT, QT>(Ahost, Bhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         GroupSize);
  }

  const allspark::DeviceType device_type = allspark::DeviceType::CPU;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type =
      allspark::DataTypeTrait<QT>::data_type;

  // Test
  TestOpUtil tu(device_type);
  tu.SetOpType("GemmA16W8");
  tu.SetOpAttribute<float>("alpha", 1.0f);
  tu.SetOpAttribute<bool>("is_pooler", false);
  if (GroupSize != -1) {
    tu.SetOpAttribute<int>("GroupSize", GroupSize);
  }

  tu.AddInput("input", AShape, device_type, ft_data_type, data_mode, Ahost,
              false);
  tu.AddInput("weight", BShape, device_type, qt_data_type, data_mode, Bhost,
              true);
  tu.AddInput("scales", BSShape, device_type, ft_data_type, data_mode, BShost,
              true);
  tu.AddInput("zeros", BZShape, device_type, ft_data_type, data_mode, BZhost,
              true);
  tu.AddOutput("ouput", CShape, device_type, ft_data_type, data_mode);

  allspark::GemmA16W8ARM op;
  allspark::TensorMap weight_buffer;
  op.InitV2(tu.GetOpProto(), *(tu.GetDeviceContext()), tu.GetWeightMap(),
            weight_buffer, &(tu.GetTensorMap()));
  op.Reshape();
  op.Forward();
  tu.device_context_->Synchronize();

  auto output_tensor = tu.GetTensorMap()["ouput"];
  output_tensor->CopyDataTo(
      Chost.data(), output_tensor->GetShape().Count() * sizeof(FT),
      allspark::DeviceType::CPU, tu.GetDeviceContext().get());
  float max_diff =
      check_equal<FT>(ChostRef.data(), Chost.data(), ChostRef.size());

  if (max_diff > EPS) printf("max_diff: %f, EPS: %f\n", max_diff, EPS);
  EXPECT_EQ(max_diff <= EPS, true);
}

TEST(GEMM_LOWP, BF16W8) {
  // SubC
  TestGemmA16W8<float, uint8_t>(1, 1, 5120, 8192, 64, 0.7);
  TestGemmA16W8<float, uint8_t>(1, 3, 5120, 8192, 64, 0.7);
  TestGemmA16W8<float, uint8_t>(1, 5, 5120, 8192, 64, 0.7);
  TestGemmA16W8<float, uint8_t>(1, 7, 5120, 8192, 64, 0.7);
  TestGemmA16W8<float, uint8_t>(1, 128, 5120, 8192, 128, 0.7);
  TestGemmA16W8<float, uint8_t>(1, 1, 5120, 8192, 256, 0.7);
  TestGemmA16W8<float, uint8_t>(1, 3, 8192, 5120, 64, 0.7);
}

TEST(GEMM_LOWP, BF16W8_Odd_Shape) {
  // Test unfriendly shapes
  TestGemmA16W8<float, uint8_t>(1, 17, 2560, 8192, 64, 0.6);
  TestGemmA16W8<float, uint8_t>(1, 31, 5125, 8192, 128, 0.6);
  TestGemmA16W8<float, uint8_t>(1, 1, 5120, 8197, 128, 0.6);
  TestGemmA16W8<float, uint8_t>(1, 128, 5125, 8192, 128, 0.6);
  TestGemmA16W8<float, uint8_t>(1, 999, 2560, 8197, 256, 0.6);
  TestGemmA16W8<float, uint8_t>(1, 3, 8197, 5125, 64, 0.6);
}
}  // namespace AS_UTEST
#endif
