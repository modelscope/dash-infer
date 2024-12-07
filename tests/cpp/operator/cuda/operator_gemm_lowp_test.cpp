/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator_gemm_lowp_test.cpp
 */

#include <core/operator/general/gemm_lowp/gemm_a16w4_gpu.h>
#include <core/operator/general/gemm_lowp/gemm_a16w8_gpu.h>
#include <core/operator/general/gemm_lowp/gemm_a8w8_gpu.h>
#include <core/operator/general/gemm_lowp/gemm_sparse_a8w8_gpu.h>
#include <cuda_runtime.h>

#include <iostream>

#include "../test_operator_utils.h"

namespace AS_UTEST {

void PackU8ToU4x2(std::vector<uint8_t>& data, std::vector<uint8_t>& data_pack,
                  const int N, const int NPack, const int K) {
  for (int ki = 0; ki < K; ++ki) {
    for (int ni = 0; ni < NPack; ++ni) {
      uint8_t pack_u8 = data[ki * N + ni * 2];
      if ((ni * 2 + 1) < N) {
        pack_u8 |= data[ki * N + ni * 2 + 1] << 4;
      }
      data_pack[ki * NPack + ni] = pack_u8;
    }
  }
}

void ComputeQuantParam(float fmax, float fmin, float qmax, float qmin,
                       float& scale, float& zero_point) {
  scale = (fmax - fmin) / (qmax - qmin);
  zero_point = qmin - fmin / scale;
  zero_point = std::max(qmin, std::min(qmax, zero_point));
}

template <typename FT, typename QT>
void CPU_Quant_Weight_PerC(std::vector<FT>& fdata, std::vector<QT>& qdata,
                           std::vector<FT>& scales, std::vector<FT>& zeros,
                           const uint32_t N, const uint32_t K, const float qmax,
                           const float qmin, const bool& symmetric = false) {
  std::vector<float> fmax(N, -INFINITY);
  std::vector<float> fmin(N, INFINITY);
  std::vector<float> scale_f32(N);
  std::vector<float> zeros_f32(N);

  // Find Max-Min
  for (int ki = 0; ki < K; ++ki) {
    for (int ni = 0; ni < N; ++ni) {
      const float val = static_cast<float>(fdata[ki * N + ni]);
      fmax[ni] = std::max(fmax[ni], val);
      fmin[ni] = std::min(fmin[ni], val);
    }
  }
  // Compute Quant Params
  for (int ni = 0; ni < N; ++ni) {
    if (!symmetric)
      ComputeQuantParam(fmax[ni], fmin[ni], qmax, qmin, scale_f32[ni],
                        zeros_f32[ni]);
    else {
      scale_f32[ni] = std::max(std::abs(fmin[ni]), std::abs(fmax[ni])) /
                      ((qmax - qmin) / 2);
      zeros_f32[ni] = 0.0f;
    }
  }
  // Quantize
  for (int ki = 0; ki < K; ++ki) {
    for (int ni = 0; ni < N; ++ni) {
      float val = static_cast<float>(fdata[ki * N + ni]);
      if (scale_f32[ni] != 0) {
        val = val / scale_f32[ni] + zeros_f32[ni];
      } else {
        assert(val == 0 && zeros_f32[ni] == 0);
      }
      qdata[ki * N + ni] = QT(rintf(std::max(qmin, std::min(qmax, val))));
    }
  }
  for (int ni = 0; ni < N; ++ni) {
    scales[ni] = FT(scale_f32[ni]);
    zeros[ni] = FT(zeros_f32[ni]);
  }
}

template <typename FT, typename QT>
void CPU_Quant_Weight_SubC(std::vector<FT>& fdata, std::vector<QT>& qdata,
                           std::vector<FT>& scales, std::vector<FT>& zeros,
                           const uint32_t N, const uint32_t K,
                           const int GroupSize, const int NumGroup,
                           const float qmax, const float qmin) {
  std::vector<float> fmax(N * NumGroup, -INFINITY);
  std::vector<float> fmin(N * NumGroup, INFINITY);
  std::vector<float> scale_f32(N * NumGroup);
  std::vector<float> zeros_f32(N * NumGroup);

  // Find Max-Min
  for (int ngi = 0; ngi < NumGroup; ++ngi) {
    const int ki_end = std::min(int(K), (ngi + 1) * GroupSize);
    for (int ki = ngi * GroupSize; ki < ki_end; ++ki) {
      for (int ni = 0; ni < N; ++ni) {
        const float val = static_cast<float>(fdata[ki * N + ni]);
        fmax[ngi * N + ni] = std::max(fmax[ngi * N + ni], val);
        fmin[ngi * N + ni] = std::min(fmin[ngi * N + ni], val);
      }
    }
  }
  // Compute Quant Params
  for (int ngi = 0; ngi < NumGroup; ++ngi) {
    for (int ni = 0; ni < N; ++ni) {
      const int idx = ngi * N + ni;
      ComputeQuantParam(fmax[idx], fmin[idx], qmax, qmin, scale_f32[idx],
                        zeros_f32[idx]);
    }
  }
  // Quantize
  for (int ngi = 0; ngi < NumGroup; ++ngi) {
    const int ki_end = std::min(int(K), (ngi + 1) * GroupSize);
    for (int ki = ngi * GroupSize; ki < ki_end; ++ki) {
      for (int ni = 0; ni < N; ++ni) {
        float val = static_cast<float>(fdata[ki * N + ni]);
        val = val / scale_f32[ngi * N + ni] + zeros_f32[ngi * N + ni];
        qdata[ki * N + ni] = QT(rintf(std::max(qmin, std::min(qmax, val))));
      }
    }
  }
  for (int i = 0; i < N * NumGroup; ++i) {
    scales[i] = FT(scale_f32[i]);
    zeros[i] = FT(zeros_f32[i]);
  }
}

template <typename FT, typename QT>
void CPU_SubC_Ref(std::vector<FT>& Ahost, std::vector<QT>& Bhost,
                  std::vector<FT>& BShost, std::vector<FT>& BZhost,
                  std::vector<FT>& ChostRef, const uint32_t M, const uint32_t N,
                  const uint32_t K, const uint32_t GroupSize,
                  const float alpha) {
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
      ChostRef[mi * N + ni] = FT(alpha * sum_t);
    }
  }
}

template <typename FT, typename QT>
void CPU_PerC_Ref(std::vector<FT>& Ahost, std::vector<QT>& Bhost,
                  std::vector<FT>& BShost, std::vector<FT>& BZhost,
                  std::vector<FT>& ChostRef, const uint32_t M, const uint32_t N,
                  const uint32_t K, const float alpha) {
#pragma omp parallel for collapse(2)
  for (uint32_t mi = 0; mi < M; ++mi) {
    for (uint32_t ni = 0; ni < N; ++ni) {
      float sum_t = float(0);
      for (uint32_t ki = 0; ki < K; ++ki) {
        float tmp =
            (float(Bhost[ki * N + ni]) - float(BZhost[ni])) * float(BShost[ni]);
        sum_t += float(Ahost[mi * K + ki]) * tmp;
      }
      ChostRef[mi * N + ni] = FT(alpha * sum_t);
    }
  }
}

template <typename FT>
void CPU_FP16W4_PerC_Ref(std::vector<FT>& Ahost,
                         std::vector<uint8_t>& BPackhost,
                         std::vector<FT>& BShost, std::vector<FT>& BZhost,
                         std::vector<FT>& ChostRef, const uint32_t M,
                         const uint32_t N, const uint32_t K,
                         const uint32_t N_PACK) {
#pragma omp parallel for collapse(2)
  for (uint32_t mi = 0; mi < M; ++mi) {
    for (uint32_t ni = 0; ni < N_PACK; ++ni) {
      float sum_t_low = float(0);
      float sum_t_high = float(0);
      bool guard = (ni * 2 + 1) < N;
      for (uint32_t ki = 0; ki < K; ++ki) {
        uint8_t pack_u8 = BPackhost[ki * N_PACK + ni];
        uint8_t low = pack_u8 & 0xf;
        float tmp_low =
            (float(low) - float(BZhost[ni * 2])) * float(BShost[ni * 2]);
        sum_t_low += float(Ahost[mi * K + ki]) * tmp_low;
        if (guard) {
          uint8_t high = pack_u8 >> 4;
          float tmp_high = (float(high) - float(BZhost[ni * 2 + 1])) *
                           float(BShost[ni * 2 + 1]);
          sum_t_high += float(Ahost[mi * K + ki]) * tmp_high;
        }
      }
      ChostRef[mi * N + ni * 2] = FT(sum_t_low);
      if (guard) {
        ChostRef[mi * N + ni * 2 + 1] = FT(sum_t_high);
      }
    }
  }
}

template <typename T>
void PrintVector(std::vector<T> data, const int Rows, const int Cols) {
  for (int r = 0; r < Rows; ++r) {
    for (int c = 0; c < Cols; ++c) {
      std::cout << data[r * Cols + c] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <>
void PrintVector<half>(std::vector<half> data, const int Rows, const int Cols) {
  for (int r = 0; r < Rows; ++r) {
    for (int c = 0; c < Cols; ++c) {
      std::cout << float(data[r * Cols + c]) << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <>
void PrintVector<int8_t>(std::vector<int8_t> data, const int Rows,
                         const int Cols) {
  for (int r = 0; r < Rows; ++r) {
    for (int c = 0; c < Cols; ++c) {
      std::cout << int(data[r * Cols + c]) << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <typename FT, typename QT>
float TestGemmA8W8(const int BS, const int M, const int N, const int K,
                   const int GroupSize, const float alpha = 1.0f,
                   const float EPS = 5e-2) {
  printf("A8W8: BS=%d, M=%d, N=%d, K=%d, GroupSize=%d\n", BS, M, N, K,
         GroupSize);

  const float qmax = static_cast<const float>(std::numeric_limits<QT>::max());
  const float qmin = static_cast<const float>(std::numeric_limits<QT>::min());

  const int NumGroup = GroupSize == -1 ? 1 : (K + GroupSize - 1) / GroupSize;

  std::vector<FT> Ahost(BS * M * K);
  std::vector<FT> Bhost(N * K);
  std::vector<QT> BQhost(N * K);
  std::vector<FT> BShost(N * NumGroup);
  std::vector<FT> BZhost(N * NumGroup);
  std::vector<FT> Chost(BS * M * N);
  std::vector<FT> ChostRef(BS * M * N);

  const float UPPER = 1.0f;
  const float LOWER = -1.0f;

  generate_random_data<FT>(Ahost, BS * M * K, FT(LOWER), FT(UPPER));
  generate_random_data<FT>(Bhost, K * N, FT(LOWER), FT(UPPER));
  if (GroupSize == -1) {
    CPU_Quant_Weight_PerC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K, qmax,
                                  qmin);
  } else {
    CPU_Quant_Weight_SubC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K,
                                  GroupSize, NumGroup, qmax, qmin);
  }

  std::vector<int64_t> AShape = {BS, M, K};
  std::vector<int64_t> BShape = {K, N};
  std::vector<int64_t> BSShape = {NumGroup, N};
  std::vector<int64_t> BZShape = {NumGroup, N};
  std::vector<int64_t> CShape = {BS, M, N};

  if (GroupSize == -1) {
    CPU_PerC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         alpha);
  } else {
    CPU_SubC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         GroupSize, alpha);
  }

  // PrintVector<FT>(Ahost, M, K);
  // PrintVector<FT>(Bhost, K, N);
  // PrintVector<QT>(BQhost, K, N);
  // PrintVector<FT>(BShost, NumGroup, N);
  // PrintVector<FT>(BZhost, NumGroup, N);

  // std::cout << "ChostRef\n";
  // PrintVector<FT>(ChostRef, M, N);

  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type =
      allspark::DataTypeTrait<QT>::data_type;

  {  // Test
    TestOpUtil tu(device_type);
    tu.SetOpType("GemmA8W8");
    tu.SetOpAttribute<float>("alpha", alpha);
    tu.SetOpAttribute<bool>("is_pooler", false);
    if (GroupSize != -1) {
      tu.SetOpAttribute<int>("GroupSize", GroupSize);
    }

    tu.AddInput("input", AShape, device_type, ft_data_type, data_mode, Ahost,
                false);
    tu.AddInput("weight", BShape, device_type, qt_data_type, data_mode, BQhost,
                true);
    tu.AddInput("scales", BSShape, device_type, ft_data_type, data_mode, BShost,
                true);
    tu.AddInput("zeros", BZShape, device_type, ft_data_type, data_mode, BZhost,
                true);
    tu.AddOutput("ouput", CShape, device_type, ft_data_type, data_mode);

    allspark::GemmA8W8GPU op;
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
    printf("MaxDiff = %f\n", max_diff);
    // EXPECT_EQ(max_diff <= EPS, true);
    return max_diff;
  }
}

#ifdef ENABLE_CUSPARSELT
template <typename FT, typename QT>
float TestGemmSparseA8W8(const int BS, const int M, const int N, const int K,
                         const int GroupSize, bool use_sparse = true,
                         const float alpha = 1.0f, const float EPS = 5e-2) {
  bool symm_quant = true;
  printf(
      "Sparse A8W8: BS=%d, M=%d, N=%d, K=%d, GroupSize=%d, "
      "Weight_symmetric_quantization=%d\n",
      BS, M, N, K, GroupSize, symm_quant);

  const float qmax = static_cast<const float>(std::numeric_limits<QT>::max());
  const float qmin = static_cast<const float>(std::numeric_limits<QT>::min());

  const int NumGroup = GroupSize == -1 ? 1 : (K + GroupSize - 1) / GroupSize;

  std::vector<FT> Ahost(BS * M * K);
  std::vector<FT> Bhost(N * K);
  std::vector<QT> BQhost(N * K);
  std::vector<FT> BShost(N * NumGroup);
  std::vector<FT> BZhost(N * NumGroup);
  std::vector<FT> Chost(BS * M * N);
  std::vector<FT> ChostRef(BS * M * N);

  const float UPPER = 1.0f;
  const float LOWER = -1.0f;
  int seed = 24;
  generate_random_data<FT>(Ahost, BS * M * K, FT(LOWER), FT(UPPER), seed);
  generate_random_data<FT>(Bhost, K * N, FT(LOWER), FT(UPPER), seed);

  if (use_sparse) {
    // mask FP16/BF16 weight value to create 2:4 sparse format
    std::uniform_int_distribution<size_t> distribution(0, 3);
    std::default_random_engine generator;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < K / 4; ++j) {
        int index1 = (j * 4) + distribution(generator);
        int index2 = (j * 4) + distribution(generator);
        while (index1 == index2) {
          index2 = (j * 4) + distribution(generator);
        }
        index1 = index1 * N + i;
        index2 = index2 * N + i;
        Bhost[index1] = 0.0f;
        Bhost[index2] = 0.0f;
      }
    }
  }

  if (GroupSize == -1) {
    CPU_Quant_Weight_PerC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K, qmax,
                                  qmin, symm_quant);
  } else {
    assert(GroupSize == -1);
    CPU_Quant_Weight_SubC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K,
                                  GroupSize, NumGroup, qmax, qmin);
  }

  std::vector<int64_t> AShape = {BS, M, K};
  std::vector<int64_t> BShape = {K, N};
  std::vector<int64_t> BSShape = {NumGroup, N};
  std::vector<int64_t> BZShape = {NumGroup, N};
  std::vector<int64_t> CShape = {BS, M, N};
  //
  if (GroupSize == -1) {
    CPU_PerC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         alpha);
  } else {
    CPU_SubC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         GroupSize, alpha);
  }
  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type =
      allspark::DataTypeTrait<QT>::data_type;

  {  // Test
    TestOpUtil tu(device_type);
    tu.SetOpType("GemmSparseA8W8");
    tu.SetOpAttribute<float>("alpha", alpha);
    tu.SetOpAttribute<bool>("is_pooler", false);
    if (GroupSize != -1) {
      tu.SetOpAttribute<int>("GroupSize", GroupSize);
    }

    std::string weight_name = "weight";
    tu.AddInput("input", AShape, device_type, ft_data_type, data_mode, Ahost,
                false);
    tu.AddInput(weight_name, BShape, device_type, qt_data_type, data_mode,
                BQhost, true);
    tu.AddInput("scales", BSShape, device_type, ft_data_type, data_mode, BShost,
                true);
    tu.AddInput("zeros", BZShape, device_type, ft_data_type, data_mode, BZhost,
                true);
    tu.AddOutput("ouput", CShape, device_type, ft_data_type, data_mode);
    tu.GetDeviceContext()->SetSparsityMatmulMode(true);

    allspark::GemmSparseA8W8GPU op;
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
    printf("MaxDiff = %f, sparse_opt = %d \n", max_diff, op.UseSparseOpt());

    EXPECT_EQ(max_diff <= EPS, true);

    if (use_sparse) EXPECT_EQ(op.UseSparseOpt(), true);

    return max_diff;
  }
}
#endif

template <typename FT, typename QT>
void TestGemmA16W8(const int BS, const int M, const int N, const int K,
                   const int GroupSize, const float alpha = 1.0f,
                   const float EPS = 1e-1) {
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
    CPU_PerC_Ref<FT, QT>(Ahost, Bhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         alpha);
  } else {
    CPU_SubC_Ref<FT, QT>(Ahost, Bhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         GroupSize, alpha);
  }

  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type =
      allspark::DataTypeTrait<QT>::data_type;

  // Test
  TestOpUtil tu(device_type);
  tu.SetOpType("GemmA16W8");
  tu.SetOpAttribute<float>("alpha", alpha);
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

  allspark::GemmA16W8GPU op;
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
  EXPECT_EQ(max_diff <= EPS, true);
}

template <typename FT, typename QT>
float TestGemmA16W8_New(const int BS, const int M, const int N, const int K,
                        const int GroupSize, const float alpha = 1.0f,
                        const float EPS = 5e-2) {
  printf("A16W8: BS=%d, M=%d, N=%d, K=%d, GroupSize=%d\n", BS, M, N, K,
         GroupSize);

  const float qmax = static_cast<const float>(std::numeric_limits<QT>::max());
  const float qmin = static_cast<const float>(std::numeric_limits<QT>::min());

  const int NumGroup = GroupSize == -1 ? 1 : (K + GroupSize - 1) / GroupSize;

  std::vector<FT> Ahost(BS * M * K);
  std::vector<FT> Bhost(N * K);
  std::vector<QT> BQhost(N * K);
  std::vector<FT> BShost(N * NumGroup);
  std::vector<FT> BZhost(N * NumGroup);
  std::vector<FT> Chost(BS * M * N);
  std::vector<FT> ChostRef(BS * M * N);

  const float UPPER = 1.0f;
  const float LOWER = -1.0f;

  generate_random_data<FT>(Ahost, BS * M * K, FT(LOWER), FT(UPPER));
  generate_random_data<FT>(Bhost, K * N, FT(LOWER), FT(UPPER));
  if (GroupSize == -1) {
    CPU_Quant_Weight_PerC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K, qmax,
                                  qmin);
  } else {
    CPU_Quant_Weight_SubC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K,
                                  GroupSize, NumGroup, qmax, qmin);
  }

  std::vector<int64_t> AShape = {BS, M, K};
  std::vector<int64_t> BShape = {K, N};
  std::vector<int64_t> BSShape = {NumGroup, N};
  std::vector<int64_t> BZShape = {NumGroup, N};
  std::vector<int64_t> CShape = {BS, M, N};

  if (GroupSize == -1) {
    CPU_PerC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         alpha);
  } else {
    CPU_SubC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         GroupSize, alpha);
  }

  // PrintVector<FT>(Ahost, M, K);
  // PrintVector<FT>(Bhost, K, N);
  // PrintVector<QT>(BQhost, K, N);
  // PrintVector<FT>(BShost, NumGroup, N);
  // PrintVector<FT>(BZhost, NumGroup, N);

  // std::cout << "ChostRef\n";
  // PrintVector<FT>(ChostRef, M, N);

  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type =
      allspark::DataTypeTrait<QT>::data_type;

  {  // Test
    TestOpUtil tu(device_type);
    tu.SetOpType("GemmA16W8");
    tu.SetOpAttribute<float>("alpha", alpha);
    tu.SetOpAttribute<bool>("is_pooler", false);
    if (GroupSize != -1) {
      tu.SetOpAttribute<int>("GroupSize", GroupSize);
    }

    tu.AddInput("input", AShape, device_type, ft_data_type, data_mode, Ahost,
                false);
    tu.AddInput("weight", BShape, device_type, qt_data_type, data_mode, BQhost,
                true);
    tu.AddInput("scales", BSShape, device_type, ft_data_type, data_mode, BShost,
                true);
    tu.AddInput("zeros", BZShape, device_type, ft_data_type, data_mode, BZhost,
                true);
    tu.AddOutput("ouput", CShape, device_type, ft_data_type, data_mode);

    allspark::GemmA16W8GPU op;
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
    printf("MaxDiff = %f\n", max_diff);
    // EXPECT_EQ(max_diff <= EPS, true);
    return max_diff;
  }
}

template <typename FT>
void TestGemmA16W4(const int BS, const int M, const int N, const int K,
                   const float EPS = 1e-1) {
  const int NumGroup = 1;
  const int N_PACK = (N + 1) / 2;
  std::vector<FT> Ahost(BS * M * K);
  std::vector<uint8_t> Bhost(K * N);
  std::vector<uint8_t> BPackhost(K * N_PACK);
  std::vector<FT> BShost(N * NumGroup);
  std::vector<FT> BZhost(N * NumGroup);
  std::vector<FT> Chost(BS * M * N);
  std::vector<FT> ChostRef(BS * M * N);

  const float UPPER = 1.0f;
  const float LOWER = -1.0f;
  const float SRANGE = (UPPER - LOWER) / 15;

  generate_random_data<FT>(Ahost, BS * M * K, FT(LOWER), FT(UPPER));
  generate_random_data<uint8_t>(Bhost, K * N, uint8_t(0), uint8_t(2));
  generate_random_data<FT>(BShost, N * NumGroup, SRANGE * 0.4f, SRANGE * 0.9f);
  generate_random_data<FT>(BZhost, N * NumGroup, 0.f, 4.0f);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N_PACK; ++j) {
      uint8_t pack_u8 = Bhost[i * N + j * 2];
      if ((j * 2 + 1) < N) {
        pack_u8 |= Bhost[i * N + j * 2 + 1] << 4;
      }
      BPackhost[i * N_PACK + j] = pack_u8;
    }
  }
  std::vector<int64_t> AShape = {BS, M, K};
  std::vector<int64_t> BPackShape = {K, N_PACK};
  std::vector<int64_t> BSShape = {NumGroup, N};
  std::vector<int64_t> BZShape = {NumGroup, N};
  std::vector<int64_t> CShape = {BS, M, N};

  CPU_FP16W4_PerC_Ref<FT>(Ahost, BPackhost, BShost, BZhost, ChostRef, BS * M, N,
                          K, N_PACK);

  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type = DataType::UINT8;

  // Test
  TestOpUtil tu(device_type);
  tu.SetOpType("GemmA16W4");
  tu.SetOpAttribute<float>("alpha", 1.0f);
  tu.SetOpAttribute<bool>("is_pooler", false);

  tu.AddInput("input", AShape, device_type, ft_data_type, data_mode, Ahost,
              false);
  tu.AddInput("weight", BPackShape, device_type, qt_data_type, data_mode,
              BPackhost, true);
  tu.AddInput("scales", BSShape, device_type, ft_data_type, data_mode, BShost,
              true);
  tu.AddInput("zeros", BZShape, device_type, ft_data_type, data_mode, BZhost,
              true);
  tu.AddOutput("ouput", CShape, device_type, ft_data_type, data_mode);

  allspark::GemmA16W4GPU op;
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
  EXPECT_EQ(max_diff <= EPS, true);
}

template <typename FT>
float TestGemmA16W4_New(const int BS, const int M, const int N, const int K,
                        const int GroupSize = -1, const float alpha = 1.0f,
                        const float EPS = 5e-2) {
  printf("A16W4: BS=%d, M=%d, N=%d, K=%d, GroupSize=%d\n", BS, M, N, K,
         GroupSize);

  using QT = uint8_t;

  const float qmax = 15.0f;
  const float qmin = 0.0f;

  const int NumGroup = GroupSize == -1 ? 1 : (K + GroupSize - 1) / GroupSize;
  const int N_PACK = (N + 1) / 2;

  std::vector<FT> Ahost(BS * M * K);
  std::vector<FT> Bhost(N * K);
  std::vector<uint8_t> BQhost(N * K);
  std::vector<uint8_t> BQhostPack(K * N_PACK);
  std::vector<FT> BShost(N * NumGroup);
  std::vector<FT> BZhost(N * NumGroup);
  std::vector<FT> Chost(BS * M * N);
  std::vector<FT> ChostRef(BS * M * N);

  std::vector<int64_t> AShape = {BS, M, K};
  std::vector<int64_t> BShape = {K, N};
  std::vector<int64_t> BPackShape = {K, N_PACK};
  std::vector<int64_t> BSShape = {NumGroup, N};
  std::vector<int64_t> BZShape = {NumGroup, N};
  std::vector<int64_t> CShape = {BS, M, N};

  const float UPPER = 1.0f;
  const float LOWER = -1.0f;

  generate_random_data<FT>(Ahost, BS * M * K, FT(LOWER), FT(UPPER));
  generate_random_data<FT>(Bhost, K * N, FT(LOWER), FT(UPPER));
  if (GroupSize == -1) {
    CPU_Quant_Weight_PerC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K, qmax,
                                  qmin);
    CPU_PerC_Ref<FT, QT>(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                         alpha);
  } else {
    CPU_Quant_Weight_SubC<FT, QT>(Bhost, BQhost, BShost, BZhost, N, K,
                                  GroupSize, NumGroup, qmax, qmin);
    CPU_SubC_Ref(Ahost, BQhost, BShost, BZhost, ChostRef, BS * M, N, K,
                 GroupSize, alpha);
  }

  // Pack BQhost to BQhostPack
  PackU8ToU4x2(BQhost, BQhostPack, N, N_PACK, K);

  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type =
      allspark::DataTypeTrait<FT>::data_type;
  const allspark::DataType qt_data_type = DataType::UINT8;

  // Test
  {
    TestOpUtil tu(device_type);
    tu.SetOpType("GemmA16W4");
    tu.SetOpAttribute<float>("alpha", 1.0f);
    tu.SetOpAttribute<bool>("is_pooler", false);
    if (GroupSize != -1) {
      tu.SetOpAttribute<int>("GroupSize", GroupSize);
    }

    tu.AddInput("input", AShape, device_type, ft_data_type, data_mode, Ahost,
                false);
    tu.AddInput("weight", BPackShape, device_type, qt_data_type, data_mode,
                BQhostPack, true);
    tu.AddInput("scales", BSShape, device_type, ft_data_type, data_mode, BShost,
                true);
    tu.AddInput("zeros", BZShape, device_type, ft_data_type, data_mode, BZhost,
                true);
    tu.AddOutput("ouput", CShape, device_type, ft_data_type, data_mode);

    allspark::GemmA16W4GPU op;
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
    printf("MaxDiff = %f\n", max_diff);
    // EXPECT_EQ(max_diff <= EPS, true);
    return max_diff;
  }
}

TEST(GEMM_LOWP, FP16W4_NEW) {
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  const int sm_version = (device_prop.major << 8 | device_prop.minor);

  const int BS = 1;
  const int TestCases = 5;
  const int Mod = 8;
  const int M_Range[2] = {32, 2048};
  const int N_Range[2] = {8, 8192};
  const int K_Range[2] = {8, 8192};
  const int GS_Range[4] = {64, 128, 256, 512};

  // Test Ampere+ Fused PerC GEMV kernel and Fused SubC GEMV kernel
  if (sm_version >= 0x0800) {
    float ave_diff_perc = 0.0f, ave_diff_subc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      int M = rand() % M_Range[0] + 1;
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[1] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      const int GroupSize = GS_Range[rand() % 4];
      ave_diff_perc += TestGemmA16W4_New<half>(BS, M, N, K, -1, 1.0f, 7e-2);
      ave_diff_subc +=
          TestGemmA16W4_New<half>(BS, M, N, K, GroupSize, 1.0f, 7e-2);
    }
    ave_diff_perc /= TestCases;
    ave_diff_subc /= TestCases;
    printf("Ampere+ Fused GEMV Ave_Diff_Perc : %f\n", ave_diff_perc);
    printf("Ampere+ Fused GEMV Ave_Diff_Subc : %f\n", ave_diff_subc);
    EXPECT_EQ(ave_diff_perc <= 1e-1, true);
    EXPECT_EQ(ave_diff_subc <= 1e-1, true);
  }

  // Test Volta Fused PerC GEMV kernel and Fused SubC GEMV kernel
  if (sm_version == 0x0700) {
    float ave_diff_perc = 0.0f, ave_diff_subc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      int M = rand() % M_Range[0] + 1;
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[1] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      const int GroupSize =
          (((rand() % (GS_Range[3] - GS_Range[0] + 1)) + GS_Range[0]) + Mod -
           1) /
          Mod * Mod;
      // Volta a16w4 tc kernel uses FP16 accumulator, so accuracy error is
      // larger Volta a16w4 perc tc kernel adopts redusum optimization, so the
      // accuracy error is larger than subc tc kernel
      ave_diff_perc += TestGemmA16W4_New<half>(BS, M, N, K, -1, 1.0f, 7e-1);
      ave_diff_subc +=
          TestGemmA16W4_New<half>(BS, M, N, K, GroupSize, 1.0f, 5e-1);
    }
    ave_diff_perc /= TestCases;
    ave_diff_subc /= TestCases;
    printf("Volta Fused GEMV Ave_Diff_Perc : %f\n", ave_diff_perc);
    printf("Volta Fused GEMV Ave_Diff_Subc : %f\n", ave_diff_subc);
    EXPECT_EQ(ave_diff_perc <= 5e-1, true);
    EXPECT_EQ(ave_diff_subc <= 5e-1, true);
  }

  // Test DQ + cuBLAS GEMM
  float ave_diff_perc = 0.0f, ave_diff_subc = 0.0f;
  for (int i = 0; i < TestCases; ++i) {
    int M = ((rand() % (M_Range[1] - M_Range[0] + 1)) + M_Range[0]);
    const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
    const int K =
        (((rand() % (K_Range[1] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
        Mod * Mod;
    const int GroupSize =
        (((rand() % (GS_Range[3] - GS_Range[0] + 1)) + GS_Range[0]) + Mod - 1) /
        Mod * Mod;

    ave_diff_perc += TestGemmA16W4_New<half>(BS, M, N, K, -1, 1.0f, 5e-2);
    ave_diff_subc +=
        TestGemmA16W4_New<half>(BS, M, N, K, GroupSize, 1.0f, 5e-2);
  }
  ave_diff_perc /= TestCases;
  ave_diff_subc /= TestCases;
  printf("DQ + cuBLAS GEMM Ave_Diff_Perc : %f\n", ave_diff_perc);
  printf("DQ + cuBLAS GEMM Ave_Diff_Subc : %f\n", ave_diff_subc);
  EXPECT_EQ(ave_diff_perc <= 5e-2, true);
  EXPECT_EQ(ave_diff_subc <= 5e-2, true);
}

TEST(GEMM_LOWP, FP16W8_NEW) {
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  const int sm_version = (device_prop.major << 8 | device_prop.minor);

  const int BS = 1;
  const int TestCases = 5;
  const int Mod = 16;
  const int M_Range[4] = {32, 512, 2048, 64};
  const int N_Range[2] = {8, 8192};
  const int K_Range[3] = {8, 2048, 8192};
  const int GS_Range[4] = {64, 128, 256, 512};
  // Test Ampere+ Fused PerC GEMV kernel and SubC GEMV kernel
  if (sm_version >= 0x0800) {
    float ave_diff_perc = 0.0f, ave_diff_perc_bf = 0.0f, ave_diff_subc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      int M = rand() % M_Range[3] + 1;
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[2] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      const int GroupSize = GS_Range[rand() % 4];

      ave_diff_perc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, -1);
      ave_diff_perc_bf +=
          TestGemmA16W8_New<hie::bfloat16, int8_t>(BS, M, N, K, -1);
      ave_diff_subc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, GroupSize);
    }
    ave_diff_perc /= TestCases;
    ave_diff_perc_bf /= TestCases;
    ave_diff_subc /= TestCases;
    printf("Ampere+ Fused GEMV Ave_Diff_Perc : %f\n", ave_diff_perc);
    printf("Ampere+ Fused BF16 GEMV Ave_Diff_Perc : %f\n", ave_diff_perc_bf);
    printf("Ampere+ Fused GEMV Ave_Diff_Subc : %f\n", ave_diff_subc);
    EXPECT_EQ(ave_diff_perc <= 5e-2, true);
    EXPECT_EQ(ave_diff_perc_bf <= 5e-1, true);
    EXPECT_EQ(ave_diff_subc <= 5e-2, true);
  }

  // Test Volta Fused PerC GEMV kernel and Fused SubC GEMV kernel
  if (sm_version == 0x0700) {
    float ave_diff_perc = 0.0f, ave_diff_subc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      const int M = rand() % M_Range[0] + 1;
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[2] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      const int GroupSize = GS_Range[rand() % 4];

      ave_diff_perc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, -1);
      ave_diff_subc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, GroupSize);
    }
    ave_diff_perc /= TestCases;
    ave_diff_subc /= TestCases;
    printf("Volta Fused GEMV Ave_Diff_Perc : %f\n", ave_diff_perc);
    printf("Volta Fused GEMV Ave_Diff_Subc : %f\n", ave_diff_subc);
    EXPECT_EQ(ave_diff_perc <= 5e-1, true);
    EXPECT_EQ(ave_diff_subc <= 5e-1, true);
  }

  // Test Turing+ Fused PerC GEMM kernel and Fused SubC GEMM kernel
  if (sm_version >= 0x0705) {
    float ave_diff_perc = 0.0f, ave_diff_subc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      const int M = (rand() % (M_Range[1] - M_Range[0] + 1)) + M_Range[0];
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[2] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      const int GroupSize = GS_Range[rand() % 4];

      ave_diff_perc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, -1);
      ave_diff_subc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, GroupSize);
    }
    ave_diff_perc /= TestCases;
    ave_diff_subc /= TestCases;
    printf("Turing+ Fused GEMM Ave_Diff_Perc : %f\n", ave_diff_perc);
    printf("Turing+ Fused GEMM Ave_Diff_Subc : %f\n", ave_diff_subc);
    EXPECT_EQ(ave_diff_perc <= 5e-2, true);
    EXPECT_EQ(ave_diff_subc <= 5e-2, true);
  }

  // Test Volta Fused SubC GEMM kernel
  if (sm_version == 0x0700) {
    float ave_diff_subc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      const int M = (rand() % (M_Range[2] - M_Range[0] + 1)) + M_Range[0];
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[2] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      const int GroupSize = GS_Range[rand() % 4];

      ave_diff_subc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, GroupSize);
    }
    ave_diff_subc /= TestCases;
    printf("Volta Fused GEMM Ave_Diff_Subc : %f\n", ave_diff_subc);
    EXPECT_EQ(ave_diff_subc <= 5e-1, true);
  }

  // Test DQ + cuBLAS
  {
    float ave_diff_perc = 0.0f;
    for (int i = 0; i < TestCases; ++i) {
      const int M = (rand() % (M_Range[2] - M_Range[1] + 1)) + M_Range[1];
      const int N = ((rand() % (N_Range[1] - N_Range[0] + 1)) + N_Range[0]);
      const int K =
          (((rand() % (K_Range[1] - K_Range[0] + 1)) + K_Range[0]) + Mod - 1) /
          Mod * Mod;
      ave_diff_perc += TestGemmA16W8_New<half, int8_t>(BS, M, N, K, -1);
    }
    ave_diff_perc /= TestCases;
    printf("DQ + cuBLAS GEMM Ave_Diff_Perc : %f\n", ave_diff_perc);
    EXPECT_EQ(ave_diff_perc <= 5e-2, true);
  }
}

TEST(GEMM_LOWP, FP16W4) {
  // PerC
  TestGemmA16W4<half>(1, 1, 5120, 8192);
  TestGemmA16W4<half>(1, 128, 5120, 8192);
  TestGemmA16W4<half>(1, 1, 5120, 8192);
  TestGemmA16W4<half>(1, 3, 8192, 5120);
}

TEST(GEMM_LOWP, FP16W4_Odd_Shape) {
  // Test unfriendly shapes
  TestGemmA16W4<half>(1, 17, 2560, 8192);
  TestGemmA16W4<half>(1, 31, 5125, 8192);
  TestGemmA16W4<half>(1, 1, 5120, 8197);
  TestGemmA16W4<half>(1, 128, 5125, 8192);
  TestGemmA16W4<half>(1, 999, 2560, 8197);
  TestGemmA16W4<half>(1, 3, 8197, 5125);
}

TEST(GEMM_LOWP, FP16W8) {
  // SubC
  TestGemmA16W8<half, int8_t>(1, 1, 5120, 8192, 64);
  TestGemmA16W8<half, int8_t>(1, 128, 5120, 8192, 128);
  TestGemmA16W8<half, int8_t>(1, 1, 5120, 8192, 256);
  TestGemmA16W8<half, int8_t>(1, 3, 8192, 5120, 64);
  TestGemmA16W8<half, int8_t>(1, 3, 8192, 5120, 64, 2.2f);

  // TestGemmA16W8<half, uint8_t>(1, 1, 5120, 8192, 64);
  // TestGemmA16W8<half, uint8_t>(1, 128, 5120, 8192, 128);
  // TestGemmA16W8<half, uint8_t>(1, 1, 5120, 8192, 256);
  // TestGemmA16W8<half, uint8_t>(1, 3, 8192, 5120, 64);
  // TestGemmA16W8<half, uint8_t>(1, 3, 8192, 5120, 64, 2.2f);
}

TEST(GEMM_LOWP, FP16W8_Odd_Shape) {
  // Test unfriendly shapes
  TestGemmA16W8<half, int8_t>(1, 17, 2560, 8192, 64);
  TestGemmA16W8<half, int8_t>(1, 31, 5125, 8192, 128);
  TestGemmA16W8<half, int8_t>(1, 1, 5120, 8197, 128);
  TestGemmA16W8<half, int8_t>(1, 128, 5125, 8192, 128);
  TestGemmA16W8<half, int8_t>(1, 999, 2560, 5127, 256);
  TestGemmA16W8<half, int8_t>(1, 3, 8197, 5125, 64);
  TestGemmA16W8<half, int8_t>(1, 3, 8197, 5125, 64, 2.0f);

  // TestGemmA16W8<half, uint8_t>(1, 17, 2560, 8192, 64);
  // TestGemmA16W8<half, uint8_t>(1, 31, 5125, 8192, 128);
  // TestGemmA16W8<half, uint8_t>(1, 1, 5120, 8197, 128);
  // TestGemmA16W8<half, uint8_t>(1, 128, 5125, 8192, 128);
  // TestGemmA16W8<half, uint8_t>(1, 999, 2560, 8197, 256);
  // TestGemmA16W8<half, uint8_t>(1, 3, 8197, 5125, 64);
  // TestGemmA16W8<half, uint8_t>(1, 3, 8197, 5125, 64, 1.2f);
}

TEST(GEMM_LOWP, BF16W8) {
  const float EPS = 5e-1;
  // SubC
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 1, 5120, 8192, 64, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 128, 5120, 8192, 128, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 1, 5120, 8192, 256, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 3, 8192, 5120, 64, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 3, 8192, 5120, 64, 2.3f, EPS);

  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 1, 5120, 8192, 64, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 128, 5120, 8192, 128, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 1, 5120, 8192, 256, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 3, 8192, 5120, 64, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 3, 8192, 5120, 64, 3.4f, EPS);
}

TEST(GEMM_LOWP, BF16W8_Odd_Shape) {
  // Test unfriendly shapes
  const float EPS = 5e-1;
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 17, 5120, 8192, 64, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 128, 5125, 8192, 128, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 1, 5125, 8197, 256, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 3, 8197, 5125, 64, EPS);
  TestGemmA16W8<hie::bfloat16, int8_t>(1, 3, 8197, 5125, 64, 4.2f, EPS);

  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 17, 5120, 8192, 64, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 128, 5125, 8192, 128, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 1, 5125, 8197, 256, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 3, 8197, 5125, 64, EPS);
  // TestGemmA16W8<hie::bfloat16, uint8_t>(1, 3, 8197, 5125, 64, 4.2f, EPS);
}

TEST(GEMM_LOWP, A8W8) {
  // PerC
  TestGemmA8W8<half, int8_t>(1, 1, 5120, 8192, -1);
  TestGemmA8W8<half, int8_t>(1, 128, 5120, 8192, -1);
  TestGemmA8W8<half, int8_t>(1, 1, 5120, 8192, -1);
  TestGemmA8W8<half, int8_t>(1, 3, 8192, 5120, -1);
}

#ifdef ENABLE_CUSPARSELT
TEST(GEMM_LOWP, SparseA8W8) {
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  const int sm_version = (device_prop.major << 8 | device_prop.minor);
  if (sm_version < allspark::CUDASMDef::SM_80 ||
      sm_version >= allspark::CUDASMDef::SM_90) {
    GTEST_SKIP();
  }
  // PerC
  TestGemmSparseA8W8<half, int8_t>(1, 16, 32, 32, -1);  // M, N, K
  TestGemmSparseA8W8<half, int8_t>(1, 14, 64, 128, -1);
  TestGemmSparseA8W8<half, int8_t>(1, 128, 128, 64, -1);
  TestGemmSparseA8W8<half, int8_t>(1, 30, 512, 128, -1);
  TestGemmSparseA8W8<half, int8_t>(1, 16, 32, 32, -1, false);
  TestGemmSparseA8W8<half, int8_t>(1, 3, 128, 128, -1, false, 1, 0.06);
  // TestGemmSparseA8W8<half, int8_t>(1, 2046, 14784, 8192, -1);
}
#endif

}  // namespace AS_UTEST
