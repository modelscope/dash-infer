/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_topp_test.cpp
 */

#include <algorithm>
#include <chrono>
#include <sstream>

#include "common.hpp"
#include "test_common.h"

namespace {

void reset_sstream(std::stringstream& ss) {
  std::stringstream tmp;
  ss.swap(tmp);
  return;
}

// ---------------------------------
// Rand
// ---------------------------------
template <typename T>
void init_uniform_float(T* begin, int n, int pivot) {
  std::uniform_int_distribution<int> dis1(0, pivot);
  std::uniform_int_distribution<int> dis2(pivot, n);
  std::default_random_engine generator;
  // generator.seed(1);

  using myclock = std::chrono::high_resolution_clock;
  myclock::time_point beginning = myclock::now();
  myclock::duration d = myclock::now() - beginning;
  unsigned seed = d.count();
  generator.seed(seed);

  std::generate(begin, begin + pivot, [&generator, &dis = dis1, n]() {
    return static_cast<T>(static_cast<float>(dis(generator)));
  });
  std::generate(begin + pivot, begin + n, [&generator, &dis = dis2, n]() {
    return static_cast<T>(static_cast<float>(dis(generator)));
  });
  return;
}

// ---------------------------------
// Ref
// ---------------------------------
template <typename HT, typename T = HT>
std::vector<HT> SoftmaxRef(const std::vector<HT>& inVec, const int stride,
                           const float* temperature, const int* taskLenPtr) {
  const int batch = inVec.size() / stride;
  std::vector<int> taskLen(batch);
  if (taskLenPtr) {
    for (int i = 0; i < batch; ++i) {
      taskLen[i] = taskLenPtr[i];
    }
  } else {
    for (int i = 0; i < batch; ++i) {
      taskLen[i] = stride;
    }
  }

  std::vector<HT> hMax(batch);
  for (int i = 0; i < batch; ++i) {
    hMax[i] = *std::max_element(inVec.begin() + i * stride,
                                inVec.begin() + i * stride + taskLen[i]);
  }

  std::vector<double> hExp(stride * batch);
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < taskLen[i]; ++j) {
      hExp[i * stride + j] = exp(inVec[i * stride + j] / temperature[i] -
                                 hMax[i] / temperature[i]);
    }
  }

  std::vector<HT> hSum(batch);
  for (int i = 0; i < batch; ++i) {
    hSum[i] = static_cast<HT>(std::accumulate(
        hExp.begin() + i * stride, hExp.begin() + i * stride + taskLen[i], 0.));
  }

  std::vector<HT> hRef(stride * batch);
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < taskLen[i]; ++j) {
      float var = static_cast<HT>(hExp[i * stride + j]) / hSum[i];
      hRef[i * stride + j] = static_cast<HT>(static_cast<T>(var));
    }
  }

  return hRef;
}

template <typename T>
void TopPSoftmaxRef(std::vector<int>& topp_count,
                    std::vector<int>& topp_indices, std::vector<T>& topp_probs,
                    const std::vector<T>& input_logits,
                    const std::vector<float>& p_values,
                    const std::vector<int>& input_len,
                    const std::vector<float> temperature, const bool sorted) {
  const int batch = topp_count.size();
  const int length = topp_indices.size() / batch;

  std::vector<float> comp_logits(batch * length);
  std::transform(input_logits.cbegin(), input_logits.cend(),
                 comp_logits.begin(),
                 [](const T& x) { return static_cast<float>(x); });

  if (!sorted) {
    // sort logits as well as indices
    for (int i = 0; i < batch; ++i) {
      std::iota(topp_indices.begin() + i * length,
                topp_indices.begin() + i * length + length, 0);
      std::stable_sort(
          topp_indices.begin() + i * length,
          topp_indices.begin() + i * length + length,
          [&comp_logits, i, length](const int64_t j1, const int64_t j2) {
            return comp_logits[i * length + j1] > comp_logits[i * length + j2];
          });
      std::stable_sort(comp_logits.begin() + i * length,
                       comp_logits.begin() + i * length + length,
                       std::greater<float>());
    }
  }

  // cut off top-p count
  const std::vector<float> all_probs = SoftmaxRef<float, T>(
      comp_logits, length, temperature.data(), input_len.data());

  std::vector<float> prefix(batch * length);
  for (int i = 0; i < batch; ++i) {
    std::vector<float> uncasted(length);
    std::partial_sum(all_probs.cbegin() + i * length,
                     all_probs.cbegin() + i * length + input_len[i],
                     uncasted.begin());
    std::transform(uncasted.cbegin(), uncasted.cbegin() + input_len[i],
                   prefix.begin() + i * length, [](const float& x) {
                     return static_cast<float>(static_cast<T>(x));
                   });
  }

  for (int i = 0; i < batch; ++i) {
    float p_casted = static_cast<float>(static_cast<T>(p_values[i]));
    if (p_casted > 0.f && p_casted < 1.f) {
      auto lbIt = std::lower_bound(prefix.cbegin() + i * length,
                                   prefix.cbegin() + i * length + input_len[i],
                                   p_casted);
      topp_count[i] = lbIt - (prefix.begin() + i * length) + 1;
      if (*lbIt == p_casted) {
        topp_count[i] += 1;
      }
    } else {
      topp_count[i] = input_len[i];
    }
  }

  // compute final prob
  std::vector<float> output_probs = SoftmaxRef<float>(
      comp_logits, length, temperature.data(), topp_count.data());
  std::transform(output_probs.cbegin(), output_probs.cend(), topp_probs.begin(),
                 [](const float& x) { return static_cast<T>(x); });
  return;
}

// ---------------------------------
// Test
// ---------------------------------
class TopPTest : public ::testing::Test {
 public:
  template <typename T>
  void test_topp_softmax(int batch, int length, float p_value,
                         float temperature, bool sorted = false,
                         float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tensors;

    std::string in0_name = "input_logits";
    std::vector<allspark::dim_t> shape_in0 = {batch, length};
    common::AddTensor(tensors, in0_name, common::toDataType<T>::dt);

    std::string in1_name = "p_values";
    std::vector<allspark::dim_t> shape_in1 = {batch};
    common::AddTensor(tensors, in1_name, asFP32);

    std::string in2_name = "temperatures";
    std::vector<allspark::dim_t> shape_in2 = {batch};
    common::AddTensor(tensors, in2_name, asFP32);

    std::string in3_name = "temp_probs";
    std::vector<allspark::dim_t> shape_in3 = {batch, length};
    common::AddTensor(tensors, in3_name, common::toDataType<T>::dt);

    std::string out0_name = "topp_count";
    std::vector<allspark::dim_t> shape_out0 = {batch};
    common::AddTensor(tensors, out0_name, asINT32);

    std::string out1_name = "topp_probs";
    std::vector<allspark::dim_t> shape_out1 = {batch, length};
    common::AddTensor(tensors, out1_name, common::toDataType<T>::dt);

    std::string out2_name = "topp_indices";
    std::vector<allspark::dim_t> shape_out2 = {batch, length};
    common::AddTensor(tensors, out2_name, asINT32);

    // workspace
    std::string ws_name = "workspace";
    size_t ws_bytes(0);
    allspark::cuda::TopPSoftmaxGetWorkspaceSize<T>(&ws_bytes, batch, length,
                                                   sorted);
    std::vector<allspark::dim_t> shape_ws = {dim_t(ws_bytes)};
    common::AddTensor(tensors, ws_name, asINT8);

    // reshape
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(in0_name)->SetShape(Shape(shape_in0)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(in1_name)->SetShape(Shape(shape_in1)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(in2_name)->SetShape(Shape(shape_in2)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(in3_name)->SetShape(Shape(shape_in3)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(out0_name)->SetShape(Shape(shape_out0)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(out1_name)->SetShape(Shape(shape_out1)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(out2_name)->SetShape(Shape(shape_out2)));
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS,
              tensors.at(ws_name)->SetShape(Shape(shape_ws)));

    device->Synchronize();

    // set data
    std::vector<T> in0_data(batch * length);
    for (int i = 0; i < batch; ++i) {
      const int pivot = length * i / batch;
      init_uniform_float(in0_data.data() + i * length, length, pivot);
    }

    std::stringstream ss;
    ss << "p_value:\t";
    std::vector<float> in1_data(batch);
    std::generate(in1_data.begin(), in1_data.end(), [p_value, &ss]() {
      int factor = rand() % 3 - 1;
      auto v = std::max(std::min(p_value + 0.05f * factor, 1.f), 0.f);
      ss << v << "\t";
      return v;
    });
    std::cout << ss.str() << std::endl;

    reset_sstream(ss);
    ss << "temperature:\t";
    std::vector<float> temperatures(batch);
    std::generate(temperatures.begin(), temperatures.end(),
                  [temperature, &ss]() {
                    int factor = rand() % 3;
                    auto v = std::min(temperature + 0.1f * factor, 1.f);
                    ss << v << "\t";
                    return v;
                  });
    std::cout << ss.str() << std::endl;

    reset_sstream(ss);
    ss << "input_len:\t";
    std::vector<int> input_len(batch);
    std::generate(input_len.begin(), input_len.end(), [length, &ss]() {
      auto v = std::max(rand() % length, 1);
      ss << v << "\t";
      return v;
    });
    std::cout << ss.str() << std::endl;

    std::vector<int> ref_topp_indices(batch * length);

    const auto& cuda_stream =
        dynamic_cast<const CUDAContext*>(device.get())->GetStream();

    if (sorted) {
      std::vector<float> comp_logits(batch * length);
      std::transform(in0_data.cbegin(), in0_data.cend(), comp_logits.begin(),
                     [](const T& x) { return static_cast<float>(x); });

      for (int i = 0; i < batch; ++i) {
        std::iota(ref_topp_indices.begin() + i * length,
                  ref_topp_indices.begin() + (i + 1) * length, 0);
        std::stable_sort(ref_topp_indices.begin() + i * length,
                         ref_topp_indices.begin() + (i + 1) * length,
                         [&comp_logits, i, length](const int j1, const int j2) {
                           return comp_logits[i * length + j1] >
                                  comp_logits[i * length + j2];
                         });
        std::stable_sort(comp_logits.begin() + i * length,
                         comp_logits.begin() + (i + 1) * length,
                         std::greater<float>());
      }
      std::transform(comp_logits.cbegin(), comp_logits.cend(), in0_data.begin(),
                     [](const float& x) { return static_cast<T>(x); });

      ASSERT_EQ(true, common::AsyncH2D(
                          ref_topp_indices.data(), tensors.at(out2_name).get(),
                          ref_topp_indices.size() * sizeof(int), cuda_stream));
    }

    ASSERT_EQ(true,
              common::AsyncH2D(in0_data.data(), tensors.at(in0_name).get(),
                               in0_data.size() * sizeof(T), cuda_stream));
    ASSERT_EQ(true,
              common::AsyncH2D(in1_data.data(), tensors.at(in1_name).get(),
                               in1_data.size() * sizeof(float), cuda_stream));
    ASSERT_EQ(true, common::AsyncH2D(
                        temperatures.data(), tensors.at(in2_name).get(),
                        temperatures.size() * sizeof(float), cuda_stream));
    ASSERT_EQ(true,
              common::AsyncH2D(input_len.data(), tensors.at(out0_name).get(),
                               input_len.size() * sizeof(int), cuda_stream));
    // run
    allspark::cuda::TopPSoftmaxLauncher<T>(
        static_cast<int*>(tensors.at(out0_name)->GetDataPtr()),
        static_cast<T*>(tensors.at(out1_name)->GetDataPtr()),
        static_cast<int*>(tensors.at(out2_name)->GetDataPtr()),
        static_cast<T*>(tensors.at(in0_name)->GetDataPtr()),
        static_cast<float*>(tensors.at(in1_name)->GetDataPtr()),
        static_cast<float*>(tensors.at(in2_name)->GetDataPtr()),
        static_cast<T*>(tensors.at(in3_name)->GetDataPtr()),
        tensors.at(ws_name)->GetDataPtr(),
        tensors.at(ws_name)->GetShape().Count(), batch, length, sorted,
        dynamic_cast<const CUDAContext*>(device.get())->GetHiednnHandle(),
        cuda_stream);

    std::vector<int> h_out0(batch);
    std::vector<T> h_out1(batch * length);
    std::vector<int> h_out2(batch * length);
    ASSERT_EQ(true, common::AsyncD2H(tensors.at(out0_name).get(), h_out0.data(),
                                     h_out0.size() * sizeof(int), cuda_stream));
    ASSERT_EQ(true, common::AsyncD2H(tensors.at(out1_name).get(), h_out1.data(),
                                     h_out1.size() * sizeof(T), cuda_stream));
    ASSERT_EQ(true, common::AsyncD2H(tensors.at(out2_name).get(), h_out2.data(),
                                     h_out2.size() * sizeof(int), cuda_stream));

    device->Synchronize();

    // check
    std::vector<int> ref_topp_count(batch);
    std::vector<T> ref_topp_probs(batch * length);
    TopPSoftmaxRef(ref_topp_count, ref_topp_indices, ref_topp_probs, in0_data,
                   in1_data, input_len, temperatures, sorted);

    for (int i = 0; i < batch; ++i) {
      EXPECT_LE(std::abs(ref_topp_count[i] - h_out0[i]),
                length > 200000 ? (length - 200000) / 10000 + 3 : 3);

      const int topp_count = std::min(ref_topp_count[i], h_out0[i]);

      for (int j = 0; j < topp_count; ++j) {
        EXPECT_EQ(ref_topp_indices[i * length + j], h_out2[i * length + j]);
      }

      float diff = check_equal<T>(ref_topp_probs.data() + i * length,
                                  h_out1.data() + i * length, topp_count);
      // printf("[DIFF-ENCODER] max diff = %f\n", diff);
      EXPECT_LT(diff, feps);
    }

    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in0_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in1_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in2_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in3_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(out0_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(out1_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(out2_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(ws_name)->Free());

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
  // context
  std::shared_ptr<allspark::DeviceContext> device;
};

}  // namespace

TEST_F(TopPTest, unsorted_b4l50000p10t10fp32) {
  test_topp_softmax<float>(4, 50000, 1.0, 1.0);
}

TEST_F(TopPTest, unsorted_b4l50000p00t10fp32) {
  test_topp_softmax<float>(4, 50000, 0.0, 1.0);
}

TEST_F(TopPTest, unsorted_b4l50000p8t10fp32) {
  test_topp_softmax<float>(4, 50000, 0.8, 1.0);
}

TEST_F(TopPTest, unsorted_b4l250000p8t10fp32) {
  test_topp_softmax<float>(4, 250000, 0.8, 1.0, false, 5e-2);
}

TEST_F(TopPTest, sorted_b4l250000p8t10fp32) {
  test_topp_softmax<float>(4, 250000, 0.8, 1.0, true, 5e-2);
}

TEST_F(TopPTest, unsorted_b4l250000p8t05fp32) {
  test_topp_softmax<float>(4, 250000, 0.8, 0.5, false, 5e-2);
}

TEST_F(TopPTest, sorted_b32l250000p8t05fp32) {
  test_topp_softmax<float>(32, 250000, 0.8, 0.5, true, 5e-1);
}

TEST_F(TopPTest, unsorted_b4l250000p5t12fp32) {
  test_topp_softmax<float>(4, 250000, 0.5, 1.2, false, 5e-2);
}

#ifdef ENABLE_FP16
TEST_F(TopPTest, unsorted_b4l50000p8t10fp16) {
  test_topp_softmax<half>(4, 50000, 0.8, 1.0);
}

TEST_F(TopPTest, unsorted_b4l250000p5t05fp16) {
  test_topp_softmax<half>(4, 250000, 0.5, 0.5, false, 5e-2);
}

TEST_F(TopPTest, sorted_b4l250000p5t05fp16) {
  test_topp_softmax<half>(4, 250000, 0.5, 0.5, true, 5e-2);
}
#endif

#ifdef ENABLE_BF16
TEST_F(TopPTest, unsorted_b4l50000p8t10bf16) {
  test_topp_softmax<hie::bfloat16>(4, 50000, 0.8, 1.0);
}

TEST_F(TopPTest, unsorted_b4l250000p5t05bf16) {
  test_topp_softmax<hie::bfloat16>(4, 250000, 0.5, 0.5, false, 5e-2);
}

TEST_F(TopPTest, sorted_b4l250000p5t05bf16) {
  test_topp_softmax<hie::bfloat16>(4, 250000, 0.5, 0.5, true, 5e-2);
}
#endif
