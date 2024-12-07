/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_logsoftmax_test.cpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
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
std::vector<HT> LogSoftmaxRef(const std::vector<HT>& inVec, const int stride,
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
      float var = static_cast<HT>(inVec[i * stride + j] / temperature[i] -
                                  hMax[i] / temperature[i]) -
                  logf(hSum[i]);
      hRef[i * stride + j] = static_cast<HT>(static_cast<T>(var));
    }
  }

  return hRef;
}

// ---------------------------------
// Test
// ---------------------------------
class LogSoftmaxTest : public ::testing::Test {
 public:
  template <typename T>
  void test_log_softmax(int batch, int length, float temperature,
                        float feps = 1e-2) {
    // tensor map
    allspark::TensorMap tensors;

    std::string in0_name = "logits";
    std::vector<allspark::dim_t> shape_in0 = {batch, length};
    common::AddTensor(tensors, in0_name, common::toDataType<T>::dt);

    std::string in2_name = "temperatures";
    std::vector<allspark::dim_t> shape_in2 = {batch};
    common::AddTensor(tensors, in2_name, asFP32);

    std::string in1_name = "input_lengths";
    std::vector<allspark::dim_t> shape_in1 = {batch};
    common::AddTensor(tensors, in1_name, asINT32);

    // workspace
    std::string ws_name = "workspace";
    size_t ws_bytes(0);
    allspark::cuda::StridedSoftmaxGetWorkspaceSize<T>(&ws_bytes, batch, length);
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
              tensors.at(ws_name)->SetShape(Shape(shape_ws)));

    device->Synchronize();

    // set data
    std::vector<T> in0_data(batch * length);
    for (int i = 0; i < batch; ++i) {
      const int pivot = length * i / batch;
      init_uniform_float(in0_data.data() + i * length, length, pivot);
    }

    std::stringstream ss;
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

    const auto& cuda_stream =
        dynamic_cast<const CUDAContext*>(device.get())->GetStream();

    ASSERT_EQ(true,
              common::AsyncH2D(in0_data.data(), tensors.at(in0_name).get(),
                               in0_data.size() * sizeof(T), cuda_stream));
    ASSERT_EQ(true,
              common::AsyncH2D(input_len.data(), tensors.at(in1_name).get(),
                               input_len.size() * sizeof(float), cuda_stream));
    ASSERT_EQ(true, common::AsyncH2D(
                        temperatures.data(), tensors.at(in2_name).get(),
                        temperatures.size() * sizeof(float), cuda_stream));

    // run inplace
    allspark::cuda::StridedLogSoftmaxLauncher<T>(
        static_cast<T*>(tensors.at(in0_name)->GetDataPtr()),
        static_cast<T*>(tensors.at(in0_name)->GetDataPtr()),
        static_cast<int*>(tensors.at(in1_name)->GetDataPtr()),
        static_cast<float*>(tensors.at(in2_name)->GetDataPtr()),
        tensors.at(ws_name)->GetDataPtr(),
        tensors.at(ws_name)->GetShape().Count(), batch, length, cuda_stream);

    std::vector<T> h_out(batch * length);
    ASSERT_EQ(true, common::AsyncD2H(tensors.at(in0_name).get(), h_out.data(),
                                     h_out.size() * sizeof(T), cuda_stream));
    device->Synchronize();

    // check
    std::vector<T> ref_out =
        LogSoftmaxRef(in0_data, length, temperatures.data(), input_len.data());

    for (int i = 0; i < batch; ++i) {
      float diff = check_equal<T>(ref_out.data() + i * length,
                                  h_out.data() + i * length, input_len[i]);
      EXPECT_LT(diff, feps);
    }

    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in0_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in1_name)->Free());
    ASSERT_EQ(AsStatus::ALLSPARK_SUCCESS, tensors.at(in2_name)->Free());
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

// only run (log) softmax with FP32
TEST_F(LogSoftmaxTest, unsorted_b4l50000p10t10fp32) {
  test_log_softmax<float>(4, 50000, 1.0);
}

TEST_F(LogSoftmaxTest, unsorted_b4l50000p00t10fp32) {
  test_log_softmax<float>(4, 50000, 1.0);
}

TEST_F(LogSoftmaxTest, unsorted_b4l50000p8t10fp32) {
  test_log_softmax<float>(4, 50000, 1.0);
}

TEST_F(LogSoftmaxTest, unsorted_b4l250000p8t10fp32) {
  test_log_softmax<float>(4, 250000, 1.0, 5e-2);
}

TEST_F(LogSoftmaxTest, sorted_b4l250000p8t10fp32) {
  test_log_softmax<float>(4, 250000, 1.0, 5e-2);
}

TEST_F(LogSoftmaxTest, unsorted_b4l250000p8t05fp32) {
  test_log_softmax<float>(4, 250000, 0.5, 5e-2);
}

TEST_F(LogSoftmaxTest, sorted_b32l250000p8t05fp32) {
  test_log_softmax<float>(32, 250000, 0.5, 5e-1);
}

TEST_F(LogSoftmaxTest, unsorted_b4l250000p5t12fp32) {
  test_log_softmax<float>(4, 250000, 1.2, 5e-2);
}
