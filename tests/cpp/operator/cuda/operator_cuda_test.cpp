/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator_cuda_test.cpp
 */

#include <core/tensor/tensor.h>
#include <device_context.h>
#include <test_common.h>

class AsOperatorCUDA : public ::testing::Test {
 protected:
  void SetUp() override {
    device_context = allspark::DeviceContextFactory::CreateCUDAContext();
    device_context->SetDeviceId(0);
  }

  // void TearDown() override {}

  std::shared_ptr<allspark::DeviceContext> device_context;
};
#if 0
TEST_F(AsOperatorCUDA, Embedding) {
    auto device_context = allspark::DeviceContextFactory::CreateCUDAContext();
    allspark::AsTensor in0("input_ids", allspark::CUDA, allspark::INT64,
                           allspark::DataMode::DENSE, allspark::Shape({2, 8}));
    allspark::AsTensor in1("token_type_ids", allspark::CUDA, allspark::INT64,
                           allspark::DataMode::DENSE, allspark::Shape({2, 8}));
    std::vector<int64_t> in0_data(2 * 8);
    std::vector<int64_t> in1_data(2 * 8);
    std::vector<float> out_data(2 * 8 * 768);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "embedding/");
    AS_CHECK(LoadBinFromFile(opdir + "input_ids.bin", in0_data.data(),
                             2 * 8 * sizeof(int64_t)));
    AS_CHECK(LoadBinFromFile(opdir + "token_type_ids.bin", in1_data.data(),
                             2 * 8 * sizeof(int64_t)));
    AS_CHECK(LoadBinFromFile(opdir + "out.bin", out_data.data(),
                             2 * 8 * 768 * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), 2*8*sizeof(int64_t), allspark::CPU);
    in1.CopyDataFrom(in1_data.data(), 2*8*sizeof(int64_t), allspark::CPU);
    allspark::DLTensorMap outputs{{"hidden_states", nullptr}};
    test_model("Embedding", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input_ids", in0.ToDLPack(device_context.get())},
                {"token_type_ids", in1.ToDLPack(device_context.get())}},
               &outputs);
    allspark::AsTensor out("hidden_states", outputs["hidden_states"]);
    std::vector<float> compute_out_data(2 * 8 * 768);
    out.CopyDataTo(compute_out_data.data(), sizeof(float) * 2 * 8 * 768, allspark::CPU);
    float eps = check_equal<float>(out_data.data(), compute_out_data.data(),
                                   2 * 8 * 768);
    EXPECT_EQ(eps < OP_EPS, true);
}
TEST_F(AsOperatorCUDA, LayerNorm) {
    allspark::AsTensor in0("input", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 768}));
    int in0_shape = 2 * 8 * 768;
    int out_shape = 2 * 8 * 768;

    auto device_context = allspark::DeviceContextFactory::CreateCUDAContext();
    std::vector<float> in0_data(in0_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "layernorm/");
    AS_CHECK(LoadBinFromFile(opdir + "layernorm_in.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "layernorm_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), in0_shape * sizeof(float), allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("LayerNorm", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input", in0.ToDLPack(device_context.get())}}, &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), sizeof(float) * out_shape, allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}
TEST_F(AsOperatorCUDA, Gemm_pooler) {
    allspark::AsTensor in0("input", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 768}));
    int in0_shape = 2 * 8 * 768;
    int out_shape = 2 * 768;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "gemm/pooler/");
    AS_CHECK(LoadBinFromFile(opdir + "pooler_in.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "pooler_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), sizeof(float) * in0_shape, allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("Gemm_pooler", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input", in0.ToDLPack(device_context.get())}}, &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), out_shape * sizeof(float), allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}
TEST_F(AsOperatorCUDA, Gemm_sgemm) {
    allspark::AsTensor in0("input0", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 768}));
    int in0_shape = 2 * 8 * 768;
    int out_shape = 2 * 8 * 768;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "gemm/sgemm/");
    AS_CHECK(LoadBinFromFile(opdir + "gemm_input0.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "gemm_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), in0_shape * sizeof(float), allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("Gemm_sgemm", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input0", in0.ToDLPack(device_context.get())}}, &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), sizeof(float) * out_shape, allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}
TEST_F(AsOperatorCUDA, Gemm_batch) {
    allspark::AsTensor in0("input0", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 768}));
    int in0_shape = 2 * 8 * 768;
    int out_shape = 3 * 2 * 8 * 768;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "gemm/batch_gemm/");
    AS_CHECK(LoadBinFromFile(opdir + "batch_gemm_input0.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "batch_gemm_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), sizeof(float) * in0_shape, allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("Gemm_batch", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input0", in0.ToDLPack(device_context.get())}}, &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), sizeof(float) * out_shape, allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}

TEST_F(AsOperatorCUDA, MHA) {
    allspark::AsTensor in0("input", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 2304}));
    allspark::AsTensor in1("attention_mask", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 8}));
    int in0_shape = 2 * 8 * 2304;
    int in1_shape = 2 * 8 * 8;
    int out_shape = 2 * 8 * 768;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> in1_data(in1_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "mha/");
    AS_CHECK(LoadBinFromFile(opdir + "mha_input0.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "mha_input1.bin", in1_data.data(),
                             in1_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "mha_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), sizeof(float) * in0_shape, allspark::CPU);
    in1.CopyDataFrom(in1_data.data(), sizeof(float) * in1_shape, allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("MultiHeadAttention", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input", in0.ToDLPack(device_context.get())},
                {"attention_mask", in1.ToDLPack(device_context.get())}},
               &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), sizeof(float) * out_shape, allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}

TEST_F(AsOperatorCUDA, Binary_Add) {
    allspark::AsTensor in0("input0", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 768}));
    allspark::AsTensor in1("input1", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 768}));
    int in0_shape = 2 * 8 * 768;
    int in1_shape = 2 * 8 * 768;
    int out_shape = 2 * 8 * 768;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> in1_data(in1_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "binary_add/");
    AS_CHECK(LoadBinFromFile(opdir + "binary_add_input0.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "binary_add_input1.bin", in1_data.data(),
                             in1_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "binary_add_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), in0_shape * sizeof(float), allspark::CPU);
    in1.CopyDataFrom(in1_data.data(), in1_shape * sizeof(float), allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("Binary_Add", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input0", in0.ToDLPack(device_context.get())},
                {"input1", in1.ToDLPack(device_context.get())}},
               &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), out_shape * sizeof(float), allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}

TEST_F(AsOperatorCUDA, Unary_GeluErf) {
    allspark::AsTensor in0("input", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 8, 3072}));
    int in0_shape = 2 * 8 * 3072;
    int out_shape = 2 * 8 * 3072;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "unary/gelu_erf/");
    AS_CHECK(LoadBinFromFile(opdir + "unary_geluerf_input0.bin",
                             in0_data.data(), in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "unary_geluerf_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), sizeof(float) * in0_shape, allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("Unary_GeluErf", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input0", in0.ToDLPack(device_context.get())}}, &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), sizeof(float) * out_shape, allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}

TEST_F(AsOperatorCUDA, Unary_Tanh) {
    allspark::AsTensor in0("input", allspark::CUDA, allspark::FLOAT32,
                           allspark::DataMode::DENSE,
                           allspark::Shape({2, 768}));
    int in0_shape = 2 * 768;
    int out_shape = 2 * 768;
    std::vector<float> in0_data(in0_shape);
    std::vector<float> out_data(out_shape);
    std::string opdir =
        std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                    "testcase/operator/" + "unary/tanh/");
    AS_CHECK(LoadBinFromFile(opdir + "unary_tanh_input0.bin", in0_data.data(),
                             in0_shape * sizeof(float)));
    AS_CHECK(LoadBinFromFile(opdir + "unary_tanh_out.bin", out_data.data(),
                             out_shape * sizeof(float)));
    in0.CopyDataFrom(in0_data.data(), sizeof(float) * in0_shape, allspark::CPU);
    allspark::DLTensorMap outputs{{"out", nullptr}};
    test_model("Unary_Tanh", "CUDA",
               std::string(std::string(getenv("ALLSPARK_TESTCASE_PATH")) +
                           "testcase/operator/config.yml"),
               {{"input0", in0.ToDLPack(device_context.get())}}, &outputs);
    allspark::AsTensor out("out", outputs["out"]);
    std::vector<float> compute_out_data(out_shape);
    out.CopyDataTo(compute_out_data.data(), out_shape * sizeof(float) , allspark::CPU);
    float eps =
        check_equal<float>(out_data.data(), compute_out_data.data(), out_shape);
    EXPECT_EQ(eps < OP_EPS, true);
}
#endif