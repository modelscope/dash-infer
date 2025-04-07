/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator_decmha_test.cpp
 */

#if 0
#include <core/operator/generate_opt/dec_opt_mha/dec_opt_mha_op.h>
#include <core/operator/generate_opt/dec_opt_mha_i8cache/dec_opt_mha_i8cache_op.h>
#include <cuda/cuda_context.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../test_operator_utils.h"
using AS_UTEST::TestOpUtil;

#define asTensor allspark::AsTensor
#define asCUDA allspark::DeviceType::CUDA
#define asCPU allspark::DeviceType::CPU
#define asDense allspark::DataMode::DENSE
#define asInt8 allspark::DataType::INT8
#define asFP32 allspark::DataType::FLOAT32
#define asFP16 allspark::DataType::FLOAT16
#define quantType cuda::mha_quant_cache::QuantType

template <typename T>
struct toDataType {
  constexpr static allspark::DataType dt =
      allspark::DataType::DATATYPE_UNDEFINED;
};
template <>
struct toDataType<float> {
  constexpr static allspark::DataType dt = asFP32;
};
template <>
struct toDataType<half> {
  constexpr static allspark::DataType dt = asFP16;
};

#define DEBUG_TENSOR(MAP, ALIAS)
// #define DEBUG_TENSOR(MAP, ALIAS) \
// printf("[" #MAP "]\t" #ALIAS ": \t%s\n", tensor_map_##MAP.at(ALIAS##_name.c_str())->ToString().c_str());

#define ENABLE_PREV 1
#define ENABLE_I8KV 1

template <typename T>
int64_t squash_vector_shape(std::vector<T> data) {
  int64_t i = 1;
  for (auto d : data) {
    i *= static_cast<int64_t>(d);
  }
  return i;
}

bool AsyncH2D(void* host, asTensor* device, size_t size, cudaStream_t stream) {
  return cudaMemcpyAsync(device->GetDataPtr(), host, size,
                         cudaMemcpyKind::cudaMemcpyHostToDevice,
                         stream) == cudaSuccess;
}

bool AsyncD2H(asTensor* device, void* host, size_t size, cudaStream_t stream) {
  return cudaMemcpyAsync(host, device->GetDataPtr(), size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost,
                         stream) == cudaSuccess;
}

template <typename T>
std::vector<T> rand_float(int count, float scale = 1.f) {
  std::vector<T> rtn(count, 0.);
  for (int i = 0; i < count; i++)
    rtn[i] = static_cast<T>(
        scale * 2.f *
        (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f));
  return rtn;
}

std::vector<float> rand_ninf_mask(int count) {
  auto fp = rand_float<float>(count, 1.f);
  for (int i = 0; i < count; i++) {
    if (fp[i] >= 0.)
      fp[i] = -1e15;
    else
      fp[i] = 0.f;
  }
  return fp;
}

class DecMHATest : public ::testing::Test {
 public:
  template <typename T>
  void test_dec_mha(int batch, int phead, int nhead, int xseql, int cache,
                    int loop_max, float feps) {
    proto.set_op_type("DecOptMHAI8Cache");
    std::unique_ptr<allspark::RuntimeContext> runtime_ctx =
        std::make_unique<RuntimeContext>();
    runtime_ctx->PushBackGenCtx(std::make_shared<GenerateContext>());
    std::shared_ptr<GenerateContext> genctx = runtime_ctx->GetGenCtx(0);
    allspark::DataType dt0 = toDataType<T>::dt;

    // param
    int loop_cnt = 0;

    // context
    genctx->max_length = cache;
    genctx->batch_size = batch;
    genctx->step = 0;

    // tensor map
    allspark::TensorMap weight_map;
    allspark::TensorMap tensor_map_prev;
    allspark::TensorMap tensor_map_curr;

    // tensor input0 qkv
    std::string in0_name = "qkv";
    std::vector<allspark::dim_t> shape_input0_context = {batch, xseql,
                                                         3 * nhead * phead};
    std::vector<T> vc_input0_context =
        rand_float<T>(squash_vector_shape(shape_input0_context));
    allspark::TensorProto& tp_input0 = *proto.add_inputs();
    tp_input0.set_name(in0_name);
#if ENABLE_PREV
    tensor_map_prev.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            in0_name.c_str(),
            std::make_unique<asTensor>(in0_name, asCUDA, dt0, asDense,
                                       allspark::Shape(shape_input0_context))));
    AsyncH2D(vc_input0_context.data(),
             tensor_map_prev.at(in0_name.c_str()).get(),
             vc_input0_context.size() * sizeof(T),
             static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_PREV
#if ENABLE_I8KV
    tensor_map_curr.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            in0_name.c_str(),
            std::make_unique<asTensor>(in0_name, asCUDA, dt0, asDense,
                                       allspark::Shape(shape_input0_context))));
    AsyncH2D(vc_input0_context.data(),
             tensor_map_curr.at(in0_name.c_str()).get(),
             vc_input0_context.size() * sizeof(T),
             static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_I8KV
    device->Synchronize();

    // tensor input1 mask
    std::string in1_name = "mask";
    std::vector<allspark::dim_t> shape_input1_context = {batch * nhead, xseql,
                                                         xseql};
    std::vector<float> vc_input1 =
        rand_ninf_mask(squash_vector_shape(shape_input1_context));
    allspark::TensorProto& tp_input1 = *proto.add_inputs();
    tp_input1.set_name(in1_name);
#if ENABLE_PREV
    tensor_map_prev.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            in1_name.c_str(),
            std::make_unique<asTensor>(in1_name, asCUDA, asFP32, asDense,
                                       allspark::Shape(shape_input1_context))));
    AsyncH2D(vc_input1.data(), tensor_map_prev.at(in1_name.c_str()).get(),
             vc_input1.size() * sizeof(float),
             static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_PREV
#if ENABLE_I8KV
    tensor_map_curr.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            in1_name.c_str(),
            std::make_unique<asTensor>(in1_name, asCUDA, asFP32, asDense,
                                       allspark::Shape(shape_input1_context))));
    AsyncH2D(vc_input1.data(), tensor_map_curr.at(in1_name.c_str()).get(),
             vc_input1.size() * sizeof(float),
             static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_I8KV
    device->Synchronize();

    // tensor input2 position embedding
    std::string in2_name = "position_embedding";
    allspark::TensorProto& tp_input2 = *proto.add_inputs();
    tp_input2.set_name(in2_name);
#if ENABLE_PREV
    tensor_map_prev.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            in2_name.c_str(), std::make_unique<asTensor>(in2_name, asCUDA)));
#endif  // ENABLE_PREV
#if ENABLE_I8KV
    tensor_map_curr.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            in2_name.c_str(), std::make_unique<asTensor>(in2_name, asCUDA)));
#endif  // ENABLE_I8KV

    // workspace
    std::string wss_name = "workspace";
#if ENABLE_PREV
    tensor_map_prev.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            wss_name.c_str(),
            std::make_unique<asTensor>(wss_name, asCUDA, asInt8)));
#endif  // ENABLE_PREV
#if ENABLE_I8KV
    tensor_map_curr.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            wss_name.c_str(),
            std::make_unique<asTensor>(wss_name, asCUDA, asInt8)));
#endif  // ENABLE_I8KV

    // tensor output
    std::string out_name = "output";
    std::vector<allspark::dim_t> shape_output = {batch, xseql, nhead * phead};
    std::vector<T> vc_output_prev(squash_vector_shape(shape_output), 0.);
    std::vector<T> vc_output_curr(squash_vector_shape(shape_output), 0.);
    allspark::TensorProto& tp_output = *proto.add_outputs();
    tp_output.set_name(out_name);
#if ENABLE_PREV
    tensor_map_prev.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            out_name.c_str(),
            std::make_unique<asTensor>(out_name, asCUDA, dt0, asDense,
                                       allspark::Shape(shape_output))));
#endif  // ENABLE_PREV
#if ENABLE_I8KV
    tensor_map_curr.insert(
        std::make_pair<std::string, std::unique_ptr<asTensor>>(
            out_name.c_str(),
            std::make_unique<asTensor>(out_name, asCUDA, dt0, asDense,
                                       allspark::Shape(shape_output))));
#endif  // ENABLE_I8KV

    // attr
    auto& proto_map =
        *proto.mutable_attr();  // PROTOBUF_NAMESPACE_ID::Map<std::string,
                                // std::string >
    proto_map["num_heads"] =
        std::string(reinterpret_cast<char*>(&nhead), sizeof(int));

    // first step
    device->Synchronize();
#if ENABLE_PREV
    allspark::DecOptMHAOp prev;
    prev.SetGenerateContext(genctx);
    prev.Init(proto, *device, weight_map, &tensor_map_prev);
    prev.Reshape();
    prev.Forward();
    DEBUG_TENSOR(prev, in0);
    DEBUG_TENSOR(prev, in1);
    DEBUG_TENSOR(prev, in2);
    DEBUG_TENSOR(prev, wss);
    DEBUG_TENSOR(prev, out);
    device->Synchronize();
    AsyncD2H(tensor_map_prev.at(out_name.c_str()).get(), vc_output_prev.data(),
             vc_output_prev.size(),
             static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_PREV

#if ENABLE_I8KV
    allspark::DecOptMHAI8CacheOp i8co;
    i8co.SetGenerateContext(genctx);
    i8co.Init(proto, *device, weight_map, &tensor_map_curr);
    i8co.Reshape(runtime_ctx.get());
    i8co.Forward(runtime_ctx.get());
    DEBUG_TENSOR(curr, in0);
    DEBUG_TENSOR(curr, in1);
    DEBUG_TENSOR(curr, in2);
    DEBUG_TENSOR(curr, wss);
    DEBUG_TENSOR(curr, out);
    device->Synchronize();
    AsyncD2H(tensor_map_curr.at(out_name.c_str()).get(), vc_output_curr.data(),
             vc_output_curr.size(),
             static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_I8KV

#if ENABLE_PREV
#if ENABLE_I8KV
    float diff = check_equal<T>(vc_output_prev.data(), vc_output_curr.data(),
                                squash_vector_shape(shape_output));
    // printf("[DIFF-ENCODER] max diff = %f\n", diff);
    EXPECT_EQ(diff < feps, true);
#endif  // ENABLE_I8KV
#endif  // ENABLE_PREV

    // first decoder
    std::vector<allspark::dim_t> shape_input0_decoder = shape_input0_context;
    std::vector<allspark::dim_t> shape_input1_decoder = shape_input1_context;
    shape_input0_decoder[1] = 1;
    genctx->step = xseql;
#if ENABLE_PREV
    tensor_map_prev.at(in0_name.c_str())
        ->SetShape(allspark::Shape(shape_input0_decoder));
#endif  // ENABLE_PREV
#if ENABLE_I8KV
    tensor_map_curr.at(in0_name.c_str())
        ->SetShape(allspark::Shape(shape_input0_decoder));
#endif  // ENABLE_I8KV

#if 1  // 2
    while (genctx->step + 1 < cache && loop_cnt < loop_max) {
      std::vector<T> vc_input0_decoder1 =
          rand_float<T>(squash_vector_shape(shape_input0_decoder));
      std::vector<float> vc_input1_decoder1 =
          rand_ninf_mask(squash_vector_shape(shape_input1_decoder));
      device->Synchronize();
#if ENABLE_PREV
      AsyncH2D(vc_input0_decoder1.data(),
               tensor_map_prev.at(in0_name.c_str()).get(),
               vc_input0_decoder1.size() * sizeof(T),
               static_cast<const CUDAContext*>(device.get())->GetStream());
      AsyncH2D(vc_input1_decoder1.data(),
               tensor_map_prev.at(in1_name.c_str()).get(),
               vc_input1_decoder1.size() * sizeof(float),
               static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_PREV
#if ENABLE_I8KV
      AsyncH2D(vc_input0_decoder1.data(),
               tensor_map_curr.at(in0_name.c_str()).get(),
               vc_input0_decoder1.size() * sizeof(T),
               static_cast<const CUDAContext*>(device.get())->GetStream());
      AsyncH2D(vc_input1_decoder1.data(),
               tensor_map_curr.at(in1_name.c_str()).get(),
               vc_input1_decoder1.size() * sizeof(float),
               static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_I8KV
      device->Synchronize();

#if ENABLE_PREV
      prev.Reshape();
      prev.Forward();
      device->Synchronize();
      DEBUG_TENSOR(prev, in0);
      DEBUG_TENSOR(prev, in1);
      DEBUG_TENSOR(prev, in2);
      DEBUG_TENSOR(prev, wss);
      DEBUG_TENSOR(prev, out);
      AsyncD2H(tensor_map_prev.at(out_name.c_str()).get(),
               vc_output_prev.data(), vc_output_prev.size(),
               static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_PREV

#if ENABLE_I8KV
      i8co.Reshape(runtime_ctx.get());
      i8co.Forward(runtime_ctx.get());
      device->Synchronize();
      DEBUG_TENSOR(curr, in0);
      DEBUG_TENSOR(curr, in1);
      DEBUG_TENSOR(curr, in2);
      DEBUG_TENSOR(curr, wss);
      DEBUG_TENSOR(curr, out);
      AsyncD2H(tensor_map_curr.at(out_name.c_str()).get(),
               vc_output_curr.data(), vc_output_curr.size(),
               static_cast<const CUDAContext*>(device.get())->GetStream());
#endif  // ENABLE_I8KV

#if ENABLE_PREV
#if ENABLE_I8KV
      diff = check_equal<T>(vc_output_prev.data(), vc_output_curr.data(),
                            squash_vector_shape(shape_output));
      // printf("[DIFF-DEC-%3d] max diff = %f\n", loop_cnt, diff);
      EXPECT_EQ(diff < feps, true);
#endif  // ENABLE_I8KV
#endif  // ENABLE_PREV

      genctx->step++;
      loop_cnt++;
    }
#endif  // 2nd
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
  }
  void TearDown() override {}

 protected:
  // ctrl
  int print_max = 0;
  int print_cnt = 0;
  // context
  allspark::OperatorProto proto;
  std::shared_ptr<allspark::DeviceContext> device;
};

template <typename T>
void TestDecMhaU4Cache(int batch, int phead, int nhead, int xseql, int cache,
                       int loop_max, float feps) {
  std::unique_ptr<allspark::RuntimeContext> runtime_ctx =
      std::make_unique<RuntimeContext>();
  runtime_ctx->PushBackGenCtx(std::make_shared<GenerateContext>());
  std::shared_ptr<GenerateContext> genctx = runtime_ctx->GetGenCtx(0);
  // context
  genctx->max_length = cache;
  genctx->batch_size = batch;
  genctx->step = 0;

  // input0 qkv
  std::string in0_name = "qkv";
  std::vector<allspark::dim_t> shape_input0_context = {batch, xseql,
                                                       3 * nhead * phead};
  std::vector<T> input0_context =
      rand_float<T>(squash_vector_shape(shape_input0_context));

  // input1 mask
  std::string in1_name = "mask";
  std::vector<allspark::dim_t> shape_input1_context = {batch * nhead, xseql,
                                                       xseql};
  std::vector<float> input1_context =
      rand_ninf_mask(squash_vector_shape(shape_input1_context));

  // output
  std::string out_name = "output";
  std::vector<allspark::dim_t> shape_output_context = {batch, xseql,
                                                       nhead * phead};
  std::vector<T> output_ref(squash_vector_shape(shape_output_context), 0.);
  std::vector<T> output_curr(squash_vector_shape(shape_output_context), 0.);

  const allspark::DeviceType device_type = allspark::DeviceType::CUDA;
  const allspark::DataMode data_mode = allspark::DataMode::DENSE;
  const allspark::DataType ft_data_type = allspark::DataTypeTrait<T>::data_type;

  // Test
  quantType quant_type = quantType::UINT4;
  TestOpUtil tu(device_type);
  tu.SetOpType("DecOptMHAI8Cache");
  tu.SetOpAttribute<int>("num_heads", nhead);
  tu.SetOpAttribute<int>("quant_type", quant_type);
  tu.AddInput(in0_name, shape_input0_context, device_type, ft_data_type,
              data_mode, input0_context, false);
  tu.AddInput(in1_name, shape_input1_context, device_type, ft_data_type,
              data_mode, input1_context, false);
  tu.AddOutput(out_name, shape_output_context, device_type, ft_data_type,
               data_mode);

  // DecMha as reference
  TestOpUtil tu_ref(device_type);
  tu_ref.SetOpType("DecOptMHA");
  tu_ref.SetOpAttribute<int>("num_heads", nhead);
  tu_ref.AddInput(in0_name, shape_input0_context, device_type, ft_data_type,
                  data_mode, input0_context, false);
  tu_ref.AddInput(in1_name, shape_input1_context, device_type, ft_data_type,
                  data_mode, input1_context, false);
  tu_ref.AddOutput(out_name, shape_output_context, device_type, ft_data_type,
                   data_mode);

  // first step
  allspark::DecOptMHAOp prev;
  prev.SetGenerateContext(genctx);
  prev.Init(tu_ref.GetOpProto(), *(tu_ref.GetDeviceContext()),
            tu_ref.GetWeightMap(), &(tu_ref.GetTensorMap()));
  prev.Reshape();
  prev.Forward();
  tu_ref.device_context_->Synchronize();
  auto prev_output_tensor = tu_ref.GetTensorMap().at(out_name.c_str());
  AsyncD2H(prev_output_tensor.get(), output_ref.data(),
           prev_output_tensor->GetShape().Count() * sizeof(T),
           static_cast<const CUDAContext*>(tu_ref.GetDeviceContext().get())
               ->GetStream());
  // printf("[prev]\t in0: \t%s\n",
  // tu_ref.GetTensorMap().at(in0_name.c_str())->ToString().c_str());
  // printf("[prev]\t in1: \t%s\n",
  // tu_ref.GetTensorMap().at(in1_name.c_str())->ToString().c_str());
  // printf("[prev]\t out: \t%s\n",
  // tu_ref.GetTensorMap().at(out_name.c_str())->ToString().c_str());

  allspark::DecOptMHAI8CacheOp u4co;
  u4co.SetGenerateContext(genctx);
  u4co.Init(tu.GetOpProto(), *(tu.GetDeviceContext()), tu.GetWeightMap(),
            &(tu.GetTensorMap()));
  u4co.Reshape(runtime_ctx.get());
  u4co.Forward(runtime_ctx.get());
  tu.device_context_->Synchronize();
  auto curr_output_tensor = tu.GetTensorMap().at(out_name.c_str());
  AsyncD2H(curr_output_tensor.get(), output_curr.data(),
           curr_output_tensor->GetShape().Count() * sizeof(T),
           static_cast<const CUDAContext*>(tu.GetDeviceContext().get())
               ->GetStream());
  // printf("[curr]\t in0: \t%s\n",
  // tu.GetTensorMap().at(in0_name.c_str())->ToString().c_str());
  // printf("[curr]\t in1: \t%s\n",
  // tu.GetTensorMap().at(in1_name.c_str())->ToString().c_str());
  // printf("[curr]\t out: \t%s\n",
  // tu.GetTensorMap().at(out_name.c_str())->ToString().c_str());

  // check
  float diff = check_equal<T>(output_ref.data(), output_curr.data(),
                              squash_vector_shape(shape_output_context));
  if (diff >= feps) {
    printf("[DIFF-ENCODER] max diff = %f, feps = %f\n", diff, feps);
  }
  // printf("[DIFF-ENCODER] max diff = %f\n", diff);
  EXPECT_EQ(diff < feps, true);

  // decoder start
#if 1
  // set decoder input/output shape
  std::vector<allspark::dim_t> shape_input0_decoder = shape_input0_context;
  std::vector<allspark::dim_t> shape_output_decoder = shape_output_context;
  shape_input0_decoder[1] = 1;
  shape_output_decoder[1] = 1;
  tu.GetTensorMap()
      .at(in0_name.c_str())
      ->SetShape(allspark::Shape(shape_input0_decoder));
  tu.GetTensorMap()
      .at(out_name.c_str())
      ->SetShape(allspark::Shape(shape_output_decoder));
  tu_ref.GetTensorMap()
      .at(in0_name.c_str())
      ->SetShape(allspark::Shape(shape_input0_decoder));
  tu_ref.GetTensorMap()
      .at(out_name.c_str())
      ->SetShape(allspark::Shape(shape_output_decoder));

  genctx->step = xseql;
  int loop_cnt = 0;
  while (genctx->step + 1 < cache && loop_cnt < loop_max) {
    // std::cout << "Decoder step : " << genctx->step + 1 - xseql << std::endl;
    // update input data
    std::vector<T> input0_decoder =
        rand_float<T>(squash_vector_shape(shape_input0_decoder));
    std::vector<allspark::dim_t> shape_input1_decoder = shape_input1_context;
    std::vector<float> input1_decoder =
        rand_ninf_mask(squash_vector_shape(shape_input1_decoder));

    AsyncH2D(
        input0_decoder.data(), tu.GetTensorMap().at(in0_name.c_str()).get(),
        input0_decoder.size() * sizeof(T),
        static_cast<const CUDAContext*>(tu.device_context_.get())->GetStream());
    AsyncH2D(
        input1_decoder.data(), tu.GetTensorMap().at(in1_name.c_str()).get(),
        input1_decoder.size() * sizeof(float),
        static_cast<const CUDAContext*>(tu.device_context_.get())->GetStream());

    AsyncH2D(input0_decoder.data(),
             tu_ref.GetTensorMap().at(in0_name.c_str()).get(),
             input0_decoder.size() * sizeof(T),
             static_cast<const CUDAContext*>(tu_ref.device_context_.get())
                 ->GetStream());
    AsyncH2D(input1_decoder.data(),
             tu_ref.GetTensorMap().at(in1_name.c_str()).get(),
             input1_decoder.size() * sizeof(float),
             static_cast<const CUDAContext*>(tu_ref.device_context_.get())
                 ->GetStream());

    tu.device_context_->Synchronize();
    tu_ref.device_context_->Synchronize();

    // Test
    prev.Reshape();
    prev.Forward();
    tu_ref.device_context_->Synchronize();
    auto prev_output_tensor = tu_ref.GetTensorMap().at(out_name.c_str());
    AsyncD2H(prev_output_tensor.get(), output_ref.data(),
             prev_output_tensor->GetShape().Count() * sizeof(T),
             static_cast<const CUDAContext*>(tu_ref.GetDeviceContext().get())
                 ->GetStream());
    // printf("[prev]\t in0: \t%s\n",
    // tu_ref.GetTensorMap().at(in0_name.c_str())->ToString().c_str());
    // printf("[prev]\t in1: \t%s\n",
    // tu_ref.GetTensorMap().at(in1_name.c_str())->ToString().c_str());
    // printf("[prev]\t out: \t%s\n",
    // tu_ref.GetTensorMap().at(out_name.c_str())->ToString().c_str());

    u4co.Reshape(runtime_ctx.get());
    u4co.Forward(runtime_ctx.get());
    tu.device_context_->Synchronize();
    auto curr_output_tensor = tu.GetTensorMap().at(out_name.c_str());
    AsyncD2H(curr_output_tensor.get(), output_curr.data(),
             curr_output_tensor->GetShape().Count() * sizeof(T),
             static_cast<const CUDAContext*>(tu.GetDeviceContext().get())
                 ->GetStream());
    // printf("[curr]\t in0: \t%s\n",
    // tu.GetTensorMap().at(in0_name.c_str())->ToString().c_str());
    // printf("[curr]\t in1: \t%s\n",
    // tu.GetTensorMap().at(in1_name.c_str())->ToString().c_str());
    // printf("[curr]\t out: \t%s\n",
    // tu.GetTensorMap().at(out_name.c_str())->ToString().c_str());

    // check
    float diff = check_equal<T>(output_ref.data(), output_curr.data(),
                                squash_vector_shape(shape_output_decoder));
    // printf("[DIFF-DEC-%3d] max diff = %f\n", loop_cnt, diff);
    if (diff >= feps) {
      printf("[DIFF-DEC-%3d] max diff = %f, feps = %f\n", loop_cnt, diff, feps);
    }
    EXPECT_EQ(diff < feps, true);

    genctx->step++;
    loop_cnt++;
  }
#endif
}

// notice, phead only support 32 / 64 / 128 / 256 / 512 now.
//                                                              batch,  phead,
//                                                              nhead,  xseql,
//                                                              cache, loop-max,
//                                                              feps
// TEST_F(DecMHATest, b3p128n10s2fp32_i8) {   test_dec_mha<float>(   3, 128, 10,
// 2,      64,     56,         2. * (1 / 128.));       } TEST_F(DecMHATest,
// b3p128n30s2fp16_i8) {   test_dec_mha<half>(    3,      128,    30,     2, 64,
// 10,         2. * (1 / 128.));       } TEST_F(DecMHATest, b11p32n10s2fp16_i8)
// {   test_dec_mha<half>(    11,     32,     10,     2,      64, 10,         2.
// * (1 / 128.));       } TEST_F(DecMHATest, b5p256n3s2fp16_i8) {
// test_dec_mha<half>(    5,      256,    3,      2,      64,     10,         2.
// * (1 / 128.));       } TEST_F(DecMHATest, b5p512n1s2fp16_i8) {
// test_dec_mha<half>(    5,      512,    1,      2,      32,     10,         2.
// * (1 / 128.));       } TEST_F(DecMHATest, b2p224n20s2fp16_i8) {
// test_dec_mha<half>(    2,      64,     20,     2,      32,     4,          2.
// * (1 / 128.));       } TEST_F(DecMHATest, perf0_fp16_i8) {
// test_dec_mha<half>(    1,      128,    20,     1536,   2048,   4,          2.
// * (1 / 128.));       } TEST_F(DecMHATest, b3p128n10s2fp32_u4)
// {TestDecMhaU4Cache<float>( 3,      128,    10,     2,      64, 56,         2.
// * (1 / 16.));       } TEST_F(DecMHATest, b3p128n30s2fp16_u4)
// {TestDecMhaU4Cache<half>(  3,      128,    30,     2,      64, 10,         2.
// * (1 / 16.));       } TEST_F(DecMHATest, b11p32n10s2fp16_u4)
// {TestDecMhaU4Cache<half>(  11,     32,     10,     2,      64, 10,         2.
// * (1 / 16.));       } TEST_F(DecMHATest, b5p256n3s2fp16_u4) {
// TestDecMhaU4Cache<half>(  5,      256,    3,      2,      64, 10,         2.
// * (1 / 16.));       }
// // this testcase can be passed locally, but the CR single test cannot be
// passed, no idea
// // TEST_F(DecMHATest, b5p512n1s2fp16_u4) { TestDecMhaU4Cache<half>(  5, 512,
// 1,      2,      32,     10,         2. * (1 / 16.));       }
// TEST_F(DecMHATest, b2p224n20s2fp16_u4) {TestDecMhaU4Cache<half>(  2,      64,
// 20,     2,      32,     4,          2. * (1 / 16.));       }
// TEST_F(DecMHATest, perf0_fp16_u4) {     TestDecMhaU4Cache<half>(  1, 128, 20,
// 1536,   2048,   4,          2. * (1 / 16.));       }
#endif
