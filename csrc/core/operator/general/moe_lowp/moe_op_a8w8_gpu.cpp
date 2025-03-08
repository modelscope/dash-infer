/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_op_a8w8_gpu.cpp
 */
#ifdef ENABLE_CUDA
#include "moe_op_a8w8_gpu.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

#if 0
// dubug code
static void print_info(void* input, const DeviceContext* ctx,
                       size_t layout_size = 0) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  std::vector<char> host_out(print_count * 3);
  cudaMemcpyAsync(host_out.data(), input, print_count, cudaMemcpyDeviceToHost,
                  cu_stream);
  cudaMemcpyAsync(host_out.data() + print_count, (char*)input + layout_size,
                  print_count, cudaMemcpyDeviceToHost, cu_stream);
  cudaMemcpyAsync(host_out.data() + 2 * print_count,
                  (char*)input + 2 * layout_size, print_count,
                  cudaMemcpyDeviceToHost, cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  half* ptr = static_cast<half*>(data_ptr);
  for (int i = 0; i < print_count * 3 / 2; i++) {
    LOG(INFO) << (float)ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}

static void print_array_pointer(void** input, int num,
                                const DeviceContext* ctx) {
  const int print_count = num;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  std::vector<char> host_out(print_count * 8);
  cudaMemcpyAsync(host_out.data(), input, print_count * 8,
                  cudaMemcpyDeviceToHost, cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  void** ptr = static_cast<void**>(data_ptr);
  for (int i = 0; i < print_count; i++) {
    LOG(INFO) << ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}

static void print_info_i32(void* input, const DeviceContext* ctx) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  std::vector<char> host_out(print_count * 4);
  cudaMemcpyAsync(host_out.data(), input, print_count * 4,
                  cudaMemcpyDeviceToHost, cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  int32_t* ptr = static_cast<int32_t*>(data_ptr);
  for (int i = 0; i < print_count; i++) {
    LOG(INFO) << ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}

static void print_info_i8(void* input, const DeviceContext* ctx) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  std::vector<char> host_out(print_count);
  cudaMemcpyAsync(host_out.data(), input, print_count, cudaMemcpyDeviceToHost,
                  cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  int8_t* ptr = static_cast<int8_t*>(data_ptr);
  for (int i = 0; i < print_count; i++) {
    LOG(INFO) << (int)ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}
static void print_arr_info(void** input_array, const DeviceContext* ctx) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  void* cpu_ptr;
  cudaMemcpyAsync(&cpu_ptr, input_array, sizeof(void**), cudaMemcpyDeviceToHost,
                  cu_stream);
  ctx->Synchronize();
  LOG(INFO) << "cpu_ptr= " << cpu_ptr;
  std::vector<char> host_out(print_count);
  cudaMemcpyAsync(host_out.data(), cpu_ptr, print_count, cudaMemcpyDeviceToHost,
                  cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  half* ptr = static_cast<half*>(data_ptr);
  for (int i = 0; i < print_count / 2; i++) {
    LOG(INFO) << (float)ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}

static void print_arr_info_i32(void** input_array, const DeviceContext* ctx) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  void* cpu_ptr;
  cudaMemcpyAsync(&cpu_ptr, input_array, sizeof(void**), cudaMemcpyDeviceToHost,
                  cu_stream);
  ctx->Synchronize();
  LOG(INFO) << "cpu_ptr= " << cpu_ptr;
  std::vector<char> host_out(print_count * 4);
  cudaMemcpyAsync(host_out.data(), cpu_ptr, print_count * 4,
                  cudaMemcpyDeviceToHost, cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  int32_t* ptr = static_cast<int32_t*>(data_ptr);
  for (int i = 0; i < print_count; i++) {
    LOG(INFO) << ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}

static void print_arr_info_i8(void** input_array, const DeviceContext* ctx) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  void* cpu_ptr;
  cudaMemcpyAsync(&cpu_ptr, input_array, sizeof(void**), cudaMemcpyDeviceToHost,
                  cu_stream);
  ctx->Synchronize();
  LOG(INFO) << "cpu_ptr= " << cpu_ptr;
  std::vector<char> host_out(print_count);
  cudaMemcpyAsync(host_out.data(), cpu_ptr, print_count, cudaMemcpyDeviceToHost,
                  cu_stream);

  ctx->Synchronize();
  void* data_ptr = host_out.data();
  int8_t* ptr = static_cast<int8_t*>(data_ptr);
  for (int i = 0; i < print_count; i++) {
    LOG(INFO) << (int32_t)ptr[i] << ",";
  }
  LOG(INFO) << std::endl;
}
static void gemm_arr_info(void** A_array, void** B_array, int m, int n, int k,
                          const DeviceContext* ctx) {
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  void* A_ptr;
  cudaMemcpyAsync(&A_ptr, A_array, sizeof(void**), cudaMemcpyDeviceToHost,
                  cu_stream);
  ctx->Synchronize();
  LOG(INFO) << "A_ptr= " << A_ptr;
  std::vector<char> host_A(m * k * 2);
  cudaMemcpyAsync(host_A.data(), A_ptr, m * k * 2, cudaMemcpyDeviceToHost,
                  cu_stream);
  void* B_ptr;
  cudaMemcpyAsync(&B_ptr, B_array, sizeof(void**), cudaMemcpyDeviceToHost,
                  cu_stream);
  ctx->Synchronize();
  LOG(INFO) << "B_ptr= " << B_ptr;
  std::vector<char> host_B(n * k * 2);
  cudaMemcpyAsync(host_B.data(), B_ptr, n * k * 2, cudaMemcpyDeviceToHost,
                  cu_stream);
  float sum = 0;
  void* data_ptr_A = host_A.data();
  half* ptr_A = static_cast<half*>(data_ptr_A);
  void* data_ptr_B = host_B.data();
  half* ptr_B = static_cast<half*>(data_ptr_B);
  for (int i = 0; i < k; i++) {
    sum += (float)ptr_A[i] * (float)ptr_B[i * n];
    LOG(INFO) << "(float)ptr_A[" << i << "] " << (float)ptr_A[i]
              << "(float)ptr_B[" << i * n << "]" << (float)ptr_B[i * n];
  }
  LOG(INFO) << "sum " << sum;
}
#endif
static int get_max_block(int input_token, int num_expert,
                         int num_expert_pertoken, int block_size) {
  int max_token = input_token * num_expert_pertoken;
  if (max_token < num_expert) {
    // max_block = max_token
    return max_token;
  }
  int max_block = num_expert + (max_token) / block_size;
  return max_block;
}
AsStatus MoeA8W8Gpu::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // Get Device SM Count
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  sm_count_ = device_prop.multiProcessorCount;
  sm_version_ = (device_prop.major << 8 | device_prop.minor);
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_experts") == attr_map.end()) {
    LOG(ERROR) << "MoeOp : can't find num_expert attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_expert_ = *(int*)(attr_map.at("num_experts").c_str());
  if (attr_map.find("num_experts_per_tok") == attr_map.end()) {
    LOG(ERROR) << "MoeOp : can't find num_expert_per_tok attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_expert_pertoken_ = *(int*)(attr_map.at("num_experts_per_tok").c_str());

  block_size_ = 64;
  first_moe_ = true;
  // default
  float_gate_score_ = std::make_unique<AsTensor>(
      "topk_value_", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_});
  topk_value_ = std::make_unique<AsTensor>(
      "topk_value_", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_});
  experts_score_ = std::make_unique<AsTensor>(
      "experts_score_", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  topk_indice_ = std::make_unique<AsTensor>(
      "topk_indice_", backend, DataType::INT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  experts_idx_ = std::make_unique<AsTensor>(
      "experts_idx_", backend, DataType::INT64, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  experts_seq_ = std::make_unique<AsTensor>(
      "experts_seq_", backend, DataType::INT64, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  indice_source_ = std::make_unique<AsTensor>(
      "indice_source_", backend, DataType::INT64, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  total_tokens_post_pad_ =
      std::make_unique<AsTensor>("total_tokens_post_pad_", backend,
                                 DataType::INT32, DataMode::DENSE, Shape{1});
  int max_block = get_max_block(ctx_->GetModelMaxLength(), num_expert_,
                                num_expert_pertoken_, block_size_);
  gate_up_proj_array_ptr = std::make_unique<AsTensor>(
      "gate_up_proj_array_ptr", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_block});
  down_proj_array_ptr = std::make_unique<AsTensor>(
      "down_proj_array_ptr", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_block});
  reorder_data_array_ptr = std::make_unique<AsTensor>(
      "reorder_data_array_ptr", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_block});
  gate_up_proj_out_array_ptr = std::make_unique<AsTensor>(
      "gate_up_proj_out_array_ptr", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_block});
  mid_result_array_ptr = std::make_unique<AsTensor>(
      "mid_result_array_ptr", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_block});
  final_result_array_ptr = std::make_unique<AsTensor>(
      "final_result_array_ptr", backend, DataType::INT64, DataMode::DENSE,
      Shape{max_block});
  std::unique_ptr<AsTensor> experts_num;
  std::unique_ptr<AsTensor> experts_seq;
  hidden_size_ = weights_[0]->GetShape()[1];
  proj_size_ = weights_[0]->GetShape()[2] / 2;
  // transpose dim1,dim2 for better batch gemm performance
  B_I8_Tranpose_Dim12_Gpu();
  return AsStatus::ALLSPARK_SUCCESS;
}
void MoeA8W8Gpu::B_I8_Tranpose_Dim12_Gpu() {
  void* rhs_gate_up_ptr = weights_[0]->GetDataPtr();
  // void* rhs_gate_up_scale_ptr = weights_[1]->GetDataPtr();
  // void* rhs_gate_up_zero_ptr = weights_[2]->GetDataPtr();
  void* rhs_down_ptr = weights_[3]->GetDataPtr();
  // void* rhs_down_scale_ptr = weights_[4]->GetDataPtr();
  // void* rhs_down_zero_ptr = weights_[5]->GetDataPtr();

  int m1 = weights_[0]->GetShape()[0];
  int k1 = weights_[0]->GetShape()[1];
  int n1 = weights_[0]->GetShape()[2];
  int m2 = weights_[3]->GetShape()[0];
  int k2 = weights_[3]->GetShape()[1];
  int n2 = weights_[3]->GetShape()[2];

  AsTensor reordered_gate_up_weight_gpu =
      AsTensor(weights_[0]->GetName() + "reordered_gate_up", DeviceType::CUDA,
               weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
               Shape({m1, n1, k1}));
  AsTensor reordered_down_weight_gpu =
      AsTensor(weights_[3]->GetName() + "reordered_down", DeviceType::CUDA,
               weights_[3]->GetDataType(), weights_[3]->GetDataMode(),
               Shape({m2, n2, k2}));
  int8_t* reordered_gate_up_weight_ptr =
      static_cast<int8_t*>(reordered_gate_up_weight_gpu.GetDataPtr());
  int8_t* reordered_down_weight_ptr =
      static_cast<int8_t*>(reordered_down_weight_gpu.GetDataPtr());

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  cuda::transpose_axis_12_kernelLauncher(reordered_gate_up_weight_ptr,
                                         (int8_t*)rhs_gate_up_ptr, m1, k1, n1,
                                         cu_stream);
  cuda::transpose_axis_12_kernelLauncher(
      reordered_down_weight_ptr, (int8_t*)rhs_down_ptr, m2, k2, n2, cu_stream);
  ctx_->Synchronize();
  auto* mutable_gate_up_weight = const_cast<AsTensor*>(weights_[0]);
  mutable_gate_up_weight->SetShape(Shape({m1, n1, k1}));
  TensorUtils::DeepCopyWholeAsync(*mutable_gate_up_weight,
                                  reordered_gate_up_weight_gpu, ctx_);
  auto* mutable_down_weight = const_cast<AsTensor*>(weights_[3]);
  mutable_down_weight->SetShape(Shape({m2, n2, k2}));
  TensorUtils::DeepCopyWholeAsync(*mutable_down_weight,
                                  reordered_down_weight_gpu, ctx_);
}

template <typename T>
void MoeA8W8Gpu::DispatchReshapeKernel(void* ws_ptr) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  int max_block = get_max_block(total_token_, num_expert_, num_expert_pertoken_,
                                block_size_);
  int max_total_tokens = max_block * block_size_;
  gate_up_proj_out = ws_ptr;
  mid_result = (char*)gate_up_proj_out +
               max_total_tokens * proj_size_ * 2 * SizeofType(dtype_);
  final_result =
      (char*)mid_result + max_total_tokens * proj_size_ * SizeofType(dtype_);
  gate_up_proj_array = (void**)gate_up_proj_array_ptr->GetDataPtr();
  down_proj_array = (void**)down_proj_array_ptr->GetDataPtr();
  reorder_data_array = (void**)reorder_data_array_ptr->GetDataPtr();
  gate_up_proj_out_array = (void**)gate_up_proj_out_array_ptr->GetDataPtr();
  mid_result_array = (void**)mid_result_array_ptr->GetDataPtr();
  final_result_array = (void**)final_result_array_ptr->GetDataPtr();

  // A reoreder qdata for batchgemmI8
  in_qdata = reinterpret_cast<int8_t*>((char*)final_result +
                                       max_block * sizeof(void**));
  in_reorder_qdata = reinterpret_cast<int8_t*>(
      (char*)in_qdata +
      aligned_size(max_total_tokens * hidden_size_) * sizeof(int8_t));
  in_scale = reinterpret_cast<float*>(
      (char*)in_reorder_qdata +
      aligned_size(max_total_tokens * hidden_size_) * sizeof(int8_t));
  in_reorder_scale = reinterpret_cast<float*>(
      (char*)in_scale + aligned_size(max_total_tokens) * sizeof(float));
  in_red_max = reinterpret_cast<float*>(
      (char*)in_reorder_scale + aligned_size(max_total_tokens) * sizeof(float));
  in_red_count = reinterpret_cast<uint32_t*>(
      (char*)in_red_max + aligned_size(max_total_tokens) * sizeof(float));
  in_red_sum = reinterpret_cast<int32_t*>(
      (char*)in_red_count + aligned_size(max_total_tokens) * sizeof(uint32_t));
  in_reorder_red_sum = reinterpret_cast<int32_t*>(
      (char*)in_red_sum + aligned_size(max_total_tokens) * sizeof(int32_t));
  // A mid qdata for batchgemmI8
  mid_qdata = reinterpret_cast<int8_t*>((char*)in_reorder_red_sum +
                                        aligned_size(max_total_tokens) *
                                            sizeof(int32_t));
  mid_scale = reinterpret_cast<float*>(
      (char*)mid_qdata +
      aligned_size(max_total_tokens * proj_size_) * sizeof(int8_t));
  mid_red_max = reinterpret_cast<float*>(
      (char*)mid_scale + aligned_size(max_total_tokens) * sizeof(float));
  mid_red_count = reinterpret_cast<uint32_t*>(
      (char*)mid_red_max + aligned_size(max_total_tokens) * sizeof(float));
  mid_red_sum = reinterpret_cast<int32_t*>(
      (char*)mid_red_count + aligned_size(max_total_tokens) * sizeof(uint32_t));

  // weight reorder scale/zero array
  gate_up_proj_scale_array = reinterpret_cast<void**>(
      (char*)mid_red_sum + aligned_size(max_total_tokens) * sizeof(int32_t));
  gate_up_proj_zero_array = reinterpret_cast<void**>(
      (char*)gate_up_proj_scale_array + max_block * sizeof(void**));
  down_proj_scale_array = reinterpret_cast<void**>(
      (char*)gate_up_proj_zero_array + max_block * sizeof(void**));
  down_proj_zero_array = reinterpret_cast<void**>((char*)down_proj_scale_array +
                                                  max_block * sizeof(void**));
  // C tmp batchgemmI8 output
  gate_up_proj_out_i32 = reinterpret_cast<int32_t*>(
      (char*)down_proj_zero_array + max_block * sizeof(void**));
  final_result_i32 = reinterpret_cast<int32_t*>(
      (char*)gate_up_proj_out_i32 +
      aligned_size(max_total_tokens * proj_size_ * 2) * sizeof(int32_t));
  cuda::MOEGetBatchArrayLauncher(
      nullptr, nullptr, (T*)in_reorder_qdata, reorder_data_array, max_block,
      block_size_ * hidden_size_, block_size_, cu_stream);
  cuda::MOEGetBatchArrayLauncher(
      nullptr, nullptr, (int32_t*)gate_up_proj_out_i32, gate_up_proj_out_array,
      max_block, block_size_ * proj_size_ * 2, block_size_, cu_stream);
  cuda::MOEGetBatchArrayLauncher(
      nullptr, nullptr, (T*)mid_qdata, mid_result_array, max_block,
      block_size_ * proj_size_, block_size_, cu_stream);
  cuda::MOEGetBatchArrayLauncher(
      nullptr, nullptr, (int32_t*)final_result_i32, final_result_array,
      max_block, block_size_ * hidden_size_, block_size_, cu_stream);
  // LOG(INFO) << "up_proj_out: " << up_proj_out << " final_result_i32: " <<
  // final_result_i32
  //           << " max_total_tokens: " << max_total_tokens << " max_block: " <<
  //           max_block << " block_size_: "<< block_size_ << " proj_size_: " <<
  //           proj_size_
  //           << " hidden_size_: " << hidden_size_;
}

template <typename FT, typename QT>
void MoeA8W8Gpu::DispatchKernel() {
  int top_k = num_expert_pertoken_;
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* expert_weight_tensor = tensor_map_->at(in_names_[1]).get();
  AsTensor* gate_up_proj_weight_tensor = weights_[0];
  AsTensor* gate_up_proj_scale_tensor = weights_[1];
  AsTensor* gate_up_proj_zero_tensor = weights_[2];
  AsTensor* down_proj_weight_tensor = weights_[3];
  AsTensor* down_proj_scale_tensor = weights_[4];
  AsTensor* down_proj_zero_tensor = weights_[5];
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  cuda::CastKernelLauncher((FT*)expert_weight_tensor->GetDataPtr(),
                           (float*)float_gate_score_->GetDataPtr(),
                           expert_weight_tensor->GetShape().Count(), cu_stream);
  cuda::StridedSoftmaxLauncher((float*)topk_value_->GetDataPtr(),
                               (float*)float_gate_score_->GetDataPtr(), nullptr,
                               nullptr, ws_ptr, ws_size_, total_token_,
                               num_expert_, cu_stream);
  cuda::TopKKernelLauncher((float*)experts_score_->GetDataPtr(),
                           (int*)topk_indice_->GetDataPtr(),
                           (float*)topk_value_->GetDataPtr(), total_token_,
                           num_expert_, top_k, cu_stream);

  int total_token_post_pad = 0;
  cuda::ReorderAndPaddingMOE(
      (int64_t*)experts_idx_->GetDataPtr(),
      (int64_t*)experts_seq_->GetDataPtr(),
      (int64_t*)indice_source_->GetDataPtr(), (int*)topk_indice_->GetDataPtr(),
      total_token_, num_expert_, num_expert_pertoken_, block_size_,
      (int*)total_tokens_post_pad_->GetDataPtr(), 0, nullptr, cu_stream);

  int* total_tokens_pad_ptr = (int*)total_tokens_post_pad_->GetDataPtr();
  int max_block = get_max_block(total_token_, num_expert_, num_expert_pertoken_,
                                block_size_);
  int max_total_tokens = max_block * block_size_;
  // LOG(INFO) << "max_block(array size): " << max_block
  //           << " m1: " << block_size_
  //           << " n1: " << proj_size_ * 2
  //           << " k1: " << hidden_size_
  //           << " m2: " << block_size_
  //           << " n2: " << hidden_size_
  //           << " k2: " << proj_size_;
  // quant A
  cuda::per_channel_symm_dynamic_quantization<FT>(
      (FT*)in_tensor->GetDataPtr(), in_qdata, in_scale, in_red_max,
      in_red_count, in_red_sum, total_token_, hidden_size_, sm_count_,
      cu_stream);
  cuda::GetReorderQData(
      in_reorder_qdata, in_reorder_scale, in_reorder_red_sum, in_qdata,
      in_scale, in_red_sum, (int64_t*)experts_idx_->GetDataPtr(),
      (int64_t*)experts_seq_->GetDataPtr(), total_tokens_pad_ptr,
      max_total_tokens, total_token_ * top_k, top_k, hidden_size_, block_size_,
      cu_stream);
  cuda::MOEQWeightGetBatchArrayLauncher(
      (int64_t*)experts_idx_->GetDataPtr(), total_tokens_pad_ptr,
      (QT*)gate_up_proj_weight_tensor->GetDataPtr(), gate_up_proj_array,
      (FT*)gate_up_proj_scale_tensor->GetDataPtr(), gate_up_proj_scale_array,
      (FT*)gate_up_proj_zero_tensor->GetDataPtr(), gate_up_proj_zero_array,
      max_block, gate_up_proj_weight_tensor->GetShape().Count(1),
      gate_up_proj_scale_tensor->GetShape().Count(1), block_size_, cu_stream);
  cuda::MOEQWeightGetBatchArrayLauncher(
      (int64_t*)experts_idx_->GetDataPtr(), total_tokens_pad_ptr,
      (QT*)down_proj_weight_tensor->GetDataPtr(), down_proj_array,
      (FT*)down_proj_scale_tensor->GetDataPtr(), down_proj_scale_array,
      (FT*)down_proj_zero_tensor->GetDataPtr(), down_proj_zero_array, max_block,
      down_proj_weight_tensor->GetShape().Count(1),
      down_proj_scale_tensor->GetShape().Count(1), block_size_, cu_stream);
  cuda::BatchGemmI8Wrapper(
      gate_up_proj_out_array, reorder_data_array, gate_up_proj_array,
      block_size_, proj_size_ * 2, hidden_size_, false, true, 1, 0,
      hidden_size_, hidden_size_, proj_size_ * 2, max_block, cublas_handle);
  // dequantized tmp output
  cuda::A_perc_symm_B_perc_array_asymm_dequantization<FT>(
      gate_up_proj_out_i32, in_reorder_scale, in_reorder_red_sum,
      (const FT**)gate_up_proj_scale_array, (const FT**)gate_up_proj_zero_array,
      nullptr, (FT*)gate_up_proj_out, block_size_, proj_size_ * 2, max_block,
      cu_stream);
  cuda::UnaryGLUKernelLauncher((FT*)mid_result, (FT*)gate_up_proj_out,
                               max_total_tokens, proj_size_, UnaryType::SILU,
                               cu_stream);
  // quantize mid_result to int8, how to???
  cuda::per_channel_symm_dynamic_quantization<FT>(
      (FT*)mid_result, mid_qdata, mid_scale, mid_red_max, mid_red_count,
      mid_red_sum, max_total_tokens, proj_size_, sm_count_, cu_stream);
  cuda::BatchGemmI8Wrapper(final_result_array, mid_result_array,
                           down_proj_array, block_size_, hidden_size_,
                           proj_size_, false, true, 1, 0, proj_size_,
                           proj_size_, hidden_size_, max_block, cublas_handle);
  // dequantized tmp output
  cuda::A_perc_symm_B_perc_array_asymm_dequantization<FT>(
      final_result_i32, mid_scale, mid_red_sum,
      (const FT**)(down_proj_scale_array), (const FT**)(down_proj_zero_array),
      nullptr, (FT*)final_result, block_size_, hidden_size_, max_block,
      cu_stream);
  cuda::FinalizeMoeRoutingKernelLauncher(
      (FT*)out_tensor->GetDataPtr(), (FT*)final_result,
      (float*)experts_score_->GetDataPtr(),
      (int64_t*)indice_source_->GetDataPtr(), (int*)topk_indice_->GetDataPtr(),
      total_tokens_pad_ptr, total_token_, num_expert_pertoken_, hidden_size_, 0,
      nullptr, cu_stream);
}
AsStatus MoeA8W8Gpu::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  ftype_ = tensor_map_->at(in_names_[0])->GetDataType();
  if (first_moe_) {
    // only reshape once,for warmup
    first_moe_ = false;
    total_token_ = ctx_->GetModelMaxLength();
    int max_block = get_max_block(total_token_, num_expert_,
                                  num_expert_pertoken_, block_size_);
    int max_total_tokens = max_block * block_size_;
    expert_size_ = (int64_t)hidden_size_ * proj_size_;
    float_gate_score_->SetShape(Shape{total_token_, num_expert_});
    topk_value_->SetShape(Shape{total_token_, num_expert_});
    experts_score_->SetShape(Shape{total_token_, num_expert_pertoken_});
    topk_indice_->SetShape(Shape{total_token_, num_expert_pertoken_});
    experts_idx_->SetShape(Shape{max_block});
    experts_seq_->SetShape(Shape{max_total_tokens});
    indice_source_->SetShape(Shape{max_total_tokens});
    ws_size_ = 0;
#ifdef ENABLE_CUDA
    size_t softmax_workspace = 0;
    cuda::StridedSoftmaxGetWorkspaceSize<float>(
        &softmax_workspace, ctx_->GetModelMaxLength(), num_expert_);
    AS_CHECK_STATUS(
        tensor_map_->at("workspace")
            ->SetShape(Shape{static_cast<dim_t>(softmax_workspace)}));
    ws_size_ += softmax_workspace;
#endif
    ws_size_ += max_total_tokens * proj_size_ * 2 *
                SizeofType(dtype_);  // gate_up_proj_out
    ws_size_ +=
        max_total_tokens * proj_size_ * SizeofType(dtype_);  // mid_result
    ws_size_ +=
        max_total_tokens * hidden_size_ * SizeofType(dtype_);  // final_result
    // ws_size_ += 6 * max_block * sizeof(void**);
    // quantized input data
    ws_size_ += aligned_size(max_total_tokens * hidden_size_) *
                sizeof(int8_t);  // perchannel qdata for input before reorder
    ws_size_ += aligned_size(max_total_tokens * hidden_size_) *
                sizeof(int8_t);  // reorder_qdata in
    ws_size_ += aligned_size(max_total_tokens) * sizeof(float);  // A scale
    ws_size_ +=
        aligned_size(max_total_tokens) * sizeof(float);  // reorder A scale
    ws_size_ += aligned_size(max_total_tokens) * sizeof(float);  // A red_max
    ws_size_ +=
        aligned_size(max_total_tokens) * sizeof(uint32_t);  // A red_count
    ws_size_ += aligned_size(max_total_tokens) * sizeof(int32_t);  // A red_sum
    ws_size_ +=
        aligned_size(max_total_tokens) * sizeof(int32_t);  // reorder A red_sum

    // quantized mid_data
    ws_size_ += aligned_size(max_total_tokens * proj_size_) *
                sizeof(int8_t);  // mid_data perchannel qdata
    ws_size_ += aligned_size(max_total_tokens) * sizeof(float);  // A scale
    ws_size_ += aligned_size(max_total_tokens) * sizeof(float);  // A red_max
    ws_size_ +=
        aligned_size(max_total_tokens) * sizeof(uint32_t);  // A red_count
    ws_size_ += aligned_size(max_total_tokens) * sizeof(int32_t);  // A red_sum

    // weight scale/zero data
    ws_size_ += max_block * sizeof(void**);  // gate_up_proj_scale_array
    ws_size_ += max_block * sizeof(void**);  // gate_up_proj_zero_array
    ws_size_ += max_block * sizeof(void**);  // down_proj_scale_array
    ws_size_ += max_block * sizeof(void**);  // down_proj_zero_array

    // tmp gate_up_proj_out int32
    ws_size_ += aligned_size(max_total_tokens * proj_size_ * 2) *
                SizeofType(DataType::INT32);
    // tmp final_result int32
    ws_size_ += aligned_size(max_total_tokens * hidden_size_) *
                SizeofType(DataType::INT32);

    AS_CHECK_STATUS(tensor_map_->at("workspace")->SetShape(Shape{(ws_size_)}));
    AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
    AsTensor* expert_weight_tensor = tensor_map_->at(in_names_[1]).get();
    AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
    void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
    ws_ptr = (char*)ws_ptr + softmax_workspace;
    DispatchReshapeKernel<int8_t>(ws_ptr);
  }

  total_token_ = out_shape[0] * out_shape[1];
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MoeA8W8Gpu::Forward() {
  if (weights_.size() < 6) {
    LOG(ERROR) << "MoeA8W8Gpu weights_: " << weights_.size()
               << " < should be greater than 6";
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  switch (ftype_) {
#ifdef ENABLE_FP16
    case DataType::FLOAT16: {
      DispatchKernel<half, int8_t>();
      break;
    }
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16: {
      DispatchKernel<hie::bfloat16, int8_t>();
      break;
    }
#endif
    default:
      LOG(ERROR) << "GemmA8W8GPU DataType Error\n";
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(MOEA8W8, CUDA, MoeA8W8Gpu)
}  // namespace allspark
#endif