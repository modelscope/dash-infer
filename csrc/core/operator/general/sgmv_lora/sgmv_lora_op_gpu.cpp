/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sgmv_lora_op_gpu.cpp
 */

#ifdef ENABLE_CUDA
#include "sgmv_lora_op_gpu.h"

#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/kernel/kernel.h>
#include <cuda/cuda_context.h>
#include <utility/check_cuda.h>
#include <utility/datatype_dispatcher.h>

#include <numeric>

#include "runtime/weight/weight_manager_lora.h"

#define ALIGN_UP_TO_16(ptr)                                         \
  reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(ptr) + 15) & \
                          ~uintptr_t(15))

#define INT_ROUND_UP_TO_16(num) (num + 15) & ~15

namespace allspark {
namespace cuda {
extern size_t sgmv_tmp_size(int num_problems);
}
}  // namespace allspark

namespace allspark {
AsStatus sgmv_cutlass(DataType dtype, void* out, const void* in,
                      const AsTensor* weight_ptrs, const AsTensor* segments,
                      const AsTensor* ranks, void* buf, int d_in, int d_out,
                      bool is_k_tensor, bool is_n_tensor, int num_problems,
                      bool unsplit, int unsplit_n, int max_rank, int CC,
                      const DeviceContext* ctx);

AsStatus sgmv_split_qkv(DataType dtype, AsTensor* out_ptrs, const void* in,
                        const AsTensor* segments, const AsTensor* lora_B_ranks,
                        int max_rank, int num_problems,
                        const DeviceContext* ctx);

AsStatus dense_gemm_rawptr(DataType dtype, void* out, const void* in,
                           const void* bias, const void* weight, int m, int n,
                           int k, int lda, int ldb, int ldc, bool transA,
                           bool transB, int batch, float alpha,
                           const void* binary_in, UnaryType activation,
                           const DeviceContext* ctx);

AsStatus SgmvLoraOpGPU::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  DLOG(INFO) << "SgmvLoraOpGPU::Init" << std::endl;
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  const int max_batch = ctx.GetModelMaxBatch();
  // check if this op is ...attention.self
  if (weight_names_[0].rfind("attention.self") != std::string::npos) {
    is_attention_self_ = true;
    for (int i = 0; i < 3; i++) {
      lora_B_weight_parts_vec_.emplace_back(std::make_shared<AsTensor>(
          "lora_B_weight_parts." + std::to_string(i), ctx.GetDeviceType(),
          DataType::INT64, DataMode::DENSE, Shape{max_batch}));
    }
  }
  AS_ENFORCE(ctx.GetDeviceType() == DeviceType::CUDA);  // sgmv暂只支持CUDA
  lora_A_weight_ptrs_ = std::make_shared<AsTensor>(
      "lora_A_weight_ptrs_", ctx.GetDeviceType(), DataType::INT64,
      DataMode::DENSE, Shape{max_batch});
  lora_B_weight_ptrs_ = std::make_shared<AsTensor>(
      "lora_B_weight_ptrs_", ctx.GetDeviceType(), DataType::INT64,
      DataMode::DENSE, Shape{max_batch});
  lora_ranks_ = std::make_shared<AsTensor>("lora_ranks_", ctx.GetDeviceType(),
                                           DataType::INT32, DataMode::DENSE,
                                           Shape{max_batch});
  segments_ = std::make_shared<AsTensor>("segments_", ctx.GetDeviceType(),
                                         DataType::INT32, DataMode::DENSE,
                                         Shape{max_batch * 2});
  if (is_attention_self_ == true) {
    lora_B_ranks_ = std::make_shared<AsTensor>(
        "lora_B_ranks_", ctx.GetDeviceType(), DataType::INT32, DataMode::DENSE,
        Shape{max_batch});
    temp_qkv_ptrs_ =
        std::make_shared<AsTensor>("temp_qkv_ptrs", ctx.GetDeviceType(),
                                   DataType::INT64, DataMode::DENSE, Shape{3});
    int nslice = ctx.GetNranks();
    AS_ENFORCE(ctx.GetNumberHeads() * ctx.GetSizePerHead() % nslice == 0);
    AS_ENFORCE(ctx.GetNumberGroups() * ctx.GetSizePerHead() % nslice == 0);
    q_outdim_size_ = ctx.GetNumberHeads() * ctx.GetSizePerHead() / nslice;
    k_outdim_size_ = ctx.GetNumberGroups() * ctx.GetSizePerHead() / nslice;
    v_outdim_size_ = k_outdim_size_;
    qkv_weight_dims_.emplace_back(q_outdim_size_);
    qkv_weight_dims_.emplace_back(k_outdim_size_);
    qkv_weight_dims_.emplace_back(v_outdim_size_);
    qkv_sum_ = q_outdim_size_ + k_outdim_size_ + v_outdim_size_;
  }
  kernel_launcher = sgmv_cutlass;

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    throw AsException("Error getting device properties in SgmvLoraOpGPU::Init");
  }
  CC_ = prop.major * 10 + prop.minor;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SgmvLoraOpGPU::Reshape(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "SgmvLoraOpGPU::Reshape" << std::endl;

  // reset
  max_lora_r_ = 0;
  need_set_zero_ = false;
  use_cublas_ = false;
  lora_A_weight_ptrs_vec_.clear();
  lora_B_weight_ptrs_vec_.clear();
  lora_B_weight_parts_data_ptrs_[0].clear();  // q
  lora_B_weight_parts_data_ptrs_[1].clear();  // k
  lora_B_weight_parts_data_ptrs_[2].clear();  // v
  lora_A_ranks_vec_.clear();
  lora_B_ranks_vec_.clear();
  segmented_batch_idx_.clear();

  //  根据batch里req用到的lora，把不在cuda中的lora换入
  auto batchsize =
      runtime_ctx->is_context ? 1 : runtime_ctx->GetGenCtxListSize();

  // 用于batch_size维度给输入分段
  std::string last_lora_name = "";
  std::vector<int64_t> lora_A_weight_data_ptrs;
  std::vector<int64_t> lora_B_weight_data_ptrs;

  const Shape& in_shape = tensor_map_->at(in_names_[0])->GetShape();
  int in_dims = in_shape.Size();
  AS_ENFORCE(in_dims == 3 &&
             batchsize == in_shape[0]);  // {bs, seq_len, in_features}

  auto seq_len = in_shape[1];
  auto in_features = in_shape[in_dims - 1];
  int64_t out_features = 0;

  // 计算weight指针和分段batch idx
  // batch中至少有一个请求带lora，不然本算子不会被调用
  // weight_names_[0] = "...lora_A.weight" weight_names_[1] = "...lora_B.weight"
  for (auto i = 0; i < batchsize; i++) {
    GenerateContext* gen_ctx = runtime_ctx->is_context
                                   ? runtime_ctx->GetContextGenCtx()
                                   : runtime_ctx->GetGenCtx(i);
    auto lora_name = gen_ctx->gen_cfg.lora_name;
    if (lora_name.empty()) {  // 加载了lora权重 但请求中可以不使用
      need_set_zero_ = true;
      if (last_lora_name.empty() == false) {
        last_lora_name = "";
        // 前面一直带有lora，当前batch不带lora，增加end index
        segmented_batch_idx_.emplace_back(i);
      }
      continue;
    }
    auto lora_weight_handle = lora_manager_->GetHandleByName(lora_name);

    // 获取weights
    auto& lora_A_weight_name = weight_names_[0];
    auto lora_A_weight_p =
        lora_manager_->GetLoraTensorByName(lora_name, lora_A_weight_name);
    auto& lora_B_weight_name = weight_names_[1];
    auto lora_B_weight_p =
        lora_manager_->GetLoraTensorByName(lora_name, lora_B_weight_name);

    // check bias
    bool has_lora_bias =
        lora_manager_->HasLoraBias(lora_name, lora_A_weight_name);
    has_lora_bias = has_lora_bias ? true
                                  : lora_manager_->HasLoraBias(
                                        lora_name, lora_B_weight_name);
    if (has_lora_bias == true) {
      throw AsException("SgmvLoraOp does not support lora with bias!");
    }

    AS_ENFORCE(in_features == lora_A_weight_p->GetShape()[0]);
    if (out_features == 0) {
      out_features = lora_B_weight_p->GetShape()[1];
      if (is_attention_self_ == true) {
        AS_ENFORCE(out_features ==
                   q_outdim_size_ + k_outdim_size_ + v_outdim_size_);
      }
    }
    // auto lora_r = std::min(weight_tensor_p->GetShape()[0],
    //                        weight_tensor_p->GetShape()[1]);
    auto lora_r = lora_A_weight_p->GetShape()[1];
    if (lora_B_weight_p->GetShape()[0] < 8 && use_cublas_ == false) {
      // lora_B weight rank < 8启用cublas
      use_cublas_ = true;
    }
    if (lora_r >= max_lora_r_) {
      max_lora_r_ = lora_r;
    }

    if (last_lora_name != lora_name) {
      if (last_lora_name.empty() == false) {
        // 前面一直带有lora，当前batch更换lora_name，增加end index
        segmented_batch_idx_.emplace_back(i);
      }
      last_lora_name = lora_name;
      lora_A_weight_data_ptrs.emplace_back(
          (int64_t)
              lora_A_weight_p->GetDataPtr());  // 可能有重复的weight，但没有影响
      lora_B_weight_data_ptrs.emplace_back(
          (int64_t)lora_B_weight_p->GetDataPtr());
      if (is_attention_self_ == true) {
        // attention.self处lora算子，需要列向拆分lora_B.weight{rank,
        // out_features}为qkv三份
        int lora_B_rank = lora_B_weight_p->GetShape()[0];
        AS_ENFORCE(lora_B_rank * 3 == lora_r);
        lora_B_ranks_vec_.emplace_back(lora_B_rank);
        for (int qkv_idx = 0; qkv_idx < 3; qkv_idx++) {
          lora_B_weight_parts_data_ptrs_[qkv_idx].emplace_back(
              (int64_t)((char*)lora_B_weight_p->GetDataPtr() +
                        std::accumulate(qkv_weight_dims_.begin(),
                                        qkv_weight_dims_.begin() + qkv_idx, 0) *
                            SizeofType(dtype_)));
        }
      }
      lora_A_ranks_vec_.emplace_back(lora_r);
      segmented_batch_idx_.emplace_back(i);  // 新的starting batch
      lora_A_weight_ptrs_vec_.emplace_back(lora_A_weight_p.get());
      lora_B_weight_ptrs_vec_.emplace_back(lora_B_weight_p.get());
    }
    if (i == batchsize - 1) {
      if (last_lora_name.empty() == false) {
        // 遍历即将结束，当前的lora请求还没有记录end index
        segmented_batch_idx_.emplace_back(batchsize);
      }
    }
  }
  AS_ENFORCE(segmented_batch_idx_.size() == lora_A_weight_data_ptrs.size() * 2);
  num_problems_ = lora_A_weight_data_ptrs.size();

  if (runtime_ctx->is_context == true) {
    // context阶段bs=1，sgmv的m应该是seq_len
    segmented_batch_idx_[0] = 0;
    segmented_batch_idx_[1] = seq_len;
  }

  // Calculate workspace size
  buf_size_ = cuda::sgmv_tmp_size(num_problems_);
  ws_size_ = 0;
  ws_size_ +=
      batchsize * seq_len * max_lora_r_ * SizeofType(dtype_);  // temp_(dtype)
  ws_size_ = INT_ROUND_UP_TO_16(ws_size_);
  ws_size_ += buf_size_;  // buf_(uint8)
  if (is_attention_self_ == true) {
    ws_size_ = INT_ROUND_UP_TO_16(ws_size_);
    ws_size_ += batchsize * seq_len * (max_lora_r_ / 3) *
                SizeofType(dtype_);  // temp_qkv_[0](dtype)
    ws_size_ = INT_ROUND_UP_TO_16(ws_size_);
    ws_size_ += batchsize * seq_len * (max_lora_r_ / 3) *
                SizeofType(dtype_);  // temp_qkv_[1](dtype)
    ws_size_ = INT_ROUND_UP_TO_16(ws_size_);
    ws_size_ += batchsize * seq_len * (max_lora_r_ / 3) *
                SizeofType(dtype_);  // temp_qkv_[2](dtype)
  }
  AS_CHECK_STATUS(tensor_map_->at("workspace")->SetShape(Shape{(ws_size_)}));

  // Reshape partial tensors
  lora_A_weight_ptrs_->SetShape(Shape{lora_A_weight_data_ptrs.size()});
  lora_B_weight_ptrs_->SetShape(Shape{lora_B_weight_data_ptrs.size()});
  lora_ranks_->SetShape(Shape{lora_A_ranks_vec_.size()});
  segments_->SetShape(Shape{segmented_batch_idx_.size()});
  tensor_map_->at(out_names_[0])
      ->SetShape(Shape{batchsize, seq_len, out_features});
  if (is_attention_self_ == true) {
    lora_B_ranks_->SetShape(Shape{lora_B_ranks_vec_.size()});
    for (int qkv_idx = 0; qkv_idx < 3; qkv_idx++) {
      lora_B_weight_parts_vec_[qkv_idx]->SetShape(
          Shape{lora_B_weight_parts_data_ptrs_[qkv_idx].size()});
    }
  }

  // Copy data
  // Init已经确保了使用CUDA
  CopyData(lora_A_weight_ptrs_->GetDataPtr(), DeviceType::CUDA,
           lora_A_weight_data_ptrs.data(), DeviceType::CPU,
           sizeof(int64_t) * lora_A_weight_data_ptrs.size(), ctx_);
  CopyData(lora_B_weight_ptrs_->GetDataPtr(), DeviceType::CUDA,
           lora_B_weight_data_ptrs.data(), DeviceType::CPU,
           sizeof(int64_t) * lora_B_weight_data_ptrs.size(), ctx_);
  CopyData(lora_ranks_->GetDataPtr(), DeviceType::CUDA,
           lora_A_ranks_vec_.data(), DeviceType::CPU,
           sizeof(int32_t) * lora_A_ranks_vec_.size(), ctx_);
  CopyData(segments_->GetDataPtr(), DeviceType::CUDA,
           segmented_batch_idx_.data(), DeviceType::CPU,
           sizeof(int32_t) * segmented_batch_idx_.size(), ctx_);
  if (is_attention_self_ == true) {
    CopyData(lora_B_ranks_->GetDataPtr(), DeviceType::CUDA,
             lora_B_ranks_vec_.data(), DeviceType::CPU,
             sizeof(int32_t) * lora_B_ranks_vec_.size(), ctx_);
    for (int qkv_idx = 0; qkv_idx < 3; qkv_idx++) {
      CopyData(lora_B_weight_parts_vec_[qkv_idx]->GetDataPtr(),
               DeviceType::CUDA, lora_B_weight_parts_data_ptrs_[qkv_idx].data(),
               DeviceType::CPU,
               sizeof(int64_t) * lora_B_weight_parts_data_ptrs_[qkv_idx].size(),
               ctx_);
    }
  }

  // 获取batch间一致的矩阵维数
  lora_A_d_in_ = in_features;
  lora_A_d_out_ = 0;  // fill by ranks tensor
  lora_B_d_in_ = 0;   // fill by ranks tensor
  lora_B_d_out_ = out_features;

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SgmvLoraOpGPU::Forward(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "SgmvLoraOpGPU::Forward" << std::endl;
  AsTensor* batch_in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* batch_out_tensor = tensor_map_->at(out_names_[0]).get();
  void* in = batch_in_tensor->GetDataPtr();
  void* out = batch_out_tensor->GetDataPtr();

  const auto& shape_in = batch_in_tensor->GetShape();
  const auto& shape_out = batch_out_tensor->GetShape();
  AS_ENFORCE(shape_in[0] == shape_out[0]);
  auto ndims_in = shape_in.Size();
  auto ndims_out = shape_out.Size();
  AS_ENFORCE(ndims_in == 3 && ndims_in == ndims_out);
  auto batchsize = shape_in[0];
  auto seq_len = shape_in[1];

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  // Calculate addresses
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  temp_ = ws_ptr;
  buf_ = (char*)temp_ + batchsize * seq_len * max_lora_r_ * SizeofType(dtype_);
  buf_ = ALIGN_UP_TO_16(buf_);
  if (is_attention_self_ == true) {
    temp_qkv_[0] = (char*)buf_ + buf_size_;
    temp_qkv_[0] = ALIGN_UP_TO_16(temp_qkv_[0]);
    temp_qkv_[1] = (char*)temp_qkv_[0] +
                   batchsize * seq_len * (max_lora_r_ / 3) * SizeofType(dtype_);
    temp_qkv_[1] = ALIGN_UP_TO_16(temp_qkv_[1]);
    temp_qkv_[2] = (char*)temp_qkv_[1] +
                   batchsize * seq_len * (max_lora_r_ / 3) * SizeofType(dtype_);
    temp_qkv_[2] = ALIGN_UP_TO_16(temp_qkv_[2]);
    CopyData(temp_qkv_ptrs_->GetDataPtr(), DeviceType::CUDA, temp_qkv_.data(),
             DeviceType::CPU, sizeof(int64_t) * 3, ctx_);
  }

  if (use_cublas_ == true) {
    // 存在lora weight rank < 8的请求，使用cublas kernel
    for (int i = 0; i < num_problems_; i++) {
      int batch_start = segmented_batch_idx_[i * 2];
      int batch_end = segmented_batch_idx_[i * 2 + 1];
      int out_byte_offset = max_lora_r_ * batch_start * SizeofType(dtype_);
      int in_byte_offset = shape_in[-1] * batch_start * SizeofType(dtype_);

      int m = batch_end - batch_start;
      int n = lora_A_ranks_vec_[i];
      int k = lora_A_d_in_;
      AS_CHECK_STATUS(
          dense_gemm_rawptr(dtype_, (void*)((char*)temp_ + out_byte_offset),
                            (const void*)((char*)in + in_byte_offset), nullptr,
                            lora_A_weight_ptrs_vec_[i]->GetDataPtr(), m, n, k,
                            k, n, n, false, false, 1, 1.0 /* default alpha */,
                            nullptr, UNARYTYPE_UNDEFINED, ctx_));
    }
  } else {
    AS_CHECK_STATUS(kernel_launcher(
        dtype_, temp_, batch_in_tensor->GetDataPtr(), lora_A_weight_ptrs_.get(),
        segments_.get(), lora_ranks_.get(), buf_, lora_A_d_in_, lora_A_d_out_,
        false, true, num_problems_, false, 0, max_lora_r_, CC_, ctx_));
  }

  if (is_attention_self_ == true) {
    sgmv_split_qkv(dtype_, temp_qkv_ptrs_.get(), temp_, segments_.get(),
                   lora_B_ranks_.get(), max_lora_r_, num_problems_, ctx_);
    for (int qkv_idx = 0; qkv_idx < 3; qkv_idx++) {
      // output_parts_[qkv_idx] = temp_ @ lora_B_weight_part[qkv_idx], shape:
      // {bs, seq_len, qkv_weight_dims_[qkv_idx]}
      if (use_cublas_ == true) {
        for (int i = 0; i < num_problems_; i++) {
          int batch_start = segmented_batch_idx_[i * 2];
          int batch_end = segmented_batch_idx_[i * 2 + 1];
          int out_byte_offset = qkv_sum_ * batch_start * SizeofType(dtype_);
          int in_byte_offset =
              (max_lora_r_ / 3) * batch_start * SizeofType(dtype_);

          int m = batch_end - batch_start;
          int n = qkv_weight_dims_[qkv_idx];
          int k = lora_B_ranks_vec_[i];
          AS_CHECK_STATUS(dense_gemm_rawptr(
              dtype_,
              (void*)((char*)batch_out_tensor->GetDataPtr() + out_byte_offset +
                      std::accumulate(qkv_weight_dims_.begin(),
                                      qkv_weight_dims_.begin() + qkv_idx, 0) *
                          SizeofType(dtype_)),
              (const void*)((char*)temp_qkv_[qkv_idx] + in_byte_offset),
              nullptr,
              (void*)(lora_B_weight_parts_data_ptrs_[qkv_idx][i] +
                      std::accumulate(qkv_weight_dims_.begin(),
                                      qkv_weight_dims_.begin() + qkv_idx, 0) *
                          SizeofType(dtype_)),
              m, n, k, k, qkv_sum_, qkv_sum_, false, false, 1,
              1.0 /* default alpha */, nullptr, UNARYTYPE_UNDEFINED, ctx_));
        }
      } else {
        AS_CHECK_STATUS(kernel_launcher(
            dtype_,
            (void*)((char*)batch_out_tensor->GetDataPtr() +
                    std::accumulate(qkv_weight_dims_.begin(),
                                    qkv_weight_dims_.begin() + qkv_idx, 0) *
                        SizeofType(dtype_)),
            temp_qkv_[qkv_idx], lora_B_weight_parts_vec_[qkv_idx].get(),
            segments_.get(), lora_B_ranks_.get(), buf_, lora_B_d_in_,
            qkv_weight_dims_[qkv_idx], true, false, num_problems_, true,
            qkv_sum_, max_lora_r_, CC_, ctx_));
      }
    }
  } else {
    // batch_out_tensor = temp_ @ lora_B, shape: {bs, seq_len, out_features}
    if (use_cublas_ == true) {
      for (int i = 0; i < num_problems_; i++) {
        int batch_start = segmented_batch_idx_[i * 2];
        int batch_end = segmented_batch_idx_[i * 2 + 1];
        int out_byte_offset =
            batch_out_tensor->GetShape()[-1] * batch_start * SizeofType(dtype_);
        int in_byte_offset = max_lora_r_ * batch_start * SizeofType(dtype_);

        int m = batch_end - batch_start;
        int n = lora_B_d_out_;
        int k = lora_A_ranks_vec_[i];  // 非attention.self的场景rank_A == rank_B
        AS_CHECK_STATUS(dense_gemm_rawptr(
            dtype_,
            (void*)((char*)batch_out_tensor->GetDataPtr() + out_byte_offset),
            (const void*)((char*)temp_ + in_byte_offset), nullptr,
            lora_B_weight_ptrs_vec_[i]->GetDataPtr(), m, n, k, k, n, n, false,
            false, 1, 1.0 /* default alpha */, nullptr, UNARYTYPE_UNDEFINED,
            ctx_));
      }
    } else {
      AS_CHECK_STATUS(kernel_launcher(
          dtype_, batch_out_tensor->GetDataPtr(), temp_,
          lora_B_weight_ptrs_.get(), segments_.get(), lora_ranks_.get(), buf_,
          lora_B_d_in_, lora_B_d_out_, true, false, num_problems_, false, 0,
          max_lora_r_, CC_, ctx_));
    }
  }

  if (need_set_zero_ == true) {
    // batch有请求不带lora_name
    int last_end_idx = 0;
    for (int i = 0; i < segmented_batch_idx_.size() - 1; i += 2) {
      if (segmented_batch_idx_[i] != last_end_idx) {
        // 这段tensor需要赋0
        int row_count = segmented_batch_idx_[i] - last_end_idx;
        AS_CHECK_CUDA(cudaMemsetAsync(
            (char*)batch_out_tensor->GetDataPtr() +
                last_end_idx * lora_B_d_out_ * SizeofType(dtype_),
            0, SizeofType(dtype_) * lora_B_d_out_ * row_count, cu_stream));
      }
      last_end_idx = segmented_batch_idx_[i + 1];
    }
    if (segmented_batch_idx_[segmented_batch_idx_.size() - 1] != batchsize) {
      // batch的最后需要赋0
      int row_count =
          batchsize - segmented_batch_idx_[segmented_batch_idx_.size() - 1];
      AS_CHECK_CUDA(cudaMemsetAsync(
          (char*)batch_out_tensor->GetDataPtr() +
              segmented_batch_idx_[segmented_batch_idx_.size() - 1] *
                  lora_B_d_out_ * SizeofType(dtype_),
          0, SizeofType(dtype_) * lora_B_d_out_ * row_count, cu_stream));
    }
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(SgmvLora, CUDA, SgmvLoraOpGPU)
}  // namespace allspark
#endif
