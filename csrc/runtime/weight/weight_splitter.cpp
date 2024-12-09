/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_splitter.cpp
 */

#include "weight_loader.h"
namespace allspark {

// no split, copy or share the data from weight tensor to target tensor.
class WeightSplitterNoSplit : public WeightSplitter {
 public:
  WeightSplitterNoSplit(RankInfo rank_info)
      : WeightSplitter(SplitMode::NOSPLIT, rank_info){};

  bool IsSplittable(TensorInfo&) { return true; }

  bool IsModelProcessable(SplitMode mode) { return mode == SplitMode::NOSPLIT; }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
  }

  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // make sure have new shape, and swap data will do other check.
    // swap will change underline data block without copy.
    if (opt_whole_tensor) {
      if (dst_tensor->GetDeviceType() == opt_whole_tensor->GetDeviceType()) {
        dst_tensor->ShareData(*opt_whole_tensor);
      } else {
        std::shared_ptr<DenseData> data_dense = std::make_shared<DenseData>(
            dst_tensor->GetName(), len, dst_tensor->GetDeviceType());
        dst_tensor->SetData(data_dense);
        dst_tensor->CopyDataFrom(whole_weight_ptr, len, DeviceType::CPU);
      }
    } else {
      // no tensor provided, we have to copy it.
      std::shared_ptr<DenseData> data_dense = std::make_shared<DenseData>(
          dst_tensor->GetName(), len, dst_tensor->GetDeviceType());
      dst_tensor->SetData(data_dense);
      dst_tensor->CopyDataFrom(whole_weight_ptr, len, DeviceType::CPU);
    }
  }
};

// split a tensor in column dim
// eg, split with 2 rank
//   _______        _______
//  |      |       |   |   |
//  |      |       |   |   |
//  |      |       | a | b |
//  |      |  x    |   |   |
//  |      |       |   |   |
//  |______|       |___|___|
//  Activation       Weight
class WeightSplitterVerticalSplit : public WeightSplitter {
 public:
  WeightSplitterVerticalSplit(RankInfo rank_info)
      : WeightSplitter(SplitMode::VSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.shape.Size() == 2) {  // it's the weight matrix.
      if (info.shape[1] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: vsplit: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else if (info.shape.Size() == 1) {  // it's bias
      if (info.shape[0] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: vsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else {
      LOG(ERROR) << " weight split: vsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }

    return true;
  }

  bool IsModelProcessable(SplitMode mode) { return mode == SplitMode::VSPLIT; }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 2) {
      new_shape[1] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 1) {        // bias
      new_shape[0] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));
    }
  }
  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 2) {
        int rank_offset = (dst_tensor_info.shape[1] / rank_info_.rank_size) *
                          rank_info_.rank_id;
        TensorUtils::DeepCopyMatrix2D(*dst_tensor, *opt_whole_tensor,
                                      rank_offset, 0);
      } else if (new_shape.Size() == 1) {  // bias
        int rank_offset =
            (new_shape[0] / rank_info_.rank_size) * rank_info_.rank_id;
        TensorUtils::DeepCopyVector(*dst_tensor, *opt_whole_tensor,
                                    rank_offset);
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};
class WeightSplitterBatchVerticalSplit : public WeightSplitter {
 public:
  WeightSplitterBatchVerticalSplit(RankInfo rank_info)
      : WeightSplitter(SplitMode::BATCH_VSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.shape.Size() == 3) {  // it's the weight matrix.
      if (info.shape[2] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: batchvsplit: rank: "
                   << " tensor shape[2]: " << info.shape[2]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else {
      if (info.shape.Size() == 2) {  // it's the scale or zero matrix.
        if (info.shape[1] % rank_info_.rank_size != 0) {
          LOG(ERROR) << " weight split: batchvsplit: rank: "
                     << " tensor shape[1]: " << info.shape[1]
                     << " cannot div by nrank: " << rank_info_.rank_size;
          return false;
        }
      }
      LOG(ERROR) << " weight split: batchvsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }

    return true;
  }

  bool IsModelProcessable(SplitMode mode) {
    return mode == SplitMode::BATCH_VSPLIT;
  }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 3) {
      new_shape[2] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 2) {
      new_shape[1] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    }
  }
  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 3) {
        int batch = new_shape[0];
        int sub_matrix_width = new_shape[2];
        int sub_matrix_height = new_shape[1];
        for (int sub_gemm_idx = 0; sub_gemm_idx < batch; sub_gemm_idx++) {
          size_t sub_matrix_width_after_div_rank =
              sub_matrix_width / rank_info_.rank_size;
          size_t base_src_idx =
              sub_gemm_idx * sub_matrix_height * sub_matrix_width +
              sub_matrix_width_after_div_rank * rank_info_.rank_id;
          size_t base_dst_idx = sub_gemm_idx * sub_matrix_height *
                                sub_matrix_width_after_div_rank;
          AsTensor* dst = dst_tensor.get();
          AsTensor* src = opt_whole_tensor.get();
          size_t data_size = SizeofType(src->GetDataType());
          void* src_ptr_with_offsets =
              (char*)src->GetDataPtr() + base_src_idx * data_size;
          void* dst_ptr_with_offsets =
              (char*)dst->GetDataPtr() + base_dst_idx * data_size;
          TensorUtils::CopyMatrix2D(dst_ptr_with_offsets, src_ptr_with_offsets,
                                    dst->GetDeviceType(), src->GetDeviceType(),
                                    dst->GetStrideInByte(),
                                    src->GetStrideInByte(), sub_matrix_height,
                                    dst->GetStrideInByte());
        }
      } else if (new_shape.Size() == 2) {
        int batch = new_shape[0];
        int sub_matrix_width = new_shape[1];
        int sub_matrix_height = 1;
        for (int sub_gemm_idx = 0; sub_gemm_idx < batch; sub_gemm_idx++) {
          size_t sub_matrix_width_after_div_rank =
              sub_matrix_width / rank_info_.rank_size;
          size_t base_src_idx =
              sub_gemm_idx * sub_matrix_height * sub_matrix_width +
              sub_matrix_width_after_div_rank * rank_info_.rank_id;
          size_t base_dst_idx = sub_gemm_idx * sub_matrix_height *
                                sub_matrix_width_after_div_rank;
          AsTensor* dst = dst_tensor.get();
          AsTensor* src = opt_whole_tensor.get();
          size_t data_size = SizeofType(src->GetDataType());
          void* src_ptr_with_offsets =
              (char*)src->GetDataPtr() + base_src_idx * data_size;
          void* dst_ptr_with_offsets =
              (char*)dst->GetDataPtr() + base_dst_idx * data_size;
          TensorUtils::CopyMatrix2D(dst_ptr_with_offsets, src_ptr_with_offsets,
                                    dst->GetDeviceType(), src->GetDeviceType(),
                                    dst->GetStrideInByte(),
                                    src->GetStrideInByte(), sub_matrix_height,
                                    dst->GetStrideInByte());
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};
class WeightSplitterBatchKVerticalSplit : public WeightSplitter {
 public:
  WeightSplitterBatchKVerticalSplit(RankInfo rank_info)
      : WeightSplitter(SplitMode::BATCH_VSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.shape.Size() == 3) {  // it's the weight matrix.
      if (info.shape[2] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: BATCH_KVSPLIT: rank: "
                   << " tensor shape[2]: " << info.shape[2]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else if (info.shape.Size() == 2) {  // it's the scale or zero matrix.
      if (info.shape[1] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: BATCH_KVSPLIT: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else {
      LOG(ERROR)
          << " weight split: BATCH_KVSPLIT: try to div higger dim matrix "
          << info.shape.Size();
      return false;
    }

    return true;
  }

  bool IsModelProcessable(SplitMode mode) {
    return mode == SplitMode::BATCH_KVSPLIT;
  }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 3) {
      new_shape[2] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 2) {
      new_shape[1] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    }
  }
  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 3) {
        int batch = new_shape[0];
        int batch_gemm_cnt = 2;
        int sub_matrix_width = new_shape[2] / batch_gemm_cnt;
        int sub_matrix_height = new_shape[1];
        for (int batch_gemm_idx = 0; batch_gemm_idx < batch; batch_gemm_idx++) {
          for (int sub_gemm_idx = 0; sub_gemm_idx < batch_gemm_cnt;
               sub_gemm_idx++) {
            size_t sub_matrix_width_after_div_rank =
                sub_matrix_width / rank_info_.rank_size;
            size_t base_src_idx =
                batch_gemm_idx * sub_matrix_height * sub_matrix_width *
                    batch_gemm_cnt +
                sub_gemm_idx * sub_matrix_width +
                sub_matrix_width_after_div_rank * rank_info_.rank_id;
            size_t base_dst_idx =
                batch_gemm_idx * sub_matrix_height *
                    sub_matrix_width_after_div_rank * batch_gemm_cnt +
                sub_gemm_idx * sub_matrix_width_after_div_rank;
            AsTensor* dst = dst_tensor.get();
            AsTensor* src = opt_whole_tensor.get();
            size_t data_size = SizeofType(src->GetDataType());
            void* src_ptr_with_offsets =
                (char*)src->GetDataPtr() + base_src_idx * data_size;
            void* dst_ptr_with_offsets =
                (char*)dst->GetDataPtr() + base_dst_idx * data_size;
            TensorUtils::CopyMatrix2D(
                dst_ptr_with_offsets, src_ptr_with_offsets,
                dst->GetDeviceType(), src->GetDeviceType(),
                dst->GetStrideInByte(), src->GetStrideInByte(),
                sub_matrix_height, dst->GetStrideInByte() / batch_gemm_cnt);
          }
        }
      } else if (new_shape.Size() == 2) {
        int batch = new_shape[0];
        int batch_gemm_cnt = 2;
        int sub_matrix_width = new_shape[1] / batch_gemm_cnt;
        int sub_matrix_height = 1;
        for (int batch_gemm_idx = 0; batch_gemm_idx < batch; batch_gemm_idx++) {
          for (int sub_gemm_idx = 0; sub_gemm_idx < batch_gemm_cnt;
               sub_gemm_idx++) {
            size_t sub_matrix_width_after_div_rank =
                sub_matrix_width / rank_info_.rank_size;
            size_t base_src_idx =
                batch_gemm_idx * sub_matrix_height * sub_matrix_width *
                    batch_gemm_cnt +
                sub_gemm_idx * sub_matrix_width +
                sub_matrix_width_after_div_rank * rank_info_.rank_id;
            size_t base_dst_idx =
                batch_gemm_idx * sub_matrix_height *
                    sub_matrix_width_after_div_rank * batch_gemm_cnt +
                sub_gemm_idx * sub_matrix_width_after_div_rank;
            AsTensor* dst = dst_tensor.get();
            AsTensor* src = opt_whole_tensor.get();
            size_t data_size = SizeofType(src->GetDataType());
            void* src_ptr_with_offsets =
                (char*)src->GetDataPtr() + base_src_idx * data_size;
            void* dst_ptr_with_offsets =
                (char*)dst->GetDataPtr() + base_dst_idx * data_size;
            TensorUtils::CopyMatrix2D(
                dst_ptr_with_offsets, src_ptr_with_offsets,
                dst->GetDeviceType(), src->GetDeviceType(),
                dst->GetStrideInByte(), src->GetStrideInByte(),
                sub_matrix_height, dst->GetStrideInByte() / batch_gemm_cnt);
          }
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};
class WeightSplitterHorizontalSplit : public WeightSplitter {
 public:
  WeightSplitterHorizontalSplit(RankInfo rank_info)
      : WeightSplitter(SplitMode::HSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.shape.Size() == 2) {  // it's the weight matrix.
      if (info.shape[0] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: hsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else if (info.shape.Size() == 1) {  // it's bias
      if (info.shape[0] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: hsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else {
      LOG(ERROR) << " weight split: hsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }
    return true;
  }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 2) {
      new_shape[0] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 1) {        // bias
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));
    }
  }

  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 2) {
        int rows = rank_info_.rank_id * new_shape[0] / rank_info_.rank_size;
        // maybe not strink memory.
        TensorUtils::DeepCopyMatrix2D(*dst_tensor, *opt_whole_tensor, 0, rows);
      } else if (new_shape.Size() == 1) {  // bias
        // In this mode, the bias will be calculated rank size times.
        // Only keep the bias with actual value in rank=0, while setting
        // other ranks to 0 to avoid adding bias too many times.
        if (rank_info_.rank_id == 0) {
          TensorUtils::DeepCopyVector(*dst_tensor, *opt_whole_tensor, 0);
        } else {
          TensorUtils::Memset(*dst_tensor, 0);
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }

  bool IsModelProcessable(SplitMode mode) { return mode == SplitMode::HSPLIT; }
};
class WeightSplitterBatchHorizontalSplit : public WeightSplitter {
 public:
  WeightSplitterBatchHorizontalSplit(RankInfo rank_info)
      : WeightSplitter(SplitMode::BATCH_HSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.shape.Size() == 3) {  // it's the weight matrix.
      if (info.shape[1] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: batchhsplit: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else {
      LOG(ERROR) << " weight split: batchhsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }

    return true;
  }

  bool IsModelProcessable(SplitMode mode) {
    return mode == SplitMode::BATCH_HSPLIT;
  }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 3) {
      new_shape[1] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    }
  }
  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 3) {
        int batch = new_shape[0];
        int sub_matrix_width = new_shape[2];
        int sub_matrix_height = new_shape[1];

        for (int sub_gemm_idx = 0; sub_gemm_idx < batch; sub_gemm_idx++) {
          size_t dst_height = sub_matrix_height / rank_info_.rank_size;
          size_t base_src_idx =
              sub_gemm_idx * sub_matrix_height * sub_matrix_width +
              rank_info_.rank_id * dst_height * sub_matrix_width;
          size_t base_dst_idx = sub_gemm_idx * dst_height * sub_matrix_width;
          AsTensor* dst = dst_tensor.get();
          AsTensor* src = opt_whole_tensor.get();
          size_t data_size = SizeofType(src->GetDataType());
          void* src_ptr_with_offsets =
              (char*)src->GetDataPtr() + base_src_idx * data_size;
          void* dst_ptr_with_offsets =
              (char*)dst->GetDataPtr() + base_dst_idx * data_size;
          TensorUtils::CopyMatrix2D(
              dst_ptr_with_offsets, src_ptr_with_offsets, dst->GetDeviceType(),
              src->GetDeviceType(), dst->GetStrideInByte(),
              src->GetStrideInByte(), dst_height, dst->GetStrideInByte());
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};
// Batch Gemm version weight splitter, the main use case is qkv, and kv weight
// split.
template <int batch_gemm_cnt, SplitMode split_mode>
class WeightSplitterVSplitBatchGEMM : public WeightSplitter {
 public:
  WeightSplitterVSplitBatchGEMM(RankInfo rank_info)
      : WeightSplitter(split_mode, rank_info) {}

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.shape.Size() == 2) {  // it's the weight matrix.
      if (info.shape[1] % (batch_gemm_cnt * rank_info_.rank_size) != 0) {
        LOG(ERROR) << " weight split: batch vsplit: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " batch gemm cnt: " << batch_gemm_cnt
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else if (info.shape.Size() == 1) {  // it's bias
      if (info.shape[0] % (rank_info_.rank_size * batch_gemm_cnt) != 0) {
        LOG(ERROR) << " weight split: batch vsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
    } else {
      LOG(ERROR) << " weight split: hsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }

    return true;
  }

  // to support visiter pattern
  bool IsModelProcessable(SplitMode mode) { return mode == split_mode; }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 2) {
      new_shape[1] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 1) {        // bias
      new_shape[0] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));
    }
  }

  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 2) {
        int sub_matrix_width = new_shape[1] / batch_gemm_cnt;
        int sub_matrix_height = new_shape[0];

        for (int sub_gemm_idx = 0; sub_gemm_idx < batch_gemm_cnt;
             sub_gemm_idx++) {
          size_t sub_matrix_width_after_div_rank =
              sub_matrix_width / rank_info_.rank_size;
          size_t base_src_idx = sub_gemm_idx * sub_matrix_width;
          size_t base_dst_idx =
              (sub_gemm_idx * sub_matrix_width) / rank_info_.rank_size;
          TensorUtils::DeepCopyMatrix2DPart(
              *dst_tensor, base_dst_idx, 0, *opt_whole_tensor,
              base_src_idx +
                  sub_matrix_width_after_div_rank * rank_info_.rank_id,
              0, sub_matrix_width_after_div_rank, sub_matrix_height);
        }
      } else if (new_shape.Size() == 1) {  // bias
        int qkv_sub_width = new_shape[0] / batch_gemm_cnt;
        int copy_len = (new_shape[0] / rank_info_.rank_size) / batch_gemm_cnt;
        for (int sub_vec_idx = 0; sub_vec_idx < batch_gemm_cnt; sub_vec_idx++) {
          size_t src_idx =
              qkv_sub_width * sub_vec_idx +
              (qkv_sub_width / rank_info_.rank_size) * rank_info_.rank_id;
          size_t dst_idx = sub_vec_idx * (qkv_sub_width / rank_info_.rank_size);

          TensorUtils::DeepCopyVectorPart(*dst_tensor, dst_idx,
                                          *opt_whole_tensor, src_idx, copy_len);
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};

template class WeightSplitterVSplitBatchGEMM<3, SplitMode::QKVSPLIT>;
template class WeightSplitterVSplitBatchGEMM<2, SplitMode::KVSPLIT>;

class WeightSplitterVSplitGroupList : public WeightSplitter {
 public:
  WeightSplitterVSplitGroupList(RankInfo rank_info)
      : WeightSplitter(SplitMode::GROUP_VSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    int total_len = 0;
    for (int i = 0; i < info.group_list.size(); i++) {
      if (info.group_list[i] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: group_vsplit[" << i
                   << "] : " << info.group_list[i]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
      total_len += info.group_list[i];
    }
    if (info.shape.Size() == 2) {  // it's the weight matrix.
      if (info.shape[1] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: group_vsplit: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
      if (info.shape[1] != total_len) {
        LOG(ERROR) << " weight split: group_vsplit: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " not equal to gourp_list: " << total_len;
        return false;
      }
    } else if (info.shape.Size() == 1) {  // it's bias
      if (info.shape[0] % rank_info_.rank_size != 0) {
        LOG(ERROR) << " weight split: group_vsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " cannot div by nrank: " << rank_info_.rank_size;
        return false;
      }
      if (info.shape[0] != total_len) {
        LOG(ERROR) << " weight split: group_vsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " not equal to gourp_list: " << total_len;
        return false;
      }
    } else {
      LOG(ERROR) << " weight split: group_vsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }

    return true;
  }

  bool IsModelProcessable(SplitMode mode) {
    return mode == SplitMode::GROUP_VSPLIT;
  }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);
    if (new_shape.Size() == 2) {
      new_shape[1] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 1) {        // bias
      new_shape[0] /= rank_info_.rank_size;
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));
    }
  }

  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 2) {
        int sub_matrix_height = new_shape[0];
        int prefix_sum = 0;
        for (int i = 0; i < dst_tensor_info.group_list.size(); i++) {
          size_t now = dst_tensor_info.group_list[i];
          size_t sub_matrix_width_after_div_rank = now / rank_info_.rank_size;
          size_t base_src_idx = prefix_sum;
          size_t base_dst_idx = prefix_sum / rank_info_.rank_size;
          TensorUtils::DeepCopyMatrix2DPart(
              *dst_tensor, base_dst_idx, 0, *opt_whole_tensor,
              base_src_idx +
                  sub_matrix_width_after_div_rank * rank_info_.rank_id,
              0, sub_matrix_width_after_div_rank, sub_matrix_height);
          prefix_sum += now;
        }
      } else if (new_shape.Size() == 1) {  // bias
        int prefix_sum = 0;
        for (int i = 0; i < dst_tensor_info.group_list.size(); i++) {
          size_t now = dst_tensor_info.group_list[i];
          int copy_len = now / rank_info_.rank_size;
          size_t src_idx =
              prefix_sum + (now / rank_info_.rank_size) * rank_info_.rank_id;
          size_t dst_idx = prefix_sum / rank_info_.rank_size;
          TensorUtils::DeepCopyVectorPart(*dst_tensor, dst_idx,
                                          *opt_whole_tensor, src_idx, copy_len);
          prefix_sum += now;
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};

class WeightSplitterVSplitMQA : public WeightSplitter {
 public:
  WeightSplitterVSplitMQA(RankInfo rank_info)
      : WeightSplitter(SplitMode::MQA_VSPLIT, rank_info){};

  bool IsSplittable(TensorInfo& info) {
    // check the outer dim can div by nrank or not.
    if (info.group_list.size() != 3) {
      LOG(ERROR) << " weight split: MQA_vsplit only support 3 size list ";
      return false;
    }
    if (info.group_list[0] % rank_info_.rank_size != 0) {
      LOG(ERROR) << " weight split: MQA_vsplit[" << 0
                 << "] : " << info.group_list[0]
                 << " cannot div by nrank: " << rank_info_.rank_size;
      return false;
    }
    int total_len = 0;
    for (int i = 0; i < info.group_list.size(); i++) {
      total_len += info.group_list[i];
    }
    if (info.shape.Size() == 2) {  // it's the weight matrix.
      if (info.shape[1] != total_len) {
        LOG(ERROR) << " weight split: MQA_vsplit: rank: "
                   << " tensor shape[1]: " << info.shape[1]
                   << " not equal to gourp_list: " << total_len;
        return false;
      }
    } else if (info.shape.Size() == 1) {  // it's bias
      if (info.shape[0] != total_len) {
        LOG(ERROR) << " weight split: MQA_vsplit: rank: "
                   << " tensor shape[0]: " << info.shape[0]
                   << " not equal to gourp_list: " << total_len;
        return false;
      }
    } else {
      LOG(ERROR) << " weight split: MQA_vsplit: try to div higger dim matrix "
                 << info.shape.Size();
      return false;
    }
    return true;
  }

  bool IsModelProcessable(SplitMode mode) {
    return mode == SplitMode::MQA_VSPLIT;
  }

  void SetShape(TensorInfo dst_tensor_info,
                std::shared_ptr<AsTensor> dst_tensor) {
    Shape new_shape(dst_tensor_info.shape);

    if (new_shape.Size() == 2) {
      new_shape[1] = dst_tensor_info.group_list[0] / rank_info_.rank_size +
                     dst_tensor_info.group_list[1] +
                     dst_tensor_info.group_list[2];
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));  // set new shape will allocate.
    } else if (new_shape.Size() == 1) {        // bias
      new_shape[0] = dst_tensor_info.group_list[0] / rank_info_.rank_size +
                     dst_tensor_info.group_list[1] +
                     dst_tensor_info.group_list[2];
      dst_tensor->Free();
      dst_tensor->SetShape(Shape(new_shape));
    }
  }

  void CopyWeight(TensorInfo dst_tensor_info,
                  std::shared_ptr<AsTensor> dst_tensor,
                  std::shared_ptr<AsTensor> opt_whole_tensor,
                  const void* whole_weight_ptr, size_t len) {
    // offset, not by bytes.
    if (opt_whole_tensor) {
      Shape new_shape(dst_tensor_info.shape);
      if (new_shape.Size() == 2) {
        int sub_matrix_height = new_shape[0];
        int src_prefix_sum = 0;
        int dst_prefix_sum = 0;
        for (int i = 0; i < dst_tensor_info.group_list.size(); i++) {
          if (i == 0) {
            size_t now = dst_tensor_info.group_list[i] / rank_info_.rank_size;
            size_t src_idx = src_prefix_sum + (now)*rank_info_.rank_id;
            size_t dst_idx = dst_prefix_sum;
            TensorUtils::DeepCopyMatrix2DPart(*dst_tensor, dst_idx, 0,
                                              *opt_whole_tensor, src_idx, 0,
                                              now, sub_matrix_height);
            src_prefix_sum += dst_tensor_info.group_list[i];
            dst_prefix_sum += now;
          } else {
            size_t now = dst_tensor_info.group_list[i];
            size_t src_idx = src_prefix_sum;
            size_t dst_idx = dst_prefix_sum;
            TensorUtils::DeepCopyMatrix2DPart(*dst_tensor, dst_idx, 0,
                                              *opt_whole_tensor, src_idx, 0,
                                              now, sub_matrix_height);
            src_prefix_sum += dst_tensor_info.group_list[i];
            dst_prefix_sum += now;
          }
        }
      } else if (new_shape.Size() == 1) {  // bias
        int src_prefix_sum = 0;
        int dst_prefix_sum = 0;
        for (int i = 0; i < dst_tensor_info.group_list.size(); i++) {
          if (i == 0) {
            size_t now = dst_tensor_info.group_list[i] / rank_info_.rank_size;
            int copy_len = now;
            size_t src_idx = src_prefix_sum + (now)*rank_info_.rank_id;
            size_t dst_idx = dst_prefix_sum;
            TensorUtils::DeepCopyVectorPart(
                *dst_tensor, dst_idx, *opt_whole_tensor, src_idx, copy_len);

            src_prefix_sum += dst_tensor_info.group_list[i];
            dst_prefix_sum += now;
          } else {
            size_t now = dst_tensor_info.group_list[i];
            int copy_len = now;
            size_t src_idx = src_prefix_sum;
            size_t dst_idx = dst_prefix_sum;
            TensorUtils::DeepCopyVectorPart(
                *dst_tensor, dst_idx, *opt_whole_tensor, src_idx, copy_len);
            src_prefix_sum += dst_tensor_info.group_list[i];
            dst_prefix_sum += now;
          }
        }
      }
    } else {
      assert(-1);
      // TODO: impl later.
    }
  }
};

// TODO: change to regsiter pattern if there is more split mode comming
std::unique_ptr<WeightSplitter> WeightSplitterFactory::GetSplitterByMode(
    SplitMode mode, RankInfo rankInfo) {
  std::unique_ptr<WeightSplitter> splitter;

  switch (mode) {
    case SplitMode::NOSPLIT:
      splitter.reset(new WeightSplitterNoSplit(rankInfo));
      break;
    case SplitMode::VSPLIT:
      splitter.reset(new WeightSplitterVerticalSplit(rankInfo));
      break;

    case SplitMode::HSPLIT:
      splitter.reset(new WeightSplitterHorizontalSplit(rankInfo));
      break;
    case SplitMode::QKVSPLIT:
      splitter.reset(
          new WeightSplitterVSplitBatchGEMM<3, SplitMode::QKVSPLIT>(rankInfo));
      break;
    case SplitMode::KVSPLIT:
      splitter.reset(
          new WeightSplitterVSplitBatchGEMM<2, SplitMode::KVSPLIT>(rankInfo));
      break;
    case SplitMode::GROUP_VSPLIT:
      splitter.reset(new WeightSplitterVSplitGroupList(rankInfo));
      break;
    case SplitMode::MQA_VSPLIT:
      splitter.reset(new WeightSplitterVSplitMQA(rankInfo));
      break;
    case SplitMode::BATCH_VSPLIT:
      splitter.reset(new WeightSplitterBatchVerticalSplit(rankInfo));
      break;
    case SplitMode::BATCH_HSPLIT:
      splitter.reset(new WeightSplitterBatchHorizontalSplit(rankInfo));
      break;
    case SplitMode::BATCH_KVSPLIT:
      splitter.reset(new WeightSplitterBatchKVerticalSplit(rankInfo));
      break;
    default:
      assert(-1);
      throw AsException("unsupported split mode.");
  }

  return splitter;
}

}  // namespace allspark
