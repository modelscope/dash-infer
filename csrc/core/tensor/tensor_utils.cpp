/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tensor_utils.cpp
 */
#include <utility/cnpy.h>
#include <utility/mem_registry.h>
#include <weight/weight_loader.h>

#include "tensor.h"

#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <cuda/cuda_context.h>
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_FP16
#ifdef ENABLE_CUDA
#include <cuda_fp16.h>
#else
#include <common/float16.h>
#endif
#endif
#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

namespace allspark {
void TensorUtils::DeepCopyWhole(AsTensor& dst, AsTensor& src) {
  // for now, don't support convert by type
  // for data convert, use a special cast op
  //
  // data type should be same, shape should be same, mode should be same
  //
  if (dst.mode_ != src.mode_ && src.mode_ != DataMode::DENSE) {
    LOG(ERROR) << "not same mode: dst: " << (int)dst.mode_
               << " src: " << (int)src.mode_;
    throw std::invalid_argument(
        "deep copy require same mode, and mode should be dense.");
  }

  if (dst.shape_ != src.shape_) {
    LOG(ERROR) << "not same shape: dst: " << dst.shape_.ToString()
               << " src: " << src.shape_.ToString();
    throw std::invalid_argument("deep copy require same shape");
  }

  if (dst.dtype_ != src.dtype_) {
    LOG(ERROR) << "not same data type: dst: " << (int)src.dtype_
               << " src: " << (int)dst.dtype_;
    throw std::invalid_argument("deep copy require same data type");
  }

  if (!dst.data_ || !src.data_) {
    LOG(ERROR) << "data not exist, dst: " << (void*)dst.data_.get()
               << " src: " << (void*)src.data_.get();
    throw std::invalid_argument("copy without data storage");
  }
  DenseData* src_data = static_cast<DenseData*>(src.data_.get());
  DenseData* dst_data = static_cast<DenseData*>(dst.data_.get());
  int64_t nbytes = src_data->GetSize();
  if (nbytes == 0) {
    LOG(ERROR) << "copy with 0 bytes ignore byte request.";
    LOG(ERROR) << "src shape : " << src.shape_.ToString()
               << "dst shape: " << dst.shape_.ToString();
    print_backtrace();
    return;
  }

#ifdef ENABLE_CUDA
  if (src.backend_ == DeviceType::CUDA || dst.backend_ == DeviceType::CUDA) {
    // enter cuda copy area.
    // after pascal arch, gpu can auto infer the DeviceKind.
    // pinned memory should have a dst or src with cuda, so it will call
    // cuda memcpy
    AS_CHECK_CUDA(cudaMemcpy(dst.GetDataPtr(), src.GetDataPtr(), nbytes,
                             GetCudaMemcpyKind(src.backend_, dst.backend_)));
    return;
  }
#endif
  memcpy(dst.GetDataPtr(), src.GetDataPtr(), nbytes);
}

void TensorUtils::DeepCopyWholeAsync(AsTensor& dst, AsTensor& src,
                                     const DeviceContext* device_context) {
  // for now, don't support convert by type
  // for data convert, use a special cast op
  //
  // data type should be same, shape shoule be same, mode should be same
  //
  if (dst.mode_ != src.mode_ && src.mode_ != DataMode::DENSE) {
    LOG(ERROR) << "not same mode: dst: " << (int)dst.mode_
               << " src: " << (int)src.mode_;
    throw std::invalid_argument(
        "deep copy require same mode, and mode should be dense.");
  }

  if (dst.shape_ != src.shape_) {
    LOG(ERROR) << "not same shape: dst: " << dst.shape_.ToString()
               << " src: " << src.shape_.ToString();
    throw std::invalid_argument("deep copy require same shape");
  }

  if (dst.dtype_ != src.dtype_) {
    LOG(ERROR) << "not same data type: dst: " << (int)src.dtype_
               << " src: " << (int)dst.dtype_;
    print_backtrace();
    throw std::invalid_argument("deep copy require same data type");
  }

  if (!dst.data_ || !src.data_) {
    LOG(ERROR) << "data not exist, dst: " << (void*)dst.data_.get()
               << " src: " << (void*)src.data_.get();
    throw std::invalid_argument("copy without data storage");
  }
  DenseData* src_data = static_cast<DenseData*>(src.data_.get());
  DenseData* dst_data = static_cast<DenseData*>(dst.data_.get());
  int64_t nbytes = src_data->GetSize();
  if (nbytes == 0) {
    LOG(ERROR) << "copy with 0 bytes ignore byte request.";
    LOG(ERROR) << "src shape : " << src.shape_.ToString()
               << "dst shape: " << dst.shape_.ToString();
    print_backtrace();
    return;
  }

#ifdef ENABLE_CUDA
  if (src.backend_ == DeviceType::CUDA || dst.backend_ == DeviceType::CUDA) {
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device_context)->GetStream();
    AS_CHECK_CUDA(cudaMemcpyAsync(dst.GetDataPtr(), src.GetDataPtr(), nbytes,
                                  GetCudaMemcpyKind(src.backend_, dst.backend_),
                                  stream));
    return;
  }
#endif
  memcpy(dst.GetDataPtr(), src.GetDataPtr(), nbytes);
}

void TensorUtils::DeepCopyWholeTolerantAsync(
    AsTensor& dst, AsTensor& src, const DeviceContext* device_context) {
  // for now, don't support convert by type
  // for data convert, use a special cast op
  if (dst.mode_ != src.mode_ && src.mode_ != DataMode::DENSE) {
    LOG(ERROR) << "not same mode: dst: " << (int)dst.mode_
               << " src: " << (int)src.mode_;
    throw std::invalid_argument(
        "deep copy require same mode, and mode should be dense.");
  }

  if (dst.GetSizeInByte() < src.GetSizeInByte()) {
    LOG(ERROR) << "dst size too small: "
               << "dst dtype: " << (int)dst.dtype_ << ", "
               << "shape: " << dst.shape_.ToString() << ", "
               << "size in byte: " << dst.GetSizeInByte() << ", "
               << "src dtype: " << (int)src.dtype_ << ", "
               << "shape: " << src.shape_.ToString() << ", "
               << "size in byte: " << src.GetSizeInByte();
    throw std::invalid_argument("deep copy require dst has larger size");
  }

  if (!dst.data_ || !src.data_) {
    LOG(ERROR) << "data not exist, dst: " << (void*)dst.data_.get()
               << " src: " << (void*)src.data_.get();
    throw std::invalid_argument("copy without data storage");
  }
  DenseData* src_data = static_cast<DenseData*>(src.data_.get());
  DenseData* dst_data = static_cast<DenseData*>(dst.data_.get());
  int64_t nbytes = src_data->GetSize();
  if (nbytes == 0) {
    LOG(ERROR) << "copy with 0 bytes ignore byte request.";
    LOG(ERROR) << "src shape : " << src.shape_.ToString()
               << "dst shape: " << dst.shape_.ToString();
    print_backtrace();
    return;
  }

#ifdef ENABLE_CUDA
  if (src.backend_ == DeviceType::CUDA || dst.backend_ == DeviceType::CUDA) {
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device_context)->GetStream();
    AS_CHECK_CUDA(cudaMemcpyAsync(dst.GetDataPtr(), src.GetDataPtr(), nbytes,
                                  GetCudaMemcpyKind(src.backend_, dst.backend_),
                                  stream));
    return;
  }
#endif
  memcpy(dst.GetDataPtr(), src.GetDataPtr(), nbytes);
}

void TensorUtils::DeepCopyVector(AsTensor& dst, const AsTensor& src,
                                 size_t src_col_offset,
                                 const DeviceContext* ctx) {
  assert(dst.shape_.Count() <= src.shape_.Count());

  if (dst.shape_.Count() > src.shape_.Count()) {
    LOG(ERROR) << "DeepCopyVector: dst tensor is larger than src tensor: "
                  "dst tensor size: "
               << dst.shape_.Count() << " src size: " << src.shape_.Count();
    throw AsException("DeepCopyVector copy dst tensor larger than src tensor");
  }

  DeepCopyVectorPart(dst, 0, src, src_col_offset, dst.shape_[0], ctx);
};

void TensorUtils::DeepCopyVectorPart(AsTensor& dst, size_t dst_col_offset,
                                     const AsTensor& src, size_t src_col_offset,
                                     size_t len, const DeviceContext* ctx) {
  DeepCopyVectorPartAsync(dst, dst_col_offset, src, src_col_offset, len, ctx);
}

void TensorUtils::DeepCopyVectorPartAsync(AsTensor& dst, size_t dst_col_offset,
                                          const AsTensor& src,
                                          size_t src_col_offset, size_t len,
                                          const DeviceContext* device_context) {
  assert(dst.shape_.Size() == src.shape_.Size());
  assert(dst.dtype_ == src.dtype_);
  if ((dst.shape_.Size() != src.shape_.Size()) || dst.shape_.Size() != 1) {
    LOG(ERROR) << "dst.shape_.Size():" << (int)dst.shape_.Size()
               << ", src.shape_.Size(): " << (int)src.shape_.Size();
    throw AsException("DeepCopyVector only support 1d tensor");
  }
  if (dst.dtype_ != src.dtype_) {
    LOG(ERROR) << "DeepCopyVector with different type tensor: src:"
               << (int)src.GetDataType() << " dst: " << (int)dst.GetDataType();
    throw AsException("Copy with different type vector");
  }
  if (len + src_col_offset > src.shape_[0] ||
      len + dst_col_offset > dst.shape_[0]) {
    char buf[1024];
    sprintf(buf,
            "dst.shape[0]: %ld, dst_col_offset: %ld, "
            "src.shape[0]: %ld, src_col_offset: %ld, len:%ld",
            dst.shape_[0], dst_col_offset, src.shape_[0], src_col_offset, len);
    LOG(ERROR) << "DeepCopyVector copy tensor will beyoud src tensor size: "
               << buf;
    throw AsException(
        "DeepCopyVector copy tensor will beyoud src tensor size. ");
  }

  const void* src_ptr_with_offset =
      (const char*)src.GetDataPtr() +
      SizeofType(src.GetDataType()) * src_col_offset;
  void* dst_ptr_with_offset =
      (char*)dst.GetDataPtr() + SizeofType(dst.GetDataType()) * dst_col_offset;
  size_t copy_bytes = len * SizeofType(dst.GetDataType());

#ifdef ENABLE_CUDA
  if (src.GetDeviceType() == DeviceType::CUDA ||
      dst.GetDeviceType() == DeviceType::CUDA) {
    auto cpyKind = GetCudaMemcpyKind(src.GetDeviceType(), dst.GetDeviceType());
    if (cpyKind == cudaMemcpyHostToHost) {
      memcpy(dst_ptr_with_offset, src_ptr_with_offset, copy_bytes);
    } else {
      if (device_context != nullptr) {
        cudaStream_t stream =
            static_cast<const CUDAContext*>(device_context)->GetStream();
        AS_CHECK_CUDA(cudaMemcpyAsync(dst_ptr_with_offset, src_ptr_with_offset,
                                      copy_bytes, cpyKind, stream));
      } else {
        AS_CHECK_CUDA(cudaMemcpy(dst_ptr_with_offset, src_ptr_with_offset,
                                 copy_bytes, cpyKind));
      }
    }
  } else
#endif
  {
    memcpy(dst_ptr_with_offset, src_ptr_with_offset, copy_bytes);
  }
}

template <>
void TensorUtils::DeepCopyFromStdVector<void*>(AsTensor& dst,
                                               size_t dst_col_offset,
                                               const std::vector<void*>& src);

template <>
void TensorUtils::DeepCopyFromStdVector<float>(AsTensor& dst,
                                               size_t dst_col_offset,
                                               const std::vector<float>& src);
template <>
void TensorUtils::DeepCopyFromStdVector<int32_t>(
    AsTensor& dst, size_t dst_col_offset, const std::vector<int32_t>& src);
template <>
void TensorUtils::DeepCopyFromStdVector<int16_t>(
    AsTensor& dst, size_t dst_col_offset, const std::vector<int16_t>& src);
template <>
void TensorUtils::DeepCopyFromStdVector<int8_t>(AsTensor& dst,
                                                size_t dst_col_offset,
                                                const std::vector<int8_t>& src);
template <>
void TensorUtils::DeepCopyFromStdVector<uint8_t>(
    AsTensor& dst, size_t dst_col_offset, const std::vector<uint8_t>& src);

#ifdef ENABLE_FP16
template <>
void TensorUtils::DeepCopyFromStdVector<half>(AsTensor& dst,
                                              size_t dst_col_offset,
                                              const std::vector<half>& src);
#endif  // ENABLE_FP16
#ifdef ENABLE_BF16
template <>
void TensorUtils::DeepCopyFromStdVector<hie::bfloat16>(
    AsTensor& dst, size_t dst_col_offset,
    const std::vector<hie::bfloat16>& src);
#endif  // ENABLE_BF16

void TensorUtils::DeepCopyMatrix2D(AsTensor& dst, AsTensor& src,
                                   size_t src_col_offset, size_t src_row_offset,
                                   const DeviceContext* ctx) {
  assert(dst.shape_.Count() <= src.shape_.Count());
  // shape[1] cols , shape[0] rows

  if (dst.shape_.Count() > src.shape_.Count()) {
    LOG(ERROR) << "DeepCopyMatrix: dst tensor is larger than src tensor: "
                  "dst tensor size: "
               << dst.shape_.Count() << " src size: " << src.shape_.Count();
    throw AsException("DeepCopymatrix copy dst tensor larger than src tensor");
  }

  DeepCopyMatrix2DPart(dst, 0, 0, src, src_col_offset, src_row_offset,
                       dst.shape_[1], dst.shape_[0], ctx);
}

void TensorUtils::DeepCopyMatrix2DFromBatch(AsTensor& dst, AsTensor& src,
                                            size_t src_batch_idx,
                                            size_t src_col_offset,
                                            size_t src_row_offset,
                                            const DeviceContext* ctx) {
  assert(dst.shape_.Count() <= src.shape_.Count());
  // shape[1] cols , shape[0] rows

  if (dst.shape_.Count() > src.shape_.Count()) {
    LOG(ERROR)
        << "DeepCopyMatrix2DFromBatch: dst tensor is larger than src tensor: "
           "dst tensor size: "
        << dst.shape_.Count() << " src size: " << src.shape_.Count();
    throw AsException(
        "DeepCopymatrix2DFromBatch copy dst tensor larger than src tensor");
  }

  DeepCopyMatrix2DPartFromBatch(dst, 0, 0, src, src_batch_idx, src_col_offset,
                                src_row_offset, dst.shape_[1], dst.shape_[0],
                                ctx);
}

void TensorUtils::DeepCopyMatrix2DPart(
    AsTensor& dst, size_t dst_col_offset, size_t dst_row_offset, AsTensor& src,
    size_t src_col_offset, size_t src_row_offset, size_t region_width,
    size_t region_height, const DeviceContext* ctx) {
  // validation check for matrix copy.
  assert(dst.shape_.Size() == src.shape_.Size());
  assert(dst.shape_.Size() == 2);
  // shape[0] == row shape[1] = cols
  // we only access slice a smaller tensor matrix from source tensor.
  assert(region_height + src_row_offset <= src.shape_[0]);
  assert(region_width + src_col_offset <= src.shape_[1]);
  assert(region_height + dst_row_offset <= dst.shape_[0]);
  assert(region_width + dst_col_offset <= dst.shape_[1]);

  // shape[1] cols , shape[0] rows

  if ((dst.shape_.Size() != src.shape_.Size()) || dst.shape_.Size() != 2) {
    throw AsException("DeepCopyMatrix only support 2d tensor");
  }

  if (dst.dtype_ != src.dtype_) {
    LOG(ERROR) << "DeepCopyMatrix with different type tensor: src:"
               << (int)src.GetDataType() << " dst: " << (int)dst.GetDataType();
    throw AsException("Copy with different type vector");
  }
  if (region_height + src_row_offset > src.shape_[0] ||
      region_width + src_col_offset > src.shape_[1] ||
      region_height + dst_row_offset > dst.shape_[0] ||
      region_width + dst_col_offset > dst.shape_[1]) {
    char buf[1024];
    sprintf(buf,
            "region_height:%d region_width:%d src_row_offset:%d "
            "src_col_offset:%d dst_row_offset:%d dst_col_offset:%d "
            "src.shape(%d,%d) dst.shape(%d,%d)\n",
            region_height, region_width, src_row_offset, src_col_offset,
            dst_row_offset, dst_col_offset, src.shape_[0], src.shape_[1],
            dst.shape_[0], dst.shape_[1]);
    LOG(ERROR) << "DeepCopymatrix size not fit: " << buf;
    throw AsException(
        "DeepCopymatrix copy tensor will beyoud src tensor size. ");
  }

#ifdef ENABLE_CUDA
  if (src.GetDeviceType() == DeviceType::CUDA ||
      dst.GetDeviceType() == DeviceType::CUDA) {
    void* src_ptr_with_offsets =
        (char*)src.GetDataPtr() +
        ((src_row_offset * src.GetStrideInByte()) +
         src_col_offset * SizeofType(src.GetDataType()));

    void* dst_ptr_with_offsets =
        (char*)dst.GetDataPtr() + (dst_row_offset * dst.GetStrideInByte()) +
        (dst_col_offset * SizeofType(dst.GetDataType()));

    cudaMemcpyKind kind =
        GetCudaMemcpyKind(src.GetDeviceType(), dst.GetDeviceType());
    if (ctx)
      AS_CHECK_CUDA(cudaMemcpy2DAsync(
          dst_ptr_with_offsets, dst.GetStrideInByte(), src_ptr_with_offsets,
          src.GetStrideInByte(), region_width * SizeofType(dst.GetDataType()),
          region_height, kind,
          static_cast<const CUDAContext*>(ctx)->GetStream()));
    else
      AS_CHECK_CUDA(cudaMemcpy2D(dst_ptr_with_offsets, dst.GetStrideInByte(),
                                 src_ptr_with_offsets, src.GetStrideInByte(),
                                 region_width * SizeofType(dst.GetDataType()),
                                 region_height, kind));
  } else
#endif
  {
    size_t type_size = SizeofType(dst.GetDataType());
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < region_height; i++) {
      char* dst_offset_ptr = (char*)dst.GetDataPtr() +
                             (i + dst_row_offset) * dst.GetStrideInByte() +
                             (dst_col_offset * SizeofType(dst.GetDataType()));
      char* src_offset_ptr = (char*)src.GetDataPtr() +
                             (i + src_row_offset) * src.GetStrideInByte() +
                             src_col_offset * SizeofType(src.GetDataType());
      memcpy(dst_offset_ptr, src_offset_ptr, region_width * type_size);
    }
  }
}

void TensorUtils::DeepCopyMatrix2DPartFromBatch(
    AsTensor& dst, size_t dst_col_offset, size_t dst_row_offset, AsTensor& src,
    size_t src_batch_idx, size_t src_col_offset, size_t src_row_offset,
    size_t region_width, size_t region_height, const DeviceContext* ctx) {
  // validation check for matrix copy.
  assert(dst.shape_.Size() + 1 == src.shape_.Size());
  assert(dst.shape_.Size() == 2);
  // shape[0] == row shape[1] = cols
  // we only access slice a smaller tensor matrix from source tensor.
  assert(region_height + src_row_offset <= src.shape_[1]);
  assert(region_width + src_col_offset <= src.shape_[2]);
  assert(region_height + dst_row_offset <= dst.shape_[0]);
  assert(region_width + dst_col_offset <= dst.shape_[1]);

  // shape[1] cols , shape[0] rows

  if ((dst.shape_.Size() + 1 != src.shape_.Size()) || dst.shape_.Size() != 2) {
    throw AsException(
        "DeepCopyMatrixFromBatch only support src=3d & dst=2d tensor");
  }

  if (dst.dtype_ != src.dtype_) {
    LOG(ERROR) << "DeepCopyMatrix with different type tensor: src:"
               << (int)src.GetDataType() << " dst: " << (int)dst.GetDataType();
    throw AsException("Copy with different type vector");
  }
  if (region_height + src_row_offset > src.shape_[1] ||
      region_width + src_col_offset > src.shape_[2] ||
      region_height + dst_row_offset > dst.shape_[0] ||
      region_width + dst_col_offset > dst.shape_[1]) {
    char buf[1024];
    sprintf(buf,
            "region_height:%d region_width:%d src_row_offset:%d "
            "src_col_offset:%d dst_row_offset:%d dst_col_offset:%d "
            "src.shape(%ld,%ld) dst.shape(%ld,%ld)\n",
            region_height, region_width, src_row_offset, src_col_offset,
            dst_row_offset, dst_col_offset, src.shape_[1], src.shape_[2],
            dst.shape_[0], dst.shape_[1]);
    LOG(ERROR) << "DeepCopymatrix size not fit: " << buf;
    throw AsException(
        "DeepCopymatrix copy tensor will beyoud src tensor size. ");
  }

#ifdef ENABLE_CUDA
  if (src.GetDeviceType() == DeviceType::CUDA ||
      dst.GetDeviceType() == DeviceType::CUDA) {
    void* src_ptr_with_offsets =
        (char*)src.GetDataPtr() +
        src_batch_idx * src.GetShape().Count(1) *
            SizeofType(src.GetDataType()) +
        ((src_row_offset * src.GetShape().Count(2) *
          SizeofType(src.GetDataType())) +
         src_col_offset * SizeofType(src.GetDataType()));

    void* dst_ptr_with_offsets =
        (char*)dst.GetDataPtr() + (dst_row_offset * dst.GetStrideInByte()) +
        (dst_col_offset * SizeofType(dst.GetDataType()));

    cudaMemcpyKind kind =
        GetCudaMemcpyKind(src.GetDeviceType(), dst.GetDeviceType());
    if (ctx)
      AS_CHECK_CUDA(cudaMemcpy2DAsync(
          dst_ptr_with_offsets, dst.GetStrideInByte(), src_ptr_with_offsets,
          src.GetShape().Count(2) * SizeofType(src.GetDataType()),
          region_width * SizeofType(dst.GetDataType()), region_height, kind,
          static_cast<const CUDAContext*>(ctx)->GetStream()));
    else
      AS_CHECK_CUDA(cudaMemcpy2D(
          dst_ptr_with_offsets, dst.GetStrideInByte(), src_ptr_with_offsets,
          src.GetShape().Count(2) * SizeofType(src.GetDataType()),
          region_width * SizeofType(dst.GetDataType()), region_height, kind));
  } else
#endif
  {
    size_t type_size = SizeofType(dst.GetDataType());
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < region_height; i++) {
      char* dst_offset_ptr = (char*)dst.GetDataPtr() +
                             (i + dst_row_offset) * dst.GetStrideInByte() +
                             (dst_col_offset * SizeofType(dst.GetDataType()));
      char* src_offset_ptr = (char*)src.GetDataPtr() +
                             src_batch_idx * src.GetShape().Count(1) *
                                 SizeofType(src.GetDataType()) +
                             (i + src_row_offset) * src.GetShape().Count(2) *
                                 SizeofType(src.GetDataType()) +
                             src_col_offset * SizeofType(src.GetDataType());
      memcpy(dst_offset_ptr, src_offset_ptr, region_width * type_size);
    }
  }
}

void TensorUtils::Memset(AsTensor& t, char val) {
  if (t.GetDataPtr() == nullptr) return;

  if (t.GetDeviceType() == DeviceType::CPU) {
    if (t.shape_.Size() == 1) {
      memset(t.GetDataPtr(), val, t.GetSizeInByte());
    } else if (t.shape_.Size() == 2) {
#pragma omp parallel for num_threads(8)
      for (int i = 0; i < t.shape_[0]; i++) {
        memset((char*)t.GetDataPtr() + i * t.GetStrideInByte(), val,
               t.GetStrideInByte());
      }
    } else {
      assert(-1);
    }

  } else if (t.GetDeviceType() == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
    // 2023.8.8 xiaoteng: cudaMemset2D API crash in online env, disable it
    // temporarily
    AS_CHECK_CUDA(cudaMemset(t.GetDataPtr(), val, t.GetSizeInByte()));
#endif
  } else {
    assert(-1);
  }
}

void TensorUtils::ConcatMatrix2DColWise(
    AsTensor& batch_dst, int batch_idx,
    std::vector<std::shared_ptr<AsTensor>>& src_arr, const DeviceContext* ctx) {
#ifdef ENABLE_CUDA
  const auto& dst_shape = batch_dst.GetShape();
  assert(dst_shape.Size() == 3);
  AS_ENFORCE(batch_dst.GetDeviceType() == DeviceType::CUDA);
  auto dst_batch_stride =
      dst_shape.Count(1) * SizeofType(batch_dst.GetDataType());
  char* dst_ptr_with_offsets =
      (char*)batch_dst.GetDataPtr() + batch_idx * dst_batch_stride;
  auto dpitch = dst_shape.Count(2) * SizeofType(batch_dst.GetDataType());
  for (auto i = 0; i < src_arr.size(); i++) {
    auto& src = *src_arr.at(i);
    AS_ENFORCE(src.GetDeviceType() == DeviceType::CUDA);
    cudaMemcpyKind kind =
        GetCudaMemcpyKind(src.GetDeviceType(), batch_dst.GetDeviceType());
    const auto& src_shape = src.GetShape();
    AS_ENFORCE(src_shape.Size() == 2);
    auto region_height = src_shape[0];
    auto region_width_nbytes = src.GetStrideInByte();
    if (ctx)
      AS_CHECK_CUDA(cudaMemcpy2DAsync(
          dst_ptr_with_offsets, dpitch, src.GetDataPtr(), region_width_nbytes,
          region_width_nbytes, region_height, kind,
          static_cast<const CUDAContext*>(ctx)->GetStream()));
    else
      AS_CHECK_CUDA(cudaMemcpy2D(dst_ptr_with_offsets, dpitch, src.GetDataPtr(),
                                 region_width_nbytes, region_width_nbytes,
                                 region_height, kind));
    dst_ptr_with_offsets += region_width_nbytes;
  }
#else
  throw AsException(
      "Currently, TensorUtils::ConcatMatrix2DColWise is only supported between "
      "CUDAs");
#endif
}

void TensorUtils::ConcatMatrix2DColWiseBatched(
    AsTensor& dst, std::vector<std::shared_ptr<AsTensor>>& src_arr,
    const DeviceContext* ctx) {
#ifdef ENABLE_CUDA
  const auto& dst_shape = dst.GetShape();
  AS_ENFORCE(dst_shape.Size() == 3);
  AS_ENFORCE(dst.GetDeviceType() == DeviceType::CUDA);
  auto dst_batch_stride = dst_shape.Count(1) * SizeofType(dst.GetDataType());
  char* dst_ptr = (char*)dst.GetDataPtr();
  auto dpitch = dst_shape.Count(2) * SizeofType(dst.GetDataType());
  for (auto i = 0; i < src_arr.size(); i++) {
    auto& src = *src_arr.at(i);
    AS_ENFORCE(src.GetDeviceType() == DeviceType::CUDA);
    cudaMemcpyKind kind =
        GetCudaMemcpyKind(src.GetDeviceType(), dst.GetDeviceType());
    const auto& src_shape = src.GetShape();
    AS_ENFORCE(src_shape.Size() == 3);
    auto region_height = src_shape[0] * src_shape[1];
    auto region_width_nbytes = src.GetStrideInByte();
    if (ctx)
      AS_CHECK_CUDA(cudaMemcpy2DAsync(
          dst_ptr, dpitch, src.GetDataPtr(), region_width_nbytes,
          region_width_nbytes, region_height, kind,
          static_cast<const CUDAContext*>(ctx)->GetStream()));
    else
      AS_CHECK_CUDA(cudaMemcpy2D(dst_ptr, dpitch, src.GetDataPtr(),
                                 region_width_nbytes, region_width_nbytes,
                                 region_height, kind));
    dst_ptr += region_width_nbytes;
  }
#else
  throw AsException(
      "Currently, TensorUtils::ConcatMatrix2DColWise is only supported between "
      "CUDAs");
#endif
}

void TensorUtils::ConcatMatrix2DColWiseBatchedRawPtr(
    AsTensor& dst, std::vector<void*>& src_ptr_arr, DeviceType src_device_type,
    std::vector<Shape>& src_shapes, DataType src_dtype,
    const DeviceContext* ctx) {
#ifdef ENABLE_CUDA
  const auto& dst_shape = dst.GetShape();
  AS_ENFORCE(dst_shape.Size() == 3);
  AS_ENFORCE(dst.GetDeviceType() == DeviceType::CUDA);
  auto dst_batch_stride = dst_shape.Count(1) * SizeofType(dst.GetDataType());
  char* dst_ptr = (char*)dst.GetDataPtr();
  auto dpitch = dst_shape.Count(2) * SizeofType(dst.GetDataType());
  for (auto i = 0; i < src_ptr_arr.size(); i++) {
    void* src_ptr = src_ptr_arr.at(i);
    auto& src_shape = src_shapes.at(i);
    AS_ENFORCE(src_device_type == DeviceType::CUDA);
    cudaMemcpyKind kind =
        GetCudaMemcpyKind(src_device_type, dst.GetDeviceType());
    AS_ENFORCE(src_shape.Size() == 3);
    auto region_height = src_shape[0] * src_shape[1];
    auto region_width_nbytes = src_shape[-1] * SizeofType(src_dtype);
    if (ctx)
      AS_CHECK_CUDA(
          cudaMemcpy2DAsync(dst_ptr, dpitch, src_ptr, region_width_nbytes,
                            region_width_nbytes, region_height, kind,
                            static_cast<const CUDAContext*>(ctx)->GetStream()));
    else
      AS_CHECK_CUDA(cudaMemcpy2D(dst_ptr, dpitch, src_ptr, region_width_nbytes,
                                 region_width_nbytes, region_height, kind));
    dst_ptr += region_width_nbytes;
  }
#else
  throw AsException(
      "Currently, TensorUtils::ConcatMatrix2DColWise is only supported between "
      "CUDAs");
#endif
}

std::shared_ptr<TensorMap> TensorUtils::DeepCopyDLTensorMapToTensorMap(
    std::shared_ptr<DLTensorMap> in_map, const DeviceType target_device_type) {
  if (!in_map) return nullptr;

  auto ret = std::make_shared<TensorMap>();
  for (auto& t : *in_map) {
    ret->insert({t.first, std::make_shared<AsTensor>(t.first, t.second,
                                                     target_device_type)});
  }
  return ret;
}

std::shared_ptr<TensorMap> TensorUtils::DeepCopyDLTensorMapToTensorMap(
    std::shared_ptr<DLTensorMap> in_map) {
  if (!in_map) return nullptr;

  auto ret = std::make_shared<TensorMap>();
  for (auto& t : *in_map) {
    ret->insert({t.first, std::make_shared<AsTensor>(t.first, t.second)});
  }
  return ret;
}

std::shared_ptr<TensorListMap>
TensorUtils::DeepCopyDLTensorListMapToTensorListMap(
    std::shared_ptr<DLTensorListMap> in_map,
    const DeviceType target_device_type) {
  if (!in_map) return nullptr;

  auto ret = std::make_shared<TensorListMap>();
  for (const auto& t : *in_map) {
    auto dl_list = t.second;
    auto dl_name = t.first;
    std::vector<std::shared_ptr<AsTensor>> tensor_list;
    for (const auto* dltensor : dl_list) {
      auto as_tensor =
          std::make_shared<AsTensor>(t.first, dltensor, target_device_type);
      tensor_list.push_back(as_tensor);
    }
    ret->emplace(t.first, std::move(tensor_list));
  }
  return ret;
}

DeviceType TensorUtils::DLDeviceTypeToAsDeviceType(
    const DLDeviceType dltensor_device_type) {
  DeviceType as_tensor_device_type;
  switch (dltensor_device_type) {
    case kDLCPU:
      as_tensor_device_type = CPU;
      break;
#ifdef ENABLE_CUDA
    case kDLGPU:
      as_tensor_device_type = CUDA;
      break;
#endif
    default:
      LOG(ERROR) << "Unsupported DLDevice" << dltensor_device_type << std::endl;
      as_tensor_device_type = DEVICETYPE_UNDEFINED;
  }
  return as_tensor_device_type;
}

void TensorUtils::CopyMatrix2D(void* dst_ptr_with_offsets,
                               void* src_ptr_with_offsets,
                               DeviceType dst_device, DeviceType src_device,
                               size_t dst_stride, size_t src_stride,
                               size_t height, size_t width,
                               const DeviceContext* ctx) {
#ifdef ENABLE_CUDA
  if (src_device == DeviceType::CUDA || dst_device == DeviceType::CUDA) {
    cudaMemcpyKind kind = GetCudaMemcpyKind(src_device, dst_device);
    if (ctx) {
      AS_CHECK_CUDA(cudaMemcpy2DAsync(
          dst_ptr_with_offsets, dst_stride, src_ptr_with_offsets, src_stride,
          width, height, kind,
          static_cast<const CUDAContext*>(ctx)->GetStream()));
    }
    AS_CHECK_CUDA(cudaMemcpy2D(dst_ptr_with_offsets, dst_stride,
                               src_ptr_with_offsets, src_stride, width, height,
                               kind));
  } else
#endif
  {
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < height; i++) {
      char* dst_offset_ptr = (char*)dst_ptr_with_offsets + (i)*dst_stride;
      char* src_offset_ptr = (char*)src_ptr_with_offsets + (i)*src_stride;
      memcpy(dst_offset_ptr, src_offset_ptr, 1 * dst_stride);
    }
  }
}
}  // end namespace allspark
