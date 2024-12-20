/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    tensor.h
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/block.h"
#include "data.h"   // NOLINT
#include "shape.h"  // NOLINT

#ifdef ENABLE_CUDA
#include <driver_types.h>
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

class DeviceContext;
class TensorUtils;

void CopyData(void* dst_data, DeviceType dst_device, const void* src_data,
              DeviceType src_device, int64_t nbytes,
              const DeviceContext* device_context = nullptr);
// bitmask define and usage pls refer:
// https://github.com/oliora/bitmask/blob/master/README.md
//
//
enum class AsTensorFlags {
  empty_flag = 0x0,
  cuda_pinned_mem = (1 << 0),
};

class AsTensor {
 public:
  // FIXME: too many constructor, with lots of duplication, reduce to a
  // unified cons, with different wrapper.
  explicit AsTensor(
      const std::string& name = "", DeviceType backend = DeviceType::CPU,
      DataType dtype = DataType::DATATYPE_UNDEFINED,
      DataMode mode = DataMode::DENSE, const Shape& shape = {},
      int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag));

  explicit AsTensor(const TensorProto& tensor_proto,
                    DeviceType backend = DeviceType::CPU);

  AsTensor(const AsTensor& tensor) = default;

  explicit AsTensor(const std::string& name,
                    const DLManagedTensor* managed_dltensor);

  explicit AsTensor(const std::string& name,
                    const DLManagedTensor* managed_dltensor,
                    const DeviceType as_backend);

  explicit AsTensor(const std::string& name,
                    const std::vector<std::vector<int64_t>>& input,
                    const DeviceType as_backend = DeviceType::CUDA);
  explicit AsTensor(
      const std::string& name,
      const std::pair<std::vector<float>, std::vector<int64_t>>& input,
      const DeviceType as_backend = DeviceType::CUDA);

  explicit AsTensor(const AsTensor& src_tensor, DeviceType backend);

  explicit AsTensor(std::string new_name, const AsTensor& src_tensor);

  DLManagedTensor* ToDLPack(DeviceContext* device_ctx,
                            bool do_duplicate = false) const;
  // for debug only
  std::vector<char> ToNumpy(std::string save_file_path = "");

  const std::string& GetName() const;
  void SetName(const std::string& name);

  const Shape& GetShape() const;

  void SetMutable(bool is_mutable);

  bool GetMutable();

  DataType GetDataType() const;

  DeviceType GetDeviceType() const;

  DataMode GetDataMode() const;

  void* GetDataPtr() const;

  Data* GetData() const;

  std::string ToString() const;

  std::string ToStringAll() const;

  size_t GetSizeInByte() const;

  // get the stride for matrix copy like operations.
  // also it's the fundition to support alignment in row.
  size_t GetStrideInByte() const;

  AS_DEPRECATED(
      "use raw pointer copy without size, use TensorUtils::Copy Instead");
  void CopyDataFrom(const void* src_data, const size_t src_bytes,
                    DeviceType src_device,
                    const DeviceContext* device_ctx = nullptr);

  AS_DEPRECATED(
      "use raw pointer copy without size, use TensorUtils::Copy Instead");
  void CopyDataTo(void* dst_data, const size_t dst_bytes, DeviceType dst_device,
                  const DeviceContext* device_ctx = nullptr) const;

  AsStatus SetShape(Shape&& shape);

  AsStatus SetDataType(DataType dtype);

  // TODO: remove data mode change function.
  AS_DEPRECATED(
      "data mode should not changed during whole tensor lifetime, should "
      "init during constructor");
  AsStatus SetDataMode(DataMode mode);

  AsStatus SetData(std::shared_ptr<Data> data);

  // swap the underline _data ptr, it's require two tensor have same shape.
  void SwapData(AsTensor& rhs);

  // share data from two identical tensor.
  void ShareData(AsTensor& rhs);

  Block::Ptr GetBlock() const { return block_; }

  void BindingBlock(Block::Ptr block) {
    block_ = block;
    block_->BindTensor(this);
  }
  AsStatus Free();

  // for debug only
  std::string GetMD5Sum();

 private:
  std::string GetDataString() const;
  std::string GetDataStringAll() const;
  std::string name_;
  DeviceType backend_;
  DataType dtype_;
  DataMode mode_;
  Shape shape_;
  std::shared_ptr<Data> data_;
  Block::Ptr block_ = nullptr;
  int32_t flags_ = 0;
  bool mutable_ = true;

  friend class TensorUtils;
  void BuildFromDLTensor(const std::string& name,
                         const DLManagedTensor* managed_dltensor,
                         const DeviceType new_tensor_device_type);
};

inline std::ostream& operator<<(std::ostream& out, AsTensor const& data) {
  out << "AsTensor: name: " << data.GetName()
      << " dtype_: " << data.GetDataType() << " " << data.GetShape()
      << " data mode: " << data.GetDataMode()
      << " backend: " << data.GetDeviceType() << " ptr: " << data.GetDataPtr()
      << " data: " << data.GetData() << " block: " << data.GetBlock().get();
  return out;
}

/**
 *
 * Builder class for a new tensor, because AsTensor property will stay
unmutable,
 * for some tensor reuse, try build a new tensor.
 *
 * example :
 * \code
 * AsTensor tensor = AsTensorBuilder()
                    .SetName("my_tensor")
                    .SetBackend(DeviceType::CPU)
                    .SetDataType(DataType::FLOAT32)
                    .SetDataMode(DataMode::DENSE)
                    .SetShape({1, 2, 3})
                    .SetFlags(0)
                    .Build();
\endcode
 *
 */

class AsTensorBuilder {
 public:
  AsTensorBuilder& SetName(const std::string& name) {
    name_ = name;
    return *this;
  }
  AsTensorBuilder& SetBackend(DeviceType backend) {
    backend_ = backend;
    return *this;
  }
  AsTensorBuilder& SetDataType(DataType dtype) {
    dtype_ = dtype;
    return *this;
  }
  AsTensorBuilder& SetDataMode(DataMode mode) {
    mode_ = mode;
    return *this;
  }
  AsTensorBuilder& SetShape(const Shape& shape) {
    shape_ = shape;
    return *this;
  }
  AsTensorBuilder& SetFlags(int32_t flags) {
    flags_ = flags;
    return *this;
  }
  AsTensor Build() {
    return AsTensor(name_, backend_, dtype_, mode_, shape_, flags_);
  }

 private:
  std::string name_;
  DeviceType backend_ = DeviceType::CPU;
  DataType dtype_ = DataType::DATATYPE_UNDEFINED;
  DataMode mode_ = DataMode::DENSE;
  Shape shape_;
  int32_t flags_ = static_cast<int32_t>(AsTensorFlags::empty_flag);
};

using DLTensorMap = std::map<std::string, DLManagedTensor*>;
using DLTensorListMap = std::map<std::string, std::vector<DLManagedTensor*>>;
using TensorMap = std::unordered_map<std::string, std::shared_ptr<AsTensor>>;
using TensorListMap =
    std::unordered_map<std::string, std::vector<std::shared_ptr<AsTensor>>>;

#ifdef ENABLE_CUDA
cudaMemcpyKind GetCudaMemcpyKind(DeviceType src_dev_type,
                                 DeviceType dst_dev_type);
#endif

// template argument for if enable type auto convert.
class TensorUtils {
 public:
  /**
   * set a tensor to special byte val.
   * @param t operating tensor
   * @param val the byte value
   */
  static void Memset(AsTensor& t, char val);

  /**
   * Copy From one tensor to another tensor, full copy, sync copy.
   *
   * @param dst dest tensor of copy part.
   * @param src source tensor of copy part
   */
  static void DeepCopyWhole(AsTensor& dst, AsTensor& src);

  /**
   * Copy From one tensor to another tensor, full copy, async copy.
   *
   * @param dst dest tensor of copy part.
   * @param src source tensor of copy part
   */
  static void DeepCopyWholeAsync(AsTensor& dst, AsTensor& src,
                                 const DeviceContext* device_context);

  /**
   * Copy From one tensor to another tensor, full copy, async copy.
   *
   * DeepCopyWholeAsync is restrict, dst and src must have the same
   * data type and shape. While DeepCopyWholeTolerantAsync is tolerant,
   * it only ask for larger dst.
   *
   * @param dst dest tensor of copy part.
   * @param src source tensor of copy part
   */
  static void DeepCopyWholeTolerantAsync(AsTensor& dst, AsTensor& src,
                                         const DeviceContext* device_context);

  /**
   * Copy a matrix size in dst size from source matrix and offset by the
   *
   * @param dst dest tensor, the copy size is equal to dst tensor's size.
   * @param src_col_offset  the column offset in the source matrix, range: [0,
   * src.width - dst.widht]
   * @param src_row_offset  the rows offset in the source matrix, range: [0,
   * src.height - dst.height)
   * @param src source tensor of copy part
   * @param src_offset  the offset(note byte) in source tensor
   * @param copy_size  copy how many items(not bytes) in source tensor
   */
  static void DeepCopyMatrix2D(AsTensor& dst, AsTensor& src,
                               size_t src_col_offset, size_t src_row_offset,
                               const DeviceContext* ctx = nullptr);

  static void DeepCopyMatrix2DFromBatch(AsTensor& dst, AsTensor& src,
                                        size_t src_batch_idx,
                                        size_t src_col_offset,
                                        size_t src_row_offset,
                                        const DeviceContext* ctx = nullptr);

  /**
   * Copy Part of Src matrix to spcified area of dst matrix
   *
   * @param dst dest tensor of copy part.
   * @param dst_col_offset col offset in the dest matrix.
   * @param dst_row_offset row offset in the dst matrix.
   * @param src source tensor of this operation.
   * @param src_col_offset  the column offset in the source matrix,
   * @param src_row_offset  the rows offset in the source matrix,
   * @param region_width  operation region width
   * @param region_height operation region height
   */
  static void DeepCopyMatrix2DPart(AsTensor& dst, size_t dst_col_offset,
                                   size_t dst_row_offset, AsTensor& src,
                                   size_t src_col_offset, size_t src_row_offset,
                                   size_t region_width, size_t region_height,
                                   const DeviceContext* ctx = nullptr);

  static void DeepCopyMatrix2DPartFromBatch(
      AsTensor& dst, size_t dst_col_offset, size_t dst_row_offset,
      AsTensor& src, size_t src_batch_idx, size_t src_col_offset,
      size_t src_row_offset, size_t region_width, size_t region_height,
      const DeviceContext* ctx = nullptr);

  static void DeepCopyVector(AsTensor& dst, const AsTensor& src,
                             size_t src_col_offset,
                             const DeviceContext* ctx = nullptr);

  static void DeepCopyVectorPart(AsTensor& dst, size_t dst_col_offset,
                                 const AsTensor& src, size_t src_col_offset,
                                 size_t len,
                                 const DeviceContext* ctx = nullptr);
  static void ConcatMatrix2DColWise(
      AsTensor& batch_dst, int batch_idx,
      std::vector<std::shared_ptr<AsTensor>>& src_arr,
      const DeviceContext* ctx = nullptr);

  static void ConcatMatrix2DColWiseBatched(
      AsTensor& dst, std::vector<std::shared_ptr<AsTensor>>& src_arr,
      const DeviceContext* ctx = nullptr);

  static void ConcatMatrix2DColWiseBatchedRawPtr(
      AsTensor& dst, std::vector<void*>& src_ptr_arr,
      DeviceType src_device_type, std::vector<Shape>& src_shapes,
      DataType src_dtype, const DeviceContext* ctx = nullptr);

  static void DeepCopyVectorPartAsync(AsTensor& dst, size_t dst_col_offset,
                                      const AsTensor& src,
                                      size_t src_col_offset, size_t len,
                                      const DeviceContext* device_context);

  template <typename TYPE>
  static void DeepCopyFromStdVector(AsTensor& dst, size_t dst_col_offset,
                                    const std::vector<TYPE>& src) {
    constexpr auto dispatch_datatype = []<typename DT>() -> DataType {
      if constexpr (std::is_pointer<DT>::value) return DataType::POINTER;
      if constexpr (std::is_same_v<DT, float>) return DataType::FLOAT32;
      if constexpr (std::is_same_v<DT, int32_t>) return DataType::INT32;
      if constexpr (std::is_same_v<DT, int16_t>) return DataType::INT16;
      if constexpr (std::is_same_v<DT, int8_t>) return DataType::INT8;
      if constexpr (std::is_same_v<DT, uint8_t>) return DataType::UINT8;
#ifdef ENABLE_FP16
      if constexpr (std::is_same_v<DT, half>) return DataType::FLOAT16;
#endif  // ENABLE_BF16
#ifdef ENABLE_BF16
      if constexpr (std::is_same_v<DT, hie::bfloat16>)
        return DataType::BFLOAT16;
#endif  // ENABLE_BF16
      return DataType::DATATYPE_UNDEFINED;
    };

    auto dtype = dispatch_datatype.template operator()<TYPE>();
    if (dtype == DataType::DATATYPE_UNDEFINED) {
      LOG(WARNING) << "TensorUtils::DeepCopyFromStdVector: undefined data type";
    }

    if (src.size() > std::numeric_limits<int64_t>::max()) {
      LOG(ERROR) << "TensorUtils::DeepCopyFromStdVector: src vector size "
                    "exceeds int64_t";
      AS_THROW(AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR);
    }
    std::unique_ptr<AsTensor> vector2tensor = std::make_unique<AsTensor>(
        "anonymous_tensor_from_std::vector", DeviceType::CPU, dtype,
        DataMode::DENSE, Shape({static_cast<int64_t>(src.size())}));
    vector2tensor->CopyDataFrom(src.data(), src.size() * sizeof(TYPE),
                                DeviceType::CPU);
    DeepCopyVectorPart(dst, dst_col_offset, *vector2tensor, 0, src.size());
  }

  static std::shared_ptr<TensorMap> DeepCopyDLTensorMapToTensorMap(
      std::shared_ptr<DLTensorMap> in_map);

  static std::shared_ptr<TensorMap> DeepCopyDLTensorMapToTensorMap(
      std::shared_ptr<DLTensorMap> in_map, const DeviceType target_device_type);

  static std::shared_ptr<TensorListMap> DeepCopyDLTensorListMapToTensorListMap(
      std::shared_ptr<DLTensorListMap> in_map,
      const DeviceType target_device_type);

  static DeviceType DLDeviceTypeToAsDeviceType(
      const DLDeviceType dltensor_device_type);
  static void CopyMatrix2D(void* dst, void* src, DeviceType dst_device,
                           DeviceType src_device, size_t dst_stride,
                           size_t src_stride, size_t height, size_t width,
                           const DeviceContext* ctx = nullptr);
};

}  // namespace allspark
