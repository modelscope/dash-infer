/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_loader.cpp
 */
#include "weight_loader.h"
#ifdef ENABLE_FP16
#include <common/float16.h>
#endif

#include "string_util.h"

namespace allspark {

WeightFileParser::WeightFileParser() {}

// caller should make sure the ptr[len-1] = '\0';
TensorInfo WeightFileParser::ParseTensorInfo(const void* ptr, size_t len) {
  TensorInfo info;
  DataType& dtype = info.dtype;
  Shape& shape = info.shape;
  DataMode& mode = info.mode;
  int& nnz = info.nnz;
  SplitMode& split_mode = info.split_mode;
  std::vector<int>& group_list = info.group_list;
  void* tmp = malloc(len + 1);
  snprintf((char*)tmp, len, "%s", (char*)ptr);
  std::string header = (char*)tmp;
  free(tmp);

  size_t loc1, loc2;
  // endian, word size, data type
  // byte order code | stands for not applicable.
  // not sure when this applies except for byte array
  loc1 = header.find("descr");
  if (loc1 == std::string::npos) {
    LOG(ERROR)
        << "parse_allsparky_header: failed to find header keyword: 'descr'"
        << std::endl;
    throw AsException("weight header parse failed: header [1]");
  }
  loc1 += 9;
  if (header[loc1] != '<') {
    LOG(ERROR) << "parse_allsparky_header: LittleEndianTest fail" << std::endl;
    throw AsException("weight header parse failed: no header [2]");
  }
  // dtype
  char type = header[loc1 + 1];
  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  int word_size = atoi(str_ws.substr(0, loc2).c_str());
  switch (type) {
    case 'f': {
      switch (word_size) {
        case 4:
          dtype = DataType::FLOAT32;
          break;
        case 2:
          dtype = DataType::FLOAT16;
        default:
          break;
      }
      break;
    }
    case 'i': {
      switch (word_size) {
        case 8:
          dtype = DataType::INT64;
          break;
        case 4:
          dtype = DataType::INT32;
          break;
        case 2:
          dtype = DataType::INT16;
          break;
        case 1:
          dtype = DataType::INT8;
          break;
        default:
          break;
      }
      break;
    }
    case 'u': {
      switch (word_size) {
        case 1:
          dtype = DataType::UINT8;
          break;
        default:
          break;
      }
      break;
    }
    case 'b': {
      switch (word_size) {
        case 2:
          dtype = DataType::BFLOAT16;
          break;
        case 1:
          dtype = DataType::BOOL;
        default:
          break;
      }
      break;
    }
    default:

      LOG(ERROR) << "parse_allsparky_header: unsupported dtype : " << type
                 << std::endl;

      throw AsException("weight header parse failed: dtype [1]");
  }
  // shape
  loc1 = header.find("'shape': (");
  loc2 = header.find(")", loc1);
  if (loc1 == std::string::npos || loc2 == std::string::npos) {
    LOG(ERROR) << "parse_allsparky_header: failed to find 'shape': (' or ')'"
               << std::endl;

    throw AsException("weight header parse failed: [3]");
  }
  std::string str_shape = header.substr(loc1 + 10, loc2 - loc1 - 10);
  std::vector<std::string> shape_vec;
  util::split(shape_vec, str_shape, ", ");
  for (auto& s : shape_vec) {
    shape.Append(atoi(s.c_str()));
  }
  // data mode
  loc1 = header.find("'sparse_type': ");
  if (loc1 == std::string::npos) {
    LOG(ERROR) << "parse_allsparky_header: failed to find 'sparse_type'"
               << std::endl;

    throw AsException("weight header parse failed: [4]");
  }
  loc2 = header.find(",", loc1);
  std::string str_sparse_type = header.substr(loc1 + 15, loc2 - loc1 - 15);
  mode = (DataMode)(atoi(str_sparse_type.c_str()));
  nnz = 0;
  if (mode != DataMode::DENSE) {
    loc1 = header.find("'nnz': ");
    if (loc1 == std::string::npos) {
      LOG(ERROR) << "parse_allsparky_header: failed to find 'nnz'" << std::endl;
      throw AsException("weight header parse failed: [5]");
    }
    loc2 = header.find(",", loc1);
    std::string str_nnz = header.substr(loc1 + 7, loc2 - loc1 - 7);
    nnz = atoi(str_nnz.c_str());
  }
  split_mode = SplitMode::NOSPLIT;
  loc1 = header.find("'split_type': ");
  if (loc1 != std::string::npos) {
    loc2 = header.find(",", loc1);
    std::string str = header.substr(loc1 + 14, loc2 - loc1 - 14);
    split_mode = (SplitMode)(atoi(str.c_str()));
  }

  // group_list
  if (split_mode == SplitMode::GROUP_VSPLIT ||
      split_mode == SplitMode::MQA_VSPLIT) {
    loc1 = header.find("'group_list': (");
    loc2 = header.find(")", loc1);
    if (loc1 == std::string::npos || loc2 == std::string::npos) {
      LOG(ERROR) << "parse_allsparky_header: failed to find "
                    "''group_list': (' or ')'"
                 << std::endl;
      throw AsException("weight header parse failed: [3]");
    }
    std::string str_group_list = header.substr(loc1 + 15, loc2 - loc1 - 15);
    std::vector<std::string> group_list_vec;
    util::split(group_list_vec, str_group_list, ", ");
    for (auto& s : group_list_vec) {
      group_list.push_back(atoi(s.c_str()));
    }
  }

  return info;
}

#define AS_HEADER_BYTES 256

size_t WeightFileParser::TensorHeaderBytes() { return AS_HEADER_BYTES; }

TensorInfo WeightFileParser::ParseTensorInfo(FILE* fp) {
  char buffer[AS_HEADER_BYTES];
  memset(buffer, 0, AS_HEADER_BYTES);
  // fgets will make sure the the buffer end with '\0'
  std::string header = fgets(buffer, AS_HEADER_BYTES, fp);
  return ParseTensorInfo(buffer, AS_HEADER_BYTES);
}

void DenseWeightLoader::LoadFromMemory(const void* ptr, size_t len,
                                       std::shared_ptr<AsTensor> opt_in_tensor,
                                       std::shared_ptr<AsTensor> out_tensor) {
  auto splitter = WeightSplitterFactory::GetSplitterByMode(
      tensor_info_.split_mode, rank_info_);

  if (!splitter->IsModelProcessable(tensor_info_.split_mode)) {
    throw AsException("model not support.");
  }

  if (!splitter->IsSplittable(tensor_info_))
    throw AsException("model file cannot split");
  int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);

  auto splitMemType = out_tensor->GetDeviceType();
  if (!this->whole_weight_tensor_) {
    this->whole_weight_tensor_ = std::make_shared<AsTensor>(
        name_,
        // out_tensor->GetDeviceType(),
        // FIXME: use device type will cause multiple
        // card result not correct, but after change,it
        // will be faster on splitting multiple card.
        splitMemType, tensor_info_.dtype, tensor_info_.mode, tensor_info_.shape,
        flags);
  }
  this->whole_weight_tensor_->SetShape(Shape{tensor_info_.shape});

  // for input tensor, it will a shape[0,0] tensor, don't copy it.
  // TODO: for those tensor, handle it before weight load.
  CopyData(this->whole_weight_tensor_->GetDataPtr(),
           this->whole_weight_tensor_->GetDeviceType(), ptr, DeviceType::CPU,
           len);

  splitter->SetShape(tensor_info_, out_tensor);
  splitter->CopyWeight(tensor_info_, out_tensor, whole_weight_tensor_, ptr,
                       len);
}

// load a dense tensor from file stream to a tenesor.
void DenseWeightLoader::LoadFromFileStream(FILE* fp,
                                           std::shared_ptr<AsTensor> tensor) {
  assert(tensor_info_.mode == DataMode::DENSE);

  const std::string& name = tensor->GetName();

  int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);

  auto whole_weight =
      std::make_shared<AsTensor>(name, DeviceType::CPU, tensor_info_.dtype,
                                 tensor_info_.mode, tensor_info_.shape, flags);

  if (tensor_info_.shape.Count() == 0) {
    assert(0);
  }

  if (fread(whole_weight->GetDataPtr(), 1, whole_weight->GetSizeInByte(), fp) !=
      whole_weight->GetSizeInByte()) {
    LOG(ERROR) << "load_the_allsparky_file: failed fread" << std::endl;
    throw AsException("IOError: weight file load failed");
  }

  // LoadFromMemory(whole_weight->GetDataPtr(), whole_weight->GetSizeInByte(),
  //                whole_weight, tensor);
  LoadFromMemory(whole_weight->GetDataPtr(), whole_weight->GetSizeInByte(),
                 nullptr, tensor);
}

template <typename T>
static inline void* memcpy_and_seek(void* out_ptr, void* in_ptr, size_t size) {
  memcpy(out_ptr, in_ptr, size * sizeof(T));
  return (char*)in_ptr + size * sizeof(T);
}

void SparseWeightLoader::LoadFromMemory(const void* ptr, size_t len,
                                        std::shared_ptr<AsTensor> opt_in_tensor,
                                        std::shared_ptr<AsTensor> out_tensor) {
  assert(ptr != nullptr);

  if (ptr == nullptr) {
    throw AsException("sparse weight load only support raw pointer");
  }

  DataType& dtype = tensor_info_.dtype;
  DeviceType device_type = out_tensor->GetDeviceType();
  int nnz = tensor_info_.nnz;
  const void* cur_ptr = ptr;

  int cols = (int)tensor_info_.shape[1];
  switch (tensor_info_.mode) {
    case DataMode::CSC: {
      std::shared_ptr<CSCData> data_sparse = std::make_shared<CSCData>(
          "CSC_Data:" + name_, nnz, cols, device_type, SizeofType(dtype));
      CopyData(data_sparse->GetColOffsets(), device_type, cur_ptr,
               DeviceType::CPU, (cols + 1) * sizeof(int));
      cur_ptr = (char*)cur_ptr + (cols + 1) * sizeof(int);

      CopyData(data_sparse->GetRowIndices(), device_type, cur_ptr,
               DeviceType::CPU, nnz * sizeof(int));
      cur_ptr = (char*)cur_ptr + (nnz) * sizeof(int);

      CopyData(data_sparse->GetRawData(), device_type, cur_ptr, DeviceType::CPU,
               nnz * SizeofType(dtype));

      out_tensor->SetData(data_sparse);
      break;
    }
    case DataMode::ELL: {
      std::shared_ptr<ELLData> data_sparse = std::make_shared<ELLData>(
          "ELL_Data" + name_, nnz, cols, device_type, SizeofType(dtype));
      CopyData(data_sparse->GetRowIndices(), device_type, cur_ptr,
               DeviceType::CPU, nnz * sizeof(unsigned short));
      cur_ptr = (char*)cur_ptr + nnz * sizeof(unsigned short);
      CopyData(data_sparse->GetRawData(), device_type, cur_ptr, DeviceType::CPU,
               nnz * SizeofType(dtype));
      out_tensor->SetData(data_sparse);
      break;
    }
    default:
      LOG(ERROR) << "invalid data mode in allsparky format" << std::endl;
      throw AsException("unsupport sparse format");
  }
}

void SparseWeightLoader::LoadFromFileStream(
    FILE* fp, std::shared_ptr<AsTensor> out_tensor) {
  DataType& dtype = tensor_info_.dtype;
  DeviceType device_type = out_tensor->GetDeviceType();
  int nnz = tensor_info_.nnz;

  switch (tensor_info_.mode) {
    case DataMode::CSC: {
      int cols = (int)tensor_info_.shape[1];
      std::vector<char> col_offsets((cols + 1) * sizeof(int));
      std::vector<char> row_indices((nnz) * sizeof(int));
      std::vector<char> val((nnz)*SizeofType(dtype));
      fread(col_offsets.data(), 1, (cols + 1) * sizeof(int), fp);
      fread(row_indices.data(), 1, (nnz) * sizeof(int), fp);
      fread(val.data(), 1, (nnz)*SizeofType(dtype), fp);
      std::shared_ptr<CSCData> data_sparse = std::make_shared<CSCData>(
          "CSC_Data:" + name_, nnz, cols, device_type, SizeofType(dtype));
      CopyData(data_sparse->GetColOffsets(), device_type, col_offsets.data(),
               DeviceType::CPU, (cols + 1) * sizeof(int));
      CopyData(data_sparse->GetRowIndices(), device_type, row_indices.data(),
               DeviceType::CPU, nnz * sizeof(int));
      CopyData(data_sparse->GetRawData(), device_type, val.data(),
               DeviceType::CPU, nnz * SizeofType(dtype));
      out_tensor->SetData(data_sparse);
      break;
    }
    case DataMode::ELL: {
      int cols = (int)tensor_info_.shape[1];
      std::vector<char> row_indices((nnz) * sizeof(unsigned short));
      std::vector<char> val((nnz)*SizeofType(dtype));
      std::shared_ptr<ELLData> data_sparse = std::make_shared<ELLData>(
          "ELL_Data" + name_, nnz, cols, device_type, SizeofType(dtype));
      fread(row_indices.data(), 1, (nnz) * sizeof(unsigned short), fp);
      fread(val.data(), 1, (nnz)*SizeofType(dtype), fp);
      CopyData(data_sparse->GetRowIndices(), device_type, row_indices.data(),
               DeviceType::CPU, nnz * sizeof(unsigned short));
      CopyData(data_sparse->GetRawData(), device_type, val.data(),
               DeviceType::CPU, nnz * SizeofType(dtype));
      out_tensor->SetData(data_sparse);
      break;
    }
    default:
      LOG(ERROR) << "invalid data mode in allsparky format" << std::endl;
      throw AsException("unsupport sparse format");
  }
}

}  // namespace allspark
