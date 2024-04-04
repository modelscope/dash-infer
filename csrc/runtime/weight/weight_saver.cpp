/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_saver.cpp
 */

#include "weight_saver.h"

namespace allspark {
inline char BigEndianTest() {
  int x = 1;
  return (((char*)&x)[0]) ? '<' : '>';
}

WeightSerialization::WeightSerialization() {}

void WeightSerialization::SerializeMultipleTensor(const TensorMap& tensors,
                                                  std::string* bin_data) {
  bin_data->clear();
  uint16_t flag = 0x01;
  for (const auto& t : tensors) {
    uint16_t name_size = t.first.size();
    // add local header
    bin_data->append("AS");
    bin_data->append((char*)&flag, 2);
    bin_data->append((char*)&name_size, 2);
    bin_data->append(t.first);
    // add allsparky data
    std::string allsparky_data;
    SerializeSingleTensor(t.second.get(), &allsparky_data);
    bin_data->append(allsparky_data);
  }
  // add end header
  flag = 0x00;
  bin_data->append("AS");
  bin_data->append((char*)&flag, 2);
  bin_data->append((char*)&flag, 2);
}

void WeightSerialization::SerializeSingleTensor(const AsTensor* tensor,
                                                std::string* bin_data) {
  DLOG(INFO) << "allsparky_saves" << std::endl;
  const Shape& shape = tensor->GetShape();
  DataMode mode = tensor->GetDataMode();
  // add allsparky_header
  bin_data->clear();
  bin_data->append("{'descr': '");
  bin_data->push_back(BigEndianTest());
  std::string type;
  DataType dtype = tensor->GetDataType();
  switch (tensor->GetDataType()) {
    case DataType::FLOAT32:
      type = "f4";
      break;
    case DataType::FLOAT16:
      type = "f2";
      break;
    case DataType::INT64:
      type = "i8";
      break;
    case DataType::INT32:
      type = "i4";
      break;
    case DataType::INT16:
      type = "i2";
      break;
    case DataType::INT8:
      type = "i1";
      break;
    case DataType::UINT8:
      type = "u1";
      break;
    case DataType::BOOL:
      type = "b";
      break;
    default:
      LOG(ERROR) << "Unsupported allsparky dtype : "
                 << DataType_Name(tensor->GetDataType()) << std::endl;
      break;
  }
  bin_data->append(type);
  bin_data->append("', 'fortran_order': False, 'shape': (");
  int len = shape.Size();
  if (len) {
    bin_data->append(std::to_string(shape[0]));
  }
  for (int i = 1; i < len; ++i) {
    bin_data->append(", ");
    bin_data->append(std::to_string(shape[i]));
  }
  if (len == 1) bin_data->append(",");
  bin_data->append("),'sparse_type': ");
  bin_data->append(std::to_string((int)mode));
  bin_data->append(",'nnz': ");
  // add bin data
  switch (mode) {
    case DataMode::DENSE: {
      bin_data->append("0");
      bin_data->append(",}");
      // 内存对齐
      int rem = bin_data->size() % 32;
      if (rem) {
        bin_data->append(std::string(32 - rem, ' '));
      }
      bin_data->back() = '\n';
      DenseData* data = (DenseData*)(tensor->GetData());
      char* data_ptr = (char*)(data->GetRawData());
      bin_data->append(data_ptr, data->GetSize());
      break;
    }
    case DataMode::CSC: {
      bin_data->append(
          std::to_string((int)((CSCData*)(tensor->GetData()))->GetNNZ()));
      bin_data->append(",}");
      // 内存对齐
      int rem = bin_data->size() % 32;
      if (rem) {
        bin_data->append(std::string(32 - rem, ' '));
      }
      bin_data->back() = '\n';
      int cols = shape[1];
      CSCData* data = (CSCData*)(tensor->GetData());
      char* data_ptr = (char*)(data->GetRawData());
      char* row_idx_ptr = (char*)(data->GetRowIndices());
      char* col_offset_ptr = (char*)(data->GetColOffsets());

      bin_data->append(col_offset_ptr, (cols + 1) * sizeof(int));
      bin_data->append(row_idx_ptr, data->GetNNZ() * sizeof(int));
      bin_data->append(data_ptr, data->GetNNZ() * SizeofType(dtype));
      break;
    }
    case DataMode::ELL: {
      bin_data->append(
          std::to_string((int)((ELLData*)(tensor->GetData()))->GetNNZ()));
      bin_data->append(",}");
      // 内存对齐
      int rem = bin_data->size() % 32;
      if (rem) {
        bin_data->append(std::string(32 - rem, ' '));
      }
      bin_data->back() = '\n';
      ELLData* data = (ELLData*)(tensor->GetData());
      char* data_ptr = (char*)(data->GetRawData());
      char* row_idx_ptr = (char*)(data->GetRowIndices());
      bin_data->append(row_idx_ptr, data->GetNNZ() * sizeof(unsigned short));
      bin_data->append(data_ptr, data->GetNNZ() * SizeofType(dtype));
      break;
    }
    default:
      LOG(ERROR) << "Currently not support save tensor with DataMode:"
                 << DataMode_Name(tensor->GetDataMode()) << std::endl;
      break;
  }
}
}  // namespace allspark
