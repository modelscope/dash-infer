/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    npz_util.cpp
 */

#include "npz_util.h"

#include <vector>

#include "string_util.h"

namespace allspark {
namespace util {

void parse_npy_header(FILE* fp, DataType& dtype, Shape& shape) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) {
    throw std::runtime_error("parse_npy_header: failed fread");
  }
  std::string header = fgets(buffer, 256, fp);
  size_t loc1, loc2;
  // shape
  loc1 = header.find("(");
  loc2 = header.find(")");
  if (loc1 == std::string::npos || loc2 == std::string::npos) {
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: '(' or ')'");
  }
  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  std::vector<std::string> shape_vec;
  split(shape_vec, str_shape, ", ");
  for (auto& s : shape_vec) {
    shape.Append(atoi(s.c_str()));
  }
  // endian, word size, data type
  // byte order code | stands for not applicable.
  // not sure when this applies except for byte array
  loc1 = header.find("descr");
  if (loc1 == std::string::npos) {
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'descr'");
  }

  loc1 += 9;
  bool littleEndian =
      (header[loc1] == '<' || header[loc1] == '|' ? true : false);
  if (!littleEndian) {
    throw std::runtime_error("Invalid npz data");
  }
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
      dtype = DataType::BOOL;
      break;
    }
    default:
      LOG(ERROR) << "Unsupported numpy dtype." << std::endl;
      break;
  }
}

std::unique_ptr<AsTensor> load_the_npy_file(FILE* fp, const std::string& name,
                                            DeviceType device_type) {
  Shape shape;
  DataType dtype;
  parse_npy_header(fp, dtype, shape);
  std::unique_ptr<AsTensor> tensor = std::make_unique<AsTensor>(
      name, device_type, dtype, DataMode::DENSE, Shape(shape));
  size_t len = shape.Count() * SizeofType(dtype);
  std::vector<char> buffer(len);
  size_t nread = fread(buffer.data(), 1, len, fp);
  if (nread != len) {
    throw std::runtime_error("load_the_npy_file: failed fread");
  }
  tensor->CopyDataFrom(buffer.data(), len, DeviceType::CPU);
  return tensor;
}

void load_npz_data(FILE* fp, TensorMap& data, DeviceType device_type) {
  while (1) {
    std::vector<char> local_header(30);
    size_t headerres = fread(&local_header[0], sizeof(char), 30, fp);
    if (headerres != 30) throw std::runtime_error("npz_load: failed fread");

    // if we've reached the global header, stop reading
    if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

    // read in the variable name
    uint16_t name_len = *(uint16_t*)&local_header[26];
    std::string varname(name_len, ' ');
    size_t vname_res = fread(&varname[0], sizeof(char), name_len, fp);
    if (vname_res != name_len)
      throw std::runtime_error("npz_load: failed fread");

    // erase the lagging .npy
    varname.erase(varname.end() - 4, varname.end());

    // read in the extra field
    uint16_t extra_field_len = *(uint16_t*)&local_header[28];
    if (extra_field_len > 0) {
      std::vector<char> buff(extra_field_len);
      size_t efield_res = fread(&buff[0], sizeof(char), extra_field_len, fp);
      if (efield_res != extra_field_len)
        throw std::runtime_error("npz_load: failed fread");
    }
    data[varname] = load_the_npy_file(fp, varname, device_type);
  }
}

void npz_load(const std::string& file_path, TensorMap& data,
              DeviceType device_type) {
  // TODO:(chenchu.zs) 添加相关的实现逻辑
  return;
}
void npz_loads(const std::string& bin_data, TensorMap& data,
               DeviceType device_type) {
  FILE* fp = fmemopen((void*)bin_data.c_str(), bin_data.size(), "rb");
  if (!fp) {
    throw std::runtime_error("Unable to read npz memory file!");
  }
  load_npz_data(fp, data, device_type);
  fclose(fp);
}
}  // namespace util
}  // namespace allspark
