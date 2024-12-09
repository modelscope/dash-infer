/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allsparkz_util.cpp
 */

#include "allsparkz_util.h"

#include <common/common.h>

#include <cstring>
#include <fstream>

#ifdef ENABLE_FP16
#ifdef ENABLE_CUDA
#include <cuda_fp16.h>
#else
#include <common/float16.h>
#endif
#endif

#define ELL_MAXC 0.1
namespace allspark {

namespace util {
template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
  // write in little endian
  size_t len = strlen(rhs);
  lhs.reserve(len);
  for (size_t byte = 0; byte < len; byte++) {
    lhs.push_back(rhs[byte]);
  }
  return lhs;
}

std::string make_string(std::vector<char> in) {
  std::string s;
  s.assign(in.begin(), in.end());
  return s;
}

void split_(std::vector<std::string>& out, std::string& str,
            const std::string delim) {
  out.clear();
  std::string s(str);
  size_t pos;
  while ((pos = s.find(delim)) != std::string::npos) {
    out.emplace_back(s.substr(0, pos));
    s = s.substr(pos + delim.size(), s.size());
  }
  out.emplace_back(s);
}
void parse_npy_header(FILE* fp, char& dtype, int& word_size,
                      std::vector<int>& shape) {
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
  split_(shape_vec, str_shape, ", ");
  for (auto& s : shape_vec) {
    shape.push_back(atoi(s.c_str()));
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
  dtype = type;
  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  word_size = atoi(str_ws.substr(0, loc2).c_str());
}
char BigEndianTest() {
  int x = 1;
  return (((char*)&x)[0]) ? '<' : '>';
}

std::vector<char> create_allsparky_header(TensorAttribute& tensor_info) {
  std::vector<char> dict;
  dict += "{'descr': '";
  dict += BigEndianTest();
  dict.push_back(tensor_info.dtype);
  dict += std::to_string(tensor_info.word_size);
  dict += "', 'fortran_order': False, ";
  dict += "'shape': (";
  dict += std::to_string(tensor_info.shape[0]);
  for (size_t i = 1; i < tensor_info.shape.size(); i++) {
    dict += ", ";
    dict += std::to_string(tensor_info.shape[i]);
  }
  if (tensor_info.shape.size() == 1) dict += ",";
  dict += "),";
  dict += "'group_list': (";
  if (tensor_info.group_list.size() > 0) {
    dict += std::to_string(tensor_info.group_list[0]);
    for (size_t i = 1; i < tensor_info.group_list.size(); i++) {
      dict += ", ";
      dict += std::to_string(tensor_info.group_list[i]);
    }
    if (tensor_info.group_list.size() == 1) dict += ",";
  }
  dict += "),";

  dict += "'sparse_type': ";
  dict += std::to_string(tensor_info.sparse_type);
  dict += ",";
  dict += "'nnz': ";
  dict += std::to_string(tensor_info.nnz);
  dict += ",";
  dict += "'split_type': ";
  dict += std::to_string(tensor_info.split_mode);
  dict += ",";
  dict += "}\n";
  std::vector<char> header;
  header.insert(header.end(), dict.begin(), dict.end());
  return header;
}

std::string save_allsparky(const std::string& bin_data,
                           TensorAttribute& tensor_info) {
  std::string res;
  res.reserve(bin_data.size() + 512);
  std::vector<int> shape = tensor_info.shape;
  int word_size = tensor_info.word_size;
  switch (tensor_info.sparse_type) {
    case 0:  // dense
    {
      std::vector<char> header = create_allsparky_header(tensor_info);
      res.insert(res.begin(), header.begin(), header.end());
      res.append(bin_data);
      break;
    }
    case 1:  // csc
    {
      int nnz;
      int row = shape[0];
      int col = shape[1];
      std::vector<char> header;
      int VECT = 16 / word_size;
      if (word_size == 4) {  // fp32
        nnz = get_nnz(reinterpret_cast<const float*>(bin_data.data()), row, col,
                      VECT);
        std::vector<char> col_offset((col + 1) * sizeof(int));
        std::vector<char> row_idx(nnz * sizeof(int));
        std::vector<char> data(nnz * word_size);
        dense_to_csc_padding(reinterpret_cast<const float*>(bin_data.data()),
                             row, col, reinterpret_cast<float*>(data.data()),
                             reinterpret_cast<int*>(row_idx.data()),
                             reinterpret_cast<int*>(col_offset.data()), VECT);
        tensor_info.nnz = nnz;
        header = create_allsparky_header(tensor_info);

        res.insert(res.begin(), header.begin(), header.end());
        res.insert(res.end(), col_offset.begin(), col_offset.end());
        res.insert(res.end(), row_idx.begin(), row_idx.end());
        res.insert(res.end(), data.begin(), data.end());
#ifdef ENABLE_FP16
      } else if (word_size == 2) {  // fp16
        nnz = get_nnz((half*)bin_data.data(), row, col, VECT);
        std::vector<char> col_offset((col + 1) * sizeof(int));
        std::vector<char> row_idx(nnz * sizeof(int));
        std::vector<char> data(nnz * word_size);
        dense_to_csc_padding((half*)bin_data.data(), row, col,
                             (half*)data.data(), (int*)row_idx.data(),
                             (int*)col_offset.data(), VECT);
        tensor_info.nnz = nnz;
        header = create_allsparky_header(tensor_info);

        res.insert(res.begin(), header.begin(), header.end());
        res.insert(res.end(), col_offset.begin(), col_offset.end());
        res.insert(res.end(), row_idx.begin(), row_idx.end());
        res.insert(res.end(), data.begin(), data.end());
#endif
      }
      break;
    }
    case 2:  // ELL
    {
      int nnz;
      int row = shape[0];
      int col = shape[1];
      std::vector<char> header;
      int VECT = 16 / word_size;
      if (word_size == 4) {  // fp32
        nnz = get_nnz_ell(reinterpret_cast<const float*>(bin_data.data()), row,
                          col, VECT);
        // if (nnz / col > row * ELL_MAXC) {
        //     LOG(ERROR) << "not support ell ,nnz/col= " << nnz / col
        //                << std::endl;
        //     return "";
        // }
        std::vector<char> row_idx(nnz * sizeof(unsigned short));
        std::vector<char> data(nnz * word_size);
        dense_to_ell_padding(reinterpret_cast<const float*>(bin_data.data()),
                             row, col, nnz,
                             reinterpret_cast<float*>(data.data()),
                             (unsigned short*)row_idx.data(), VECT);
        tensor_info.nnz = nnz;
        header = create_allsparky_header(tensor_info);

        res.insert(res.begin(), header.begin(), header.end());
        res.insert(res.end(), row_idx.begin(), row_idx.end());
        res.insert(res.end(), data.begin(), data.end());
#ifdef ENABLE_FP16
      } else if (word_size == 2) {  // fp16
        nnz = get_nnz_ell(reinterpret_cast<const half*>(bin_data.data()), row,
                          col, VECT);
        // if (nnz / col > row * ELL_MAXC) {
        //     LOG(ERROR) << "not support ell ,nnz/col= " << nnz / col
        //                << std::endl;
        //     return "";
        // }
        std::vector<char> row_idx(nnz * sizeof(unsigned short));
        std::vector<char> data(nnz * word_size);
        dense_to_ell_padding(reinterpret_cast<const half*>(bin_data.data()),
                             row, col, nnz,
                             reinterpret_cast<half*>(data.data()),
                             (unsigned short*)row_idx.data(), VECT);
        tensor_info.nnz = nnz;
        header = create_allsparky_header(tensor_info);
        res.insert(res.begin(), header.begin(), header.end());
        res.insert(res.end(), row_idx.begin(), row_idx.end());
        res.insert(res.end(), data.begin(), data.end());
#endif
      }
      break;
    }
    default:
      throw std::runtime_error("not support sparse type");
      break;
  }
  return res;
}

void save_allsparkz(std::map<std::string, std::string>& weights,
                    const std::string& weights_path) {
  std::ofstream fout2(weights_path, std::ios::out);
  for (auto& w : weights) {
    std::vector<char> local_header;
    local_header += "AS";
    local_header += (uint16_t)0x01;
    local_header += (uint16_t)w.first.size();
    local_header += w.first;
    std::string l_h;
    l_h.assign(local_header.begin(), local_header.end());
    fout2 << l_h << w.second;
    w.second = "";  // 转换完直接释放
  }
  std::vector<char> global_header;
  global_header += "AS";
  global_header += (uint16_t)0;
  global_header += (uint16_t)0;
  fout2 << make_string(global_header);
}
void save_allsparky_tofile(const std::string& weights_path,
                           const std::string& name, const std::string& bin_data,
                           TensorAttribute& tensor_info) {
  std::ofstream fout2(weights_path, std::ios::app);  // 续写文件
  std::vector<char> local_header;
  local_header += "AS";
  local_header += (uint16_t)0x01;
  local_header += (uint16_t)name.size();
  local_header += name;
  std::string l_h;
  l_h.assign(local_header.begin(), local_header.end());

  if (tensor_info.sparse_type == 0) {  // DENSE
    std::vector<char> header = create_allsparky_header(tensor_info);
    l_h.append(header.begin(), header.end());
    fout2 << l_h;
    fout2 << bin_data;
  } else {  // other sparse type
    std::string w = save_allsparky(bin_data, tensor_info);
    fout2 << l_h << w;
  }
}
void save_allsparky_tofile(const std::string& weights_path,
                           const std::string& name, void* data_ptr,
                           int64_t nbytes, TensorAttribute& tensor_info) {
  std::ofstream fout2(weights_path, std::ios::app);  // 续写文件
  std::vector<char> local_header;
  local_header += "AS";
  local_header += (uint16_t)0x01;
  local_header += (uint16_t)name.size();
  local_header += name;
  std::string l_h;
  l_h.assign(local_header.begin(), local_header.end());

  if (tensor_info.sparse_type == 0) {  // DENSE
    std::vector<char> header = create_allsparky_header(tensor_info);
    l_h.append(header.begin(), header.end());
    fout2 << l_h;
    fout2.write(reinterpret_cast<const char*>(data_ptr), nbytes);
  } else {  // other sparse type
    std::vector<char> vec(nbytes);
    memcpy(vec.data(), data_ptr, nbytes);
    std::string bin_data(vec.data(), vec.size());
    std::string w = save_allsparky(bin_data, tensor_info);
    fout2 << l_h << w;
  }
}
void set_global_header(const std::string& weights_path) {
  std::ofstream fout2(weights_path, std::ios::app);  // 续写文件
  std::vector<char> global_header;
  global_header += "AS";
  global_header += (uint16_t)0;
  global_header += (uint16_t)0;
  fout2 << make_string(global_header);
}
}  // namespace util
}  // namespace allspark
