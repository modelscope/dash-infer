/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    multi_media_info.cpp
 */

#include <utility/check.h>

#include <common/as_param_check.hpp>
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
namespace allspark {

// using FloatPair = std::pair<std::vector<float>, std::vector<int64_t>>;
class MultiMediaInfoImpl final {
 public:
  MultiMediaInfoImpl() {}
  ~MultiMediaInfoImpl() {}
  AsStatus add_multimedia_content(
      std::string key, std::vector<std::vector<float>> data_list,
      std::vector<std::vector<int64_t>> shape_list) {
    std::vector<std::pair<std::vector<float>, std::vector<int64_t>>>
        tensor_list;
    int size = data_list.size();
    for (int i = 0; i < size; i++) {
      std::vector<float> data = data_list[i];
      std::vector<int64_t> shape = shape_list[i];
      std::pair<std::vector<float>, std::vector<int64_t>> tensor =
          std::make_pair(data, shape);
      tensor_list.push_back(tensor);
    }
    tensor_list_map.insert({key, tensor_list});
    return AsStatus::ALLSPARK_SUCCESS;
  }
  AsStatus add_multimedia_content(
      std::string key, std::vector<DLManagedTensor*>& dl_tensor_list) {
    multimedia_map.insert({key, dl_tensor_list});
    return AsStatus::ALLSPARK_SUCCESS;
  }
  AsStatus set_multimedia_type(int multimedia_type_new) {
    multimedia_type = multimedia_type_new;
    return AsStatus::ALLSPARK_SUCCESS;
  }
  std::map<std::string,
           std::vector<std::pair<std::vector<float>, std::vector<int64_t>>>>&
  get_tensor_list_map() {
    return tensor_list_map;
  }
  DLTensorListMap get_multimedia_map() { return multimedia_map; }

 private:
  int multimedia_type = 0;
  DLTensorListMap multimedia_map;
  std::map<std::string,
           std::vector<std::pair<std::vector<float>, std::vector<int64_t>>>>
      tensor_list_map;
};
MultiMediaInfo::MultiMediaInfo()
    : multi_media_info_impl_(std::make_unique<MultiMediaInfoImpl>()) {}
MultiMediaInfo::~MultiMediaInfo() = default;

AsStatus MultiMediaInfo::add_multimedia_content(
    std::string key, std::vector<std::vector<float>> data_list,
    std::vector<std::vector<int64_t>> shape_list) {
  return multi_media_info_impl_->add_multimedia_content(key, data_list,
                                                        shape_list);
}
AsStatus MultiMediaInfo::add_multimedia_content(
    std::string key, std::vector<DLManagedTensor*>& dl_tensor_py_list) {
  return multi_media_info_impl_->add_multimedia_content(key, dl_tensor_py_list);
}
AsStatus MultiMediaInfo::set_multimedia_type(int multimedia_type_new) {
  return multi_media_info_impl_->set_multimedia_type(multimedia_type_new);
}
std::map<std::string,
         std::vector<std::pair<std::vector<float>, std::vector<int64_t>>>>&
MultiMediaInfo::get_tensor_list_map() {
  return multi_media_info_impl_->get_tensor_list_map();
}
DLTensorListMap MultiMediaInfo::get_multimedia_map() {
  return multi_media_info_impl_->get_multimedia_map();
}
}  // namespace allspark
