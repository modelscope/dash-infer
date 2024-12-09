/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    extra_embedding.hpp
 */

#pragma once

#include <iomanip>
#include <sstream>

#include "City.h"
#include "allspark.h"
#include "as_param_check.hpp"
#include "core/tensor/tensor.h"

namespace allspark {

class ExtraEmbeddingUtils {
 public:
  struct RichEmbeddingInfo {
    int64_t place_holder_token;
    std::string place_holder_str;
    int start_pos;
    int place_holder_cnt;
    int offset = 0;
    int64_t hash;
    AsTensor* embedding;
  };
  using REInfoList = std::vector<RichEmbeddingInfo>;

  static void ParseMmInfo(TensorListMap& extra_embedding,
                          MultiMediaInfo* mm_info) {
    extra_embedding.clear();
    if (mm_info == nullptr) {
      return;
    }
    DLTensorListMap batch_embedding = mm_info->get_multimedia_map();
    for (auto& t_list : batch_embedding) {
      std::vector<std::shared_ptr<AsTensor>> tensor_list;
      for (DLManagedTensor* t : t_list.second) {
        tensor_list.push_back(
            std::make_shared<AsTensor>(t_list.first, t, DeviceType::CPU));
      }
      extra_embedding.insert({t_list.first, tensor_list});
    }
  }

  static AsStatus ParseExtraEmbedding(
      TensorListMap& extra_embedding, int64_t* input_ids_ptr, int seq_len,
      std::shared_ptr<REInfoList> reinfo_vec = nullptr) {
    if (extra_embedding.empty()) return AsStatus::ALLSPARK_SUCCESS;

    // for (auto item : extra_embedding) {
    //   LOG(INFO) << "###### "
    //             << "extra_embedding[" << item.first << "]" << ", "
    //             << "size: " << item.second.size() << ", "
    //             << "shape: " << item.second[0]->GetShape().ToString();
    // }

    std::map<int64_t, int> place_holder_image_counter;

    // 遍历所有输入，map的key是填充符，value就是对应模态的所有tensor id
    for (int pos = 0; pos < seq_len; pos++) {
      std::string place_holder_token_str = std::to_string(input_ids_ptr[pos]);
      if (extra_embedding.count(place_holder_token_str) > 0) {
        // 这时候已经找到对应的占位token id
        int each_image_place_holder_length = 1;
        int start_pos = pos;
        int64_t place_holder_token_id = input_ids_ptr[pos];

        // 找到第一个不是token的
        while (pos + 1 < seq_len &&
               input_ids_ptr[pos + 1] == place_holder_token_id) {
          each_image_place_holder_length++;
          pos++;
        }

        if (place_holder_image_counter[place_holder_token_id]) {
          place_holder_image_counter[place_holder_token_id] += 1;
        } else {
          place_holder_image_counter[place_holder_token_id] = 1;
        }

        // 如果找到占位符的图片数量，比给的图片数量更多
        // 这里会有错误。
        if (place_holder_image_counter[place_holder_token_id] >
            extra_embedding[place_holder_token_str].size()) {
          // clang-format off
          LOG(ERROR) << "Embedding relpace num error, "
                     << "input_num = " << place_holder_image_counter[place_holder_token_id] << ", "
                     << "embedding_num = " << extra_embedding[place_holder_token_str].size()
                     << std::endl;
          // clang-format on
          return AsStatus::ALLSPARK_PARAM_ERROR;
        }

        // 占位符的长度和图片的token的长度不同。
        AsTensor* embedding =
            (extra_embedding[place_holder_token_str])
                [place_holder_image_counter[place_holder_token_id] - 1]
                    .get();
        if (each_image_place_holder_length != embedding->GetShape()[0]) {
          LOG(ERROR) << "Embedding relpace length error, "
                     << "input_len = " << each_image_place_holder_length << ", "
                     << "embedding_len = " << embedding->GetShape()[0]
                     << std::endl;
          return AsStatus::ALLSPARK_PARAM_ERROR;
        }

        RichEmbeddingInfo reinfo;
        reinfo.place_holder_token = place_holder_token_id;
        reinfo.place_holder_str = place_holder_token_str;
        reinfo.start_pos = start_pos;
        reinfo.place_holder_cnt = each_image_place_holder_length;
        reinfo.embedding = embedding;

        if (reinfo_vec) {
          reinfo_vec->emplace_back(reinfo);
        }
      }
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

  static AsStatus UpdateREInfo(std::shared_ptr<REInfoList> reinfo_vec,
                               int prefix_len) {
    if (prefix_len == 0) return AsStatus::ALLSPARK_SUCCESS;

    int cached_idx = prefix_len - 1;
    for (auto& reinfo : *reinfo_vec) {
      int start_pos = reinfo.start_pos;
      int end_pos = reinfo.start_pos + reinfo.place_holder_cnt - 1;
      if (cached_idx < start_pos) {
        reinfo.start_pos = start_pos - prefix_len;
        reinfo.offset = 0;
      } else if (start_pos <= cached_idx && cached_idx < end_pos) {
        reinfo.start_pos = 0;
        reinfo.offset = prefix_len - start_pos;
      } else /* end_pos <= cached_idx */ {
        reinfo.start_pos = -1;
        reinfo.offset = reinfo.place_holder_cnt;
      }
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

  static AsStatus CreateTensorForHash(std::shared_ptr<Request> request,
                                      TensorMap& tensor_map,
                                      std::string src_tensor_name) {
    std::string dst_tensor_name = src_tensor_name + "_for_hash";

    if (!request->extra_embedding.empty()) {
      if (request->extra_embedding.count("hash_input") > 0) {
        // step 1: parse extra_embedding info
        int64_t* tensor_ptr =
            (int64_t*)tensor_map[src_tensor_name]->GetDataPtr();
        int seq_len = tensor_map[src_tensor_name]->GetShape()[1];
        auto reinfo_vec = std::make_shared<ExtraEmbeddingUtils::REInfoList>();
        AS_CHECK_STATUS(ExtraEmbeddingUtils::ParseExtraEmbedding(
            request->extra_embedding, tensor_ptr, seq_len, reinfo_vec));

        // step 2: hash inputs
        AS_CHECK_STATUS(ExtraEmbeddingUtils::HashEmbedding(
            request->extra_embedding, reinfo_vec));

        // step 3: create a new input tensor
        auto dst_tensor = std::make_shared<AsTensor>(
            dst_tensor_name, *tensor_map[src_tensor_name]);

        // step 4: replace place holder with hashes
        ExtraEmbeddingUtils::ReplacePlaceHolder(dst_tensor, reinfo_vec);

        tensor_map.insert({dst_tensor_name, dst_tensor});
      } else {
        LOG(ERROR) << "multi-media content `hash_input` "
                   << "of request " << request->request_id << " is missing.";
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
    } else {
      // no extra embedding, use original input_ids for hash
      tensor_map.insert({dst_tensor_name, tensor_map[src_tensor_name]});
    }

    return AsStatus::ALLSPARK_SUCCESS;
  }

 private:
  static int64_t Hash64(void* data, int len) {
    uint64_t hash_u64 = CityHash64((const char*)data, len);
    // hash_u64 = hash_u64 & 0xfffff; // for print
    int64_t* hash_s64_ptr = reinterpret_cast<int64_t*>(&hash_u64);
    int64_t hash_s64 = *hash_s64_ptr;

    return hash_s64;
  }

  static AsStatus HashEmbedding(TensorListMap& extra_embedding,
                                std::shared_ptr<REInfoList> reinfo_vec) {
    if (extra_embedding.count("hash_input") <= 0)
      return AsStatus::ALLSPARK_SUCCESS;

    int hash_input_num = extra_embedding["hash_input"].size();
    if (hash_input_num != reinfo_vec->size()) {
      LOG(ERROR) << "Embedding hash input num error, "
                 << "input_num = " << reinfo_vec->size() << ", "
                 << "hash_input num = " << hash_input_num << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }

    for (int i = 0; i < hash_input_num; i++) {
      std::shared_ptr<AsTensor> item = extra_embedding["hash_input"][i];

      int64_t hash = Hash64(item->GetDataPtr(), item->GetSizeInByte());
      (*reinfo_vec)[i].hash = hash;
    }
    return AsStatus::ALLSPARK_SUCCESS;
  }

  static AsStatus ReplacePlaceHolder(std::shared_ptr<AsTensor> target_tensor,
                                     std::shared_ptr<REInfoList> reinfo_vec) {
    if (target_tensor->GetDeviceType() != DeviceType::CPU) {
      LOG(ERROR) << __FUNCTION__ << " only support cpu tensors. " << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }

    int64_t* target_tensor_ptr = (int64_t*)target_tensor->GetDataPtr();
    for (const auto& reinfo : *reinfo_vec) {
      for (int pos = reinfo.start_pos;
           pos < reinfo.start_pos + reinfo.place_holder_cnt; pos++) {
        target_tensor_ptr[pos] = reinfo.hash;
      }
    }

    return AsStatus::ALLSPARK_SUCCESS;
  }

#if 0
public:
  static void test_UpdateREInfo() {
    RichEmbeddingInfo reinfo;
    reinfo.place_holder_token = 151859;
    reinfo.place_holder_str = "151859";
    reinfo.start_pos = 64;
    reinfo.place_holder_cnt = 192;
    reinfo.embedding = nullptr;

    std::vector<int> length_vec = {0, 32, 63, 64, 65, 128, 255, 256, 257, 288};
    for (int len : length_vec) {
      auto reinfo_vec = std::make_shared<ExtraEmbeddingUtils::REInfoList>();
      reinfo_vec->emplace_back(reinfo);

      UpdateREInfo(reinfo_vec, len);
      LOG(INFO) << "###### " << __FUNCTION__ << ", "
                << "prefix_len: " << len << ", "
                << "place_holder_cnt: " << (*reinfo_vec)[0].place_holder_cnt << ", "
                << "start_pos: " << (*reinfo_vec)[0].start_pos << ", "
                << "offset: " << (*reinfo_vec)[0].offset << ", "
                << "copy_len: " << (*reinfo_vec)[0].place_holder_cnt - (*reinfo_vec)[0].offset;
    }
  }
#endif
};

};  // namespace allspark
