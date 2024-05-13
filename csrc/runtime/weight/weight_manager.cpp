/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_manager.cpp
 */
#include "weight_manager.h"

#include <common/device_context.h>
#include <core/tensor/tensor.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <utility/file_util.h>
#include <utility/mutex_wrapper.h>
#include <utility/timer.h>

#include <algorithm>
#include <as_param_check.hpp>
#include <mutex>
#include <shared_mutex>

#include "weight_loader.h"
#include "weight_saver.h"

#define AS_WEIGHT_LOAD_FROM_MMEMAP

#include <mutex>

namespace allspark {

// all the weight and weight buffer storeage in  class
WeightManager::WeightManager() {
  // setup manager, one engine , one weight manager.
}

std::shared_ptr<ModelWeightHandler> WeightManagerImpl::RegisterModel(
    AsModelConfig& config, std::shared_ptr<TransformerProto> model_ir) {
  rw_write_lock lk(lock_, "RegisterModel");
  size_t new_id = weight_handler_store_.size();
  weight_handler_store_.emplace_back(
      std::make_shared<ModelWeightHandler>(new_id, config, model_ir));

  proto_store_[new_id] = model_ir;
  auto ret_handle = weight_handler_store_.back();
  return ret_handle;
}

std::vector<ModelWeightAccessInfo>
WeightManagerImpl::GetAccessOrderOfWeightFile(
    std::shared_ptr<ModelWeightHandler> mhandle) {
  std::vector<ModelWeightAccessInfo> ret;

  LOG(INFO) << "Start open model file "
            << mhandle->GetModelConfig().weights_path;
  std::unique_ptr<FILE, int (*)(FILE*)> fp_ptr(
      fopen(mhandle->GetModelConfig().weights_path.c_str(), "rb"), fclose);
  // close the fd when out of scope.
  FILE* fp = fp_ptr.get();

  LOG(INFO) << "Open model file success. ";
  while (true) {
    std::vector<char> local_header(6);
    ModelWeightAccessInfo new_info;

    if (fread(&local_header[0], sizeof(char), 6, fp) != 6) {
      LOG(ERROR) << "allsparkz_load: failed fread" << std::endl;
      throw AsModelException("invalid weight file: no header");
    }
    // if we've reached the global header, stop reading
    uint16_t flag = *(uint16_t*)&local_header[2];
    if (flag != 0x01) {
      break;
    }
    // read in the variable name
    uint16_t name_len = *(uint16_t*)&local_header[4];
    std::string varname(name_len, ' ');
    if (fread(&varname[0], sizeof(char), name_len, fp) != name_len) {
      LOG(ERROR) << "allsparkz_load: failed fread" << std::endl;
      throw AsModelException("invalid weight file: io err");
    }
    varname[name_len] = '\0';

    new_info.name = varname;

    WeightFileParser weight_header_parser;
    TensorInfo info = weight_header_parser.ParseTensorInfo(fp);
    new_info.size_bytes = SizeofType(info.dtype) * info.shape.Count();
    new_info.weight_offset = ftell(fp);
    // offset is wegith
    new_info.info = info;

    ret.push_back(std::move(new_info));

    // seek to next weight, keep current info.
    if (!SeekToNextTensor(fp, info)) {
      break;
    } else {
      // continue process until there is no weight header.
    }
  }

  // todo: add var to control this
  std::stable_sort(
      ret.begin(), ret.end(),
      [](const ModelWeightAccessInfo& lhs, const ModelWeightAccessInfo& rhs) {
        return lhs.size_bytes > rhs.size_bytes;
      });
  LOG(INFO) << "Weight file header parse success..." << ret.size()
            << " weight tensors are going to load. ";

  return ret;
};

#define SEEK_PTR_BYTES(ptr, byte_size) (char*)ptr + byte_size
// memory saving strategy.
// access the weight file, get the order and size of each tensor,
// sort the list with decent order
// start 8 task to handle each task.
// in each task {
//  a) each thread mmap the content with shared, and access the pointer with
//  their offset, even through user space still can access whole file,
//     but it's only allocate memory on demand
//   b) after this task finished, unmap the file, and loop back to a, so memory
//   only allcoate on demand again.
//  }
//
//  for the iops, it should only happens when mmap page fault, and will cached
//  in kernel buffer, so it's O(n) times io.
AsStatus WeightManagerImpl::LoadWeightForModel(
    const DeviceContext& target_device_ctx,
    std::shared_ptr<ModelWeightHandler>& weight_handler, RankInfo& rank_info) {
  LOG(INFO) << "Start Loading weight for model " << rank_info;
  util::Timer start;

  // for restore from cold device.
  // the only thing needs to do is check the handler have cold store.
  // and make them gpu tensor for each rank,
  // that's all.
  // don't needs to load from file or split.
  //

  TensorMap* model_weights_buffer;  // for swap, add later
  auto processOrder = GetAccessOrderOfWeightFile(weight_handler);

  WeightFileParser weight_parser;

  int finished = 0;

  auto tensor_map_for_current_rank = std::make_shared<TensorMap>();

  if (processOrder.size() == 0) {
    LOG(ERROR) << "weight file header parse failed, with only 0 weight.";
    throw AsException("weight with no weight tensor.");
  }

  // check the model's correctness.
  //  CheckModelConsistency(weight_handler);

  DLOG(INFO) << "start process weight in order before openmp: process order: "
             << processOrder.size();
  int loop_count_one_shot = 4;
  int loop_count_total =
      (processOrder.size() + loop_count_one_shot - 1) / loop_count_one_shot;
  int total_size = processOrder.size();
  for (int i = 0; i < loop_count_total; i++) {
    // set shape will allocate memory in this function call.
    std::unique_ptr<FILE, int (*)(FILE*)> fp_ptr(
        fopen(weight_handler->GetModelConfig().weights_path.c_str(), "rb"),
        fclose);
    // close this file when this weight finish.

    int fd = fileno(fp_ptr.get());
    struct stat sb;
    fstat(fd, &sb);

    // multiple thread can share this file mapping
    // mmap will start with the current file's seek poisition, not beginning
    // of the file.

#ifdef AS_WEIGHT_LOAD_FROM_MMEMAP
    const char* weight_mem_base =
        (const char*)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (weight_mem_base == MAP_FAILED) {
      LOG(ERROR) << "weight load map failed:  " << i
                 << " error: " << strerror(errno);
      continue;
    }
#endif

    for (int j = 0;
         j < loop_count_one_shot && j + i * loop_count_one_shot < total_size;
         j++) {
      ModelWeightAccessInfo weight_info =
          processOrder[i * loop_count_one_shot + j];

      std::shared_ptr<AsTensor> tensor = std::make_unique<AsTensor>(
          weight_info.name, target_device_ctx.GetDeviceType());

      tensor->SetDataType(weight_info.info.dtype);
      tensor->SetDataMode(weight_info.info.mode);
      AS_CHECK_STATUS(tensor->SetShape(Shape(weight_info.info.shape)));

      if (weight_info.info.mode != DataMode::DENSE) {
        fseek(fp_ptr.get(), weight_info.weight_offset, SEEK_SET);
        SparseWeightLoader sparse_loader(weight_info.info, rank_info,
                                         tensor->GetName());
        sparse_loader.LoadFromFileStream(fp_ptr.get(), tensor);
      } else {
        DenseWeightLoader dense_loader(weight_info.info, rank_info,
                                       tensor->GetName(), nullptr);

        if (weight_info.weight_offset + weight_info.size_bytes >= sb.st_size) {
          LOG(ERROR) << "file weight memory less than weight size, "
                        "something wrong.";

#ifdef AS_WEIGHT_LOAD_FROM_MMEMAP
          int ret = munmap((void*)weight_mem_base, sb.st_size);

          if (ret != 0) {
            LOG(ERROR) << "mumap failed for " << (void*)weight_mem_base
                       << " size: " << sb.st_size;
          }
#endif
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }

#ifdef AS_WEIGHT_LOAD_FROM_MMEMAP
        try {
          dense_loader.LoadFromMemory(
              SEEK_PTR_BYTES(weight_mem_base, weight_info.weight_offset),
              weight_info.size_bytes, nullptr, tensor);
        } catch (AsException& e) {
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }

#else

        fseek(fp_ptr.get(), weight_info.weight_offset, SEEK_SET);
        dense_loader.LoadFromFileStream(fp_ptr.get(), tensor);
#endif
      }
      {
        finished++;
        (*tensor_map_for_current_rank)[weight_info.name] = std::move(tensor);
      }
    }

#ifdef AS_WEIGHT_LOAD_FROM_MMEMAP
    int ret = munmap((void*)weight_mem_base, sb.st_size);

    if (ret != 0) {
      LOG(ERROR) << "mumap failed for " << (void*)weight_mem_base
                 << " size: " << sb.st_size;
    }
#endif
  }

  LOG(INFO) << "finish weight load for model " << rank_info << " "
            << "time  spend: " << start.elapsed() / 1000.0f << " seconds.";

  {
    rw_write_lock lk(lock_, "LoadModel");

    if (weight_storage_.count(weight_handler) != 0) {
      // replace the weight and prints some log.
      weight_storage_[weight_handler][rank_info] =
          std::move(tensor_map_for_current_rank);
    } else {
      weights_of_rank_t private_rank_weights;
      private_rank_weights[rank_info] = std::move(tensor_map_for_current_rank);
      weight_storage_[weight_handler] = private_rank_weights;
    }
  }
  // store the tensor map for current rank info, can be access by multiple
  // thread, needs with lock.
  return AsStatus::ALLSPARK_SUCCESS;
}

std::shared_ptr<AsTensor> WeightManagerImpl::GetWeightTensor(
    std::shared_ptr<ModelWeightHandler>& handler, RankInfo& rank_info,
    const std::string& name) {
  rw_read_lock lk(lock_, "GetWeightTensor");

  if (!handler_is_avalibile(handler) ||
      !weight_on_rank_is_avalibile(handler, rank_info)) {
    LOG(ERROR) << "Try to find weight for non exist rank or handler "
               << rank_info
               << " found handler: " << handler_is_avalibile(handler)
               << " avalibile: "
               << weight_on_rank_is_avalibile(handler, rank_info);
    throw AsException("weight get: no such rank");
  }

  auto& weight_map = get_weight_on_rank(handler, rank_info);

  if (weight_map->count(name) == 0) {
    LOG(ERROR) << "Try to find weight for non exist name " << rank_info
               << " name : " << name;
    throw AsException("weight get: no such name");
  }

  return weight_map->at(name);
}

void WeightManagerImpl::CheckModelConsistency(
    std::shared_ptr<ModelWeightHandler> weight_handler) {
  if (proto_store_.count(weight_handler->GetId()) == 0)
    throw AsException("unknown proto");
  if (!proto_store_[weight_handler->GetId()]->has_build_meta()) return;

  std::unique_ptr<FILE, int (*)(FILE*)> fp_ptr(
      fopen(weight_handler->GetModelConfig().weights_path.c_str(), "rb"),
      fclose);
  // todo: needs to return when file it's sparse.
  int fd = fileno(fp_ptr.get());
  struct stat sb;
  fstat(fd, &sb);

  // multiple thread can share this file mapping
  // mmap will start with the current file's seek poisition, not beginning
  // of the file.
  const char* weight_mem_base =
      (const char*)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (weight_mem_base == MAP_FAILED) {
    LOG(ERROR) << "weight load map failed:  "
               << " error: " << strerror(errno);

    throw AsModelException("mmap failed");
  }

  AsParamGuard guard;
  const BuildMetaProto& build_meta =
      proto_store_[weight_handler->GetId()]->build_meta();
  if (sb.st_size >= build_meta.weight_hash().hash_length(0)) {
    guard.append_weight_md5(
        weight_mem_base,
        static_cast<size_t>(build_meta.weight_hash().hash_length(0)));
  }

  guard.torch_build_config.insert(std::make_pair(
      "model_name", weight_handler->GetModelConfig().model_name));
  AsStatus check_status = guard(build_meta);

  int ret = munmap((void*)weight_mem_base, sb.st_size);

  if (ret != 0) {
    LOG(ERROR) << "mumap failed for " << (void*)weight_mem_base
               << " size: " << sb.st_size;

    if (check_status != AsStatus::ALLSPARK_SUCCESS) {
      throw AsModelException("model check failed.");
    }
  }
}

void WeightManagerImpl::SaveWeights(std::shared_ptr<ModelWeightHandler> handler,
                                    std::string* out_allsparkz) {
  WeightSerialization weight_saver;
  if (!handler_is_avalibile(handler)) {
    LOG(ERROR) << "Try to find weight for non exist rank or handler "
               << " found handler: " << handler_is_avalibile(handler);
    throw AsException("weight get: handler");
  }

  auto tensor_map =
      get_weight_on_rank(handler, weight_storage_[handler].begin()->first);

  weight_saver.SerializeMultipleTensor(*tensor_map, out_allsparkz);
}

bool WeightManagerImpl::SeekToNextTensor(FILE* fp, TensorInfo& info) {
  DataMode& mode = info.mode;
  Shape& shape = info.shape;
  DataType& dtype = info.dtype;
  int nnz = info.nnz;
  if (mode == DataMode::DENSE) {
    int64_t len = shape.Count() * SizeofType(dtype);
    auto ret = fseek(fp, len, SEEK_CUR);
    if (ret != 0) {
      LOG(ERROR) << "fseek error, len: " << len << std::endl;
      return false;
    }
    return true;
  } else {
    // sparse tensor
    switch (mode) {
      case DataMode::CSC: {
        int cols = (int)shape[1];
        auto ret0 = fseek(fp, (cols + 1) * sizeof(int), SEEK_CUR);
        auto ret1 = fseek(fp, (nnz) * sizeof(int), SEEK_CUR);
        auto ret2 = fseek(fp, (nnz)*SizeofType(dtype), SEEK_CUR);
        if (ret0 != 0 || ret1 != 0 || ret1 != 0) {
          LOG(ERROR) << "fseek error, DataMode::CSC" << std::endl;
          return false;
        }
        break;
      }
      case DataMode::ELL: {
        auto ret0 = fseek(fp, (nnz) * sizeof(unsigned short), SEEK_CUR);
        auto ret1 = fseek(fp, (nnz)*SizeofType(dtype), SEEK_CUR);
        if (ret0 != 0 || ret1 != 0) {
          LOG(ERROR) << "fseek error, DataMode::ELL" << std::endl;
          return false;
        }
        break;
      }
      default:
        LOG(ERROR) << "invalid data mode in allsparky format" << std::endl;
        return false;
    }
  }
  return false;
}

// inplace replace all the tensor from device tensor to host tensor.
void WeightManagerImpl::DuplicateTensorsToDeviceType(
    weights_of_rank_t& weight_on_handler, DeviceType target_type,
    const RankInfo* rank_info_p) {
  for (auto& weight_on_rank : weight_on_handler) {
    // for each rank's tensor map
    auto tensor_map = weight_on_rank.second;

    if (rank_info_p != nullptr && !(*rank_info_p == weight_on_rank.first)) {
      continue;
    }

    auto new_map = std::make_shared<TensorMap>();
    weight_on_rank.second = new_map;

    for (auto tm_it = tensor_map->begin(); tm_it != tensor_map->end();
         ++tm_it) {
      auto src_tensor = tm_it->second;
      auto map_name = tm_it->first;

      int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);

      auto target_tensor = std::make_shared<AsTensor>(
          src_tensor->GetName(), target_type, src_tensor->GetDataType(),
          src_tensor->GetDataMode(), src_tensor->GetShape(), flags);

      TensorUtils::DeepCopyWhole(*target_tensor, *src_tensor);

      new_map->emplace(std::make_pair(map_name, target_tensor));
    }
  }
}

// -------------------------------------------------------
// Manager Class
// -------------------------------------------------------

std::shared_ptr<ModelWeightHandler> WeightManager::RegisterModel(
    AsModelConfig& config, std::shared_ptr<TransformerProto> model_ir) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->RegisterModel(config, model_ir);
}

AsStatus WeightManager::LoadWeightForModel(
    const DeviceContext& target_device_ctx,
    std::shared_ptr<ModelWeightHandler>& weight_handler, RankInfo& rank_info) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->LoadWeightForModel(target_device_ctx, weight_handler, rank_info);
}

std::shared_ptr<AsTensor> WeightManager::GetWeightTensor(
    std::shared_ptr<ModelWeightHandler>& handler, RankInfo& rank_info,
    const std::string& name) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->GetWeightTensor(handler, rank_info, name);
}

std::shared_ptr<WeightManager> WeightManager::Create() {
  std::shared_ptr<WeightManager> child_ptr =
      std::make_shared<WeightManagerImpl>();
  return std::static_pointer_cast<WeightManager>(child_ptr);
}

void WeightManager::SaveWeights(std::shared_ptr<ModelWeightHandler> handler,
                                std::string* out_allsparkz) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->SaveWeights(handler, out_allsparkz);
}

void WeightManager::CheckModelConsistency(
    std::shared_ptr<ModelWeightHandler> weight_handler) {
  WeightManagerImpl* impl = GetImpl(this);
  impl->CheckModelConsistency(weight_handler);
}
};  // namespace allspark
