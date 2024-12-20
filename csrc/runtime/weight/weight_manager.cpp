/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_manager.cpp
 */

#include "weight_manager.h"

#include <common/device_context.h>
#include <common/env_config.h>
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
#include <utility/progress_bar.hpp>

#include "weight_loader.h"
#include "weight_saver.h"
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

#define DEBUG_SWAP 0
#define DEBUG_SWAP_MD5 0

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

int WeightManagerImpl::GetNumModels() {
  rw_write_lock lk(lock_, "GetNumModels");
  return weight_handler_store_.size();
}

static inline void SetCUDADeviceOptional(
    const DeviceContext& target_device_ctx) {
#ifdef ENABLE_CUDA
  if (target_device_ctx.GetDeviceType() == DeviceType::CUDA) {
    DeviceContext* ctx = const_cast<DeviceContext*>(&target_device_ctx);
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(ctx);
    cudaSetDevice(cuda_ctx->GetDeviceId());
  }
#endif
}

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

  if (GetSwapStatus(weight_handler) == SwapStatus::SwapIn) {
    LOG(INFO) << "Ignore Load model from file, since already swap in";
    return AsStatus::ALLSPARK_SUCCESS;
  }

  TensorMap* model_weights_buffer;  // for swap, add later
  std::vector<ModelWeightAccessInfo> processOrder;
  WeightFileParser weight_parser;

  int finished = 0;

  unsigned long total_loaded_bytes = 0;

  util::Timer start_weight_load;
  auto tensor_map_for_current_rank = std::make_shared<TensorMap>();
  const bool use_mmap =
      WeightManagerEnvConfig::IsWeightLoadingFromMmapEnabled();

  if (!use_mmap) {
    LOG(INFO) << "Model loaded in fread mode, switch to mmap by set env "
                 "AS_WEIGHT_LOAD_FROM_MMAP = on";
  } else {
    LOG(INFO) << "Model loaded in mmap mode, switch to fread mode by set env "
                 "AS_WEIGHT_LOAD_FROM_MMAP = off";
  }
  LOG(INFO) << "Start open model file "
            << weight_handler->GetModelConfig().weights_path;
  std::unique_ptr<FILE, int (*)(FILE*)> fp_ptr(
      fopen(weight_handler->GetModelConfig().weights_path.c_str(), "rb"),
      fclose);
  // close the fd when out of scope.
  FILE* fp = fp_ptr.get();

  LOG(INFO) << "Open model file success. ";
  LOG(INFO) << "Start Loading weights. ";
  while (true) {
    std::vector<char> local_header(6);

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
    ModelWeightAccessInfo weight_info;
    weight_info.name = varname;

    WeightFileParser weight_header_parser;
    TensorInfo info = weight_header_parser.ParseTensorInfo(fp);
    weight_info.size_bytes = SizeofType(info.dtype) * info.shape.Count();
    weight_info.weight_offset = ftell(fp);
    weight_info.info = info;

    // before proceeding, we need validate the weight according some rules
    AsStatus ret =
        ValidateWeight(weight_handler, weight_info, target_device_ctx);
    if (ret != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "ValidateWeight failed for tensor " << weight_info.name
                 << ", fail reason=" << int(ret);
      return ret;
    }

    int fd = fileno(fp);
    struct stat sb;
    fstat(fd, &sb);
    const char* weight_mem_base = nullptr;
    if (use_mmap) {
      weight_mem_base = static_cast<const char*>(
          mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0));
      if (weight_mem_base == MAP_FAILED) {
        LOG(ERROR) << "weight load map failed:  "
                   << " error: " << strerror(errno);
        continue;
      }
    }
    // start load weight
    SetCUDADeviceOptional(target_device_ctx);
    std::shared_ptr<AsTensor> tensor = std::make_unique<AsTensor>(
        weight_info.name, target_device_ctx.GetDeviceType());

    tensor->SetDataType(weight_info.info.dtype);
    tensor->SetDataMode(weight_info.info.mode);
    AS_CHECK_STATUS(tensor->SetShape(Shape(weight_info.info.shape)));

    if (weight_info.info.mode != DataMode::DENSE) {
      SparseWeightLoader sparse_loader(weight_info.info, rank_info,
                                       tensor->GetName());
      sparse_loader.LoadFromFileStream(fp_ptr.get(), tensor);
    } else {
      DenseWeightLoader dense_loader(weight_info.info, rank_info,
                                     tensor->GetName(), nullptr);

      total_loaded_bytes += (weight_info.size_bytes / rank_info.rank_size);

      if (weight_info.weight_offset + weight_info.size_bytes >= sb.st_size) {
        LOG(ERROR) << "file weight memory less than weight size, "
                      "something wrong.";

        if (use_mmap) {
          int ret = munmap((void*)weight_mem_base, sb.st_size);

          if (ret != 0) {
            LOG(ERROR) << "mumap failed for " << (void*)weight_mem_base
                       << " size: " << sb.st_size;
          }
        }
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }

      if (use_mmap) {
        try {
          dense_loader.LoadFromMemory(
              SEEK_PTR_BYTES(weight_mem_base, weight_info.weight_offset),
              weight_info.size_bytes, nullptr, tensor);
        } catch (AsException& e) {
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }
      } else {
        dense_loader.LoadFromFileStream(fp_ptr.get(), tensor);
      }
    }
    {
      finished++;
      std::cout << "." << std::flush;

      (*tensor_map_for_current_rank)[weight_info.name] = std::move(tensor);
#if DEBUG_SWAP
      DLOG(INFO) << "weight process progress "
                 << " ( " << finished << " / " << processOrder.size() << " ) ";
#endif
    }
    processOrder.push_back(std::move(weight_info));
  }

#if 0  // XXX: this reorder may cause loading performance issue on nas like
       // device.
  // sort the access order in big little order to avoid max gpu memory usage
  // exceed gpu size.
  const bool use_mmap =
      WeightManagerEnvConfig::IsWeightLoadingFromMmapEnabled();
 
  if (use_mmap) {
    std::stable_sort(
        processOrder.begin(), processOrder.end(),
        [](const ModelWeightAccessInfo& lhs, const ModelWeightAccessInfo& rhs) {
          return lhs.size_bytes > rhs.size_bytes;
        });
  }
#endif
  std::cout << std::endl << std::flush;
  LOG(INFO) << "Weight file header parse success... " << processOrder.size()
            << " weight tensors are going to load. ";

  float size_in_mb = (total_loaded_bytes / (1024.0 * 1024.0));
  float speed_in_mb = size_in_mb / start_weight_load.elapsed_seconds();

  LOG(INFO) << "Finish weight load for model " << rank_info
            << " time: " << start.elapsed_seconds()
            << " seconds, size: " << size_in_mb
            << " MiB, speed: " << speed_in_mb << " (MiB/s)";

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

    // if swap enable, change do the host copy.
    // Note: because some aglinment op will modify weight's size, so we can
    // only store the tensor tensor after load, otherwise some shape logic
    // will be wrong.
    if (weight_handler->GetModelConfig().is_lora_cfg) {
      swap_status_[weight_handler][rank_info] = SwapStatus::SwapInit;
      LOG(INFO) << "finish load lora "
                << weight_handler->GetModelConfig().model_name << " for rank "
                << rank_info.rank_id << "/" << rank_info.rank_size;

    } else if (IsSwapEnable(weight_handler)) {
      LOG(INFO) << "Weight Swap config is enable, start save device "
                   "memory tensor to host memory.";

      util::Timer swap_start;
      if (swap_weight_storage_.count(weight_handler) == 0 &&
          weight_storage_[weight_handler].size() == rank_info.rank_size) {
        LOG(INFO) << "Weight swap: last rank's weight already loaded, start "
                     "copy to host.";
        swap_weight_storage_[weight_handler] = weight_storage_[weight_handler];
        DuplicateTensorsToDeviceType(swap_weight_storage_[weight_handler],
                                     DeviceType::CPU, nullptr);
      } else {
        LOG(INFO) << "Ignore rank " << rank_info.rank_id
                  << "swap on host because current loading "
                  << weight_storage_[weight_handler].size() << "/"
                  << rank_info.rank_size;
      }

      swap_status_[weight_handler][rank_info] = SwapStatus::SwapInit;

      LOG(INFO) << "finish weight swap prepare for model "
                << weight_handler->GetModelConfig().model_name
                << " swap weight size:" << swap_weight_storage_.size()
                << " time  spend: " << swap_start.elapsed() / 1000.0f
                << " seconds.";
    }
  }
  // store the tensor map for current rank info, can be access by multiple
  // thread, needs with lock.

  for (auto& callback : weight_event_callback) {
    callback(weight_handler, WeightEvent::WeightOnLoad);
  }

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

  if (handler_is_swapout(handler, rank_info)) {
    throw AsException("access swap out tensor");
  }

  auto& weight_map = get_weight_on_rank(handler, rank_info);

  if (weight_map->count(name) == 0) {
    LOG(ERROR) << "Try to find weight for non exist name " << rank_info
               << " name : " << name;
    throw AsException("weight get: no such name");
  }

#if DEBUG_SWAP
  // DLOG(INFO) << "Weight MD5: " << name << " " << rank_info << " "
  //            << weight_map->at(name)->GetMD5Sum();

  // DLOG(INFO) << "Weight MD5: " << name << " "
  //            << weight_map->at(name)->ToString();
#endif
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
  if (!build_meta.has_weight_hash()) {
    LOG(INFO) << "skip model consistency check, no weight hash.";
    return;
  }
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
  if (target_type == DeviceType::CUDA && rank_info_p == nullptr) {
    LOG(ERROR) << "Error: Target device is cuda require a rank info "
                  "information, otherwise tensor will allcate to wrong card.";
  }
  for (auto& weight_on_rank : weight_on_handler) {
    // for each rank's tensor map
    auto tensor_map = weight_on_rank.second;

#if DEBUG_SWAP
    DLOG(INFO) << "duplicate tensor for device: " << target_type
               << " rank: " << weight_on_rank.first;
#endif
    if (rank_info_p != nullptr && !(*rank_info_p == weight_on_rank.first)) {
#if DEBUG_SWAP
      DLOG(INFO) << "skip duplication for this rank because target rank"
                 << *rank_info_p << " and this rank: " << weight_on_rank.first
                 << " are not same.";
#endif
      continue;
    }

    auto new_map = std::make_shared<TensorMap>();
    weight_on_rank.second = new_map;

    for (auto tm_it = tensor_map->begin(); tm_it != tensor_map->end();
         ++tm_it) {
      auto src_tensor = tm_it->second;
      auto map_name = tm_it->first;

      int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);

#ifdef ENABLE_CUDA_PINNED_WEIGHT
      if (target_type == DeviceType::CPU)
        flags |= static_cast<int32_t>(AsTensorFlags::cuda_pinned_mem);
#endif
      auto target_tensor = std::make_shared<AsTensor>(
          src_tensor->GetName(), target_type, src_tensor->GetDataType(),
          src_tensor->GetDataMode(), src_tensor->GetShape(), flags);

#if DEBUG_SWAP
#if DEBUG_SWAP_MD5
      LOG(INFO) << " before copy target md5: " << target_tensor->GetMD5Sum();
#endif
      LOG(INFO) << " target tensor before : " << target_tensor->ToString()
                << " name in map: " << map_name;
#endif
      TensorUtils::DeepCopyWhole(*target_tensor, *src_tensor);

#if DEBUG_SWAP
#if DEBUG_SWAP_MD5
      LOG(INFO) << "duplication src: " << src_tensor->GetMD5Sum() LOG(INFO)
                << "duplication src: " << src_tensor->GetMD5Sum()
                << " target md5: " << target_tensor->GetMD5Sum();
#endif
      LOG(INFO) << "src: " << *src_tensor << std::endl
                << " dst: " << *target_tensor;
#endif
      new_map->emplace(std::make_pair(map_name, target_tensor));
    }
  }
}

void WeightManagerImpl::SwapOutWeight(
    std::shared_ptr<ModelWeightHandler>& handler, RankInfo info) {
  // make a copy of this handler's tensor of every rank,
  // make the tensor to device tensor.
  //
  // make a new status of those tensor already swap out for eaiser debug.

  util::Timer start;

  LOG(INFO) << "WeightManager: swap out model: "
            << handler->GetModelConfig().model_name;

  {
    rw_read_lock lk(lock_, "SwapOutWeight:1");
    if (!IsSwapEnable(handler)) {
      LOG(ERROR) << "Try to swap out weight not enable swap function, try "
                    "model config swap_threshold=1 to enable function ";
      throw AsException("try enable swap without enable config.");
    }
    try {
      if (swap_status_.at(handler).at(info) == SwapStatus::SwapOut) {
        DLOG(INFO) << "ignore weight swap out, already swap out...";
        return;
      }
    } catch (std::out_of_range& e) {
      LOG(ERROR) << "out of range on swap status." << e.what();
      return;
    }
    if (!handler_is_avalibile(handler)) {
      LOG(ERROR) << "Try to swap out weight which not in manager";
      throw AsException("unknown weight handler");
    }
  }

  {
    rw_write_lock lk(lock_, "SwapOutWeight:2");
    // free up the weight on device memory
    // loop through the weight and free them.
    weight_storage_[handler].erase(info);

    swap_status_[handler][info] = SwapStatus::SwapOut;
  }

  for (auto& callback : weight_event_callback) {
    callback(handler, WeightEvent::WeightOnSwapOut);
  }
  LOG(INFO) << "finish swap out model " << handler->GetModelConfig().model_name
            << " swap map size: " << swap_weight_storage_.size()
            << "rank : " << info << "time: " << start.elapsed() / 1000.0f
            << " seconds.";
}

void WeightManagerImpl::SwapInWeight(
    std::shared_ptr<ModelWeightHandler>& handler, RankInfo rankInfo) {
  util::Timer start;
  LOG(INFO) << "WeightManager: swap in model: "
            << handler->GetModelConfig().model_name;
  {
    rw_read_lock lk(lock_, "SwapInWeight:1");
    if (!IsSwapEnable(handler)) {
      LOG(ERROR) << "Try to swap in weight not enable swap function, try "
                    "model config swap_threshold=1 to enable function ";
      throw AsException("try enable swap without enable config.");
    }
    if (swap_status_.count(handler) == 0) {
      LOG(ERROR) << "error: try to swap in a weight not have been swap out.";
      return;
    }
    try {
      if (swap_status_.at(handler).at(rankInfo) == SwapStatus::SwapIn) {
        LOG(INFO) << "ignore weight swap in, already swap in...";
        return;
      }
    } catch (std::out_of_range& e) {
      LOG(ERROR) << "out of range on swap status." << e.what();
      return;
    }

    if (!handler_swap_in_is_avalibile(handler)) {
      LOG(ERROR) << "Try to swap in weight which is not swap out";
      throw AsException("unknown weight handler");
    }
  }

  {
    rw_write_lock lk(lock_, "SwapInWeight:2");

    LOG(INFO) << "start swap in weight in " << rankInfo;
    weight_storage_[handler][rankInfo] =
        swap_weight_storage_[handler][rankInfo];

    DuplicateTensorsToDeviceType(weight_storage_[handler], DeviceType::CUDA,
                                 &rankInfo);

    swap_status_[handler][rankInfo] = SwapStatus::SwapIn;
  }
  LOG(INFO) << "WeightManager: swap in: finish swap in model "
            << handler->GetModelConfig().model_name << " " << rankInfo
            << " time: " << start.elapsed() / 1000.0f << " seconds.";

  for (auto& callback : weight_event_callback) {
    callback(handler, WeightEvent::WeightOnSwapIn);
  }
}

void WeightManagerImpl::FreeSwapResource(
    std::shared_ptr<ModelWeightHandler>& handler) {
  LOG(INFO) << "WeightManager: free swap resource of : "
            << handler->GetModelConfig().model_name;
  rw_write_lock lk(lock_);

  swap_weight_storage_.erase(handler);
}

SwapStatus WeightManagerImpl::GetSwapStatus(
    std::shared_ptr<ModelWeightHandler>& handler) {
  rw_read_lock lk(lock_, "GetSwapStatus");
  if (swap_status_.count(handler) > 0) {
    return swap_status_[handler].begin()->second;
  } else {
    return SwapStatus::SwapInit;
  }
}

void WeightManagerImpl::SetSwapConfig(
    std::shared_ptr<ModelWeightHandler>& handler,
    WeightSwapConfig swap_config) {
  rw_write_lock lk(lock_, "SetSwapConfig");

  swap_config_[handler] = std::move(swap_config);
}

void WeightManagerImpl::FreeWeight(
    std::shared_ptr<ModelWeightHandler> handler) {
  rw_write_lock lk(lock_, "FreeWeight");

  LOG(INFO) << "WeightManager: FreeWeight Start, current weight number: "
            << weight_storage_.size();
  if (weight_storage_.find(handler) != std::end(weight_storage_)) {
    weight_storage_.erase(handler);
    weight_handler_store_.erase(
        std::remove(weight_handler_store_.begin(), weight_handler_store_.end(),
                    handler),
        weight_handler_store_.end());
  }

  LOG(INFO) << "WeightManager: FreeWeight finished, current weight number: "
            << weight_storage_.size();
  for (auto& callback : weight_event_callback) {
    callback(handler, WeightEvent::WeightOnFree);
  }
}

bool WeightManagerImpl::IsSwapEnable(
    std::shared_ptr<ModelWeightHandler>& handler) {
#ifdef ENABLE_CUDA
  if (swap_config_.count(handler) == 0) {
    return false;
  } else {
    return swap_config_[handler].enable;
  }
#else
  return false;
#endif
}

void WeightManagerImpl::RegisterWeightEventListener(
    WeightEventCallback callback) {
  rw_write_lock lk(lock_, "RegisterWeightEventListener");

  weight_event_callback.push_back(callback);
}

WeightManagerImpl::~WeightManagerImpl() { LOG(INFO) << "~WeightManager"; }

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

void WeightManager::SwapOutWeight(std::shared_ptr<ModelWeightHandler>& handler,
                                  RankInfo info) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->SwapOutWeight(handler, info);
}

void WeightManager::SwapInWeight(std::shared_ptr<ModelWeightHandler>& handler,
                                 RankInfo info) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->SwapInWeight(handler, info);
}

void WeightManager::FreeSwapResource(
    std::shared_ptr<ModelWeightHandler>& handler) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->FreeSwapResource(handler);
}

SwapStatus WeightManager::GetSwapStatus(
    std::shared_ptr<ModelWeightHandler>& handler) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->GetSwapStatus(handler);
}

void WeightManager::SetSwapConfig(std::shared_ptr<ModelWeightHandler>& handler,
                                  WeightSwapConfig swap_config) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->SetSwapConfig(handler, swap_config);
}

void WeightManager::CheckModelConsistency(
    std::shared_ptr<ModelWeightHandler> weight_handler) {
  WeightManagerImpl* impl = GetImpl(this);
  impl->CheckModelConsistency(weight_handler);
}
void WeightManager::FreeWeight(std::shared_ptr<ModelWeightHandler> handler) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->FreeWeight(handler);
}
void WeightManager::RegisterWeightEventListener(WeightEventCallback callback) {
  WeightManagerImpl* impl = GetImpl(this);
  return impl->RegisterWeightEventListener(std::move(callback));
}

};  // namespace allspark
