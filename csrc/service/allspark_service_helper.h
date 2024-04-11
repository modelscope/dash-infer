/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_service_helper.h
 */

#pragma once
#include <allspark.h>
#include <glog/logging.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "allspark_service.grpc.pb.h"

namespace allspark {
namespace allspark_service {
class DLTensorManager;
using SharedDLTensorMap =
    std::map<std::string, std::shared_ptr<DLTensorManager>>;
using SharedDLTensorListMap =
    std::map<std::string, std::vector<std::shared_ptr<DLTensorManager>>>;
class DLTensorManager {
 public:
  DLTensorManager() : dl_managed_tensor_(nullptr) {}
  ~DLTensorManager() {
    if (dl_managed_tensor_) {
      if (dl_managed_tensor_->deleter) {
        dl_managed_tensor_->deleter(dl_managed_tensor_);
        dl_managed_tensor_->deleter = nullptr;
      }
      dl_managed_tensor_ = nullptr;
    }
  }
  void ToDlTensor(const allspark_service::Tensor& tensor) {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
      dl_managed_tensor_->deleter = nullptr;
    }
    dl_managed_tensor_ = new DLManagedTensor();
    // only CPU support now
    dl_managed_tensor_->dl_tensor.device.device_id = 0;
    dl_managed_tensor_->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
    dl_managed_tensor_->dl_tensor.ndim = tensor.shape().dims_size();
    dl_managed_tensor_->dl_tensor.strides = nullptr;
    int shape_count = 1;
    int64_t* shape = new int64_t[tensor.shape().dims_size()];
    for (int i = 0; i < tensor.shape().dims_size(); i++) {
      shape[i] = tensor.shape().dims(i);
      shape_count *= tensor.shape().dims(i);
    }
    dl_managed_tensor_->dl_tensor.shape = shape;
    dl_managed_tensor_->dl_tensor.dtype.lanes = 1;
    dl_managed_tensor_->dl_tensor.byte_offset = 0;
    switch (tensor.data_type()) {
      case allspark_service::DATA_TYPE::FLOAT32:
        dl_managed_tensor_->dl_tensor.dtype.code = DLDataTypeCode::kDLFloat;
        dl_managed_tensor_->dl_tensor.dtype.bits = 32;
        break;
      case allspark_service::DATA_TYPE::FLOAT16:
        dl_managed_tensor_->dl_tensor.dtype.code = DLDataTypeCode::kDLFloat;
        dl_managed_tensor_->dl_tensor.dtype.bits = 16;
        break;
      case allspark_service::DATA_TYPE::INT64:
        dl_managed_tensor_->dl_tensor.dtype.code = DLDataTypeCode::kDLInt;
        dl_managed_tensor_->dl_tensor.dtype.bits = 64;
        break;
      case allspark_service::DATA_TYPE::INT32:
        dl_managed_tensor_->dl_tensor.dtype.code = DLDataTypeCode::kDLInt;
        dl_managed_tensor_->dl_tensor.dtype.bits = 32;
        break;
      default:
        break;
    }
    // copy data
    int data_size = shape_count * dl_managed_tensor_->dl_tensor.dtype.bits / 8;
    // user manage dl tensor, we need malloc data since proto may
    // destroy later
    char* data = new char[data_size];
    memcpy(data, static_cast<const char*>(tensor.data().data()), data_size);
    dl_managed_tensor_->dl_tensor.data = data;
    dl_managed_tensor_->deleter = [](DLManagedTensor* self) {
      if (self) {
        if (self->dl_tensor.shape) {
          delete[] self->dl_tensor.shape;
        }
        if (self->dl_tensor.strides) {
          delete[] self->dl_tensor.strides;
        }
        if (self->manager_ctx) {
          DLTensorManager* p_manager =
              static_cast<DLTensorManager*>(self->manager_ctx);
          if (self->dl_tensor.data) {
            delete[] static_cast<int64_t*>(self->dl_tensor.data);
          }
        }
        delete self;
      }
    };
    dl_managed_tensor_->manager_ctx = this;
  }

  // make tensor proto from std::vector<std::vector<int64_t>>
  void ToTensorProto(allspark_service::Tensor& tensor,
                     DLManagedTensor* dl_tensor) {
    toTensorProtoFromDlTensor(tensor, dl_tensor);
  }

  void ToTensorProto(allspark_service::Tensor& tensor,
                     const std::vector<std::vector<int64_t>>& input_vec) {
    toTensorProtoFromVec(tensor, input_vec);
  }

  DLManagedTensor* GetDlTensor() { return dl_managed_tensor_; }

  void ToVectorData(std::vector<std::vector<int64_t>>& output,
                    DLManagedTensor* dl_managed_tensor) {
    assert(dl_managed_tensor && dl_managed_tensor->dl_tensor.ndim == 2);
    // set data
    for (int i = 0; i < dl_managed_tensor->dl_tensor.shape[0]; i++) {
      std::vector<int64_t> out(dl_managed_tensor->dl_tensor.shape[1]);
      int data_size = dl_managed_tensor->dl_tensor.shape[1] *
                      dl_managed_tensor->dl_tensor.dtype.bits / 8;
      char* data_ptr =
          reinterpret_cast<char*>(dl_managed_tensor->dl_tensor.data) +
          i * data_size;
      memcpy(out.data(), data_ptr, data_size);
      output.push_back(out);
    }
  }

 private:
  // the class should not call deleter function
  DLManagedTensor* dl_managed_tensor_;
  void toTensorProtoFromVec(
      allspark_service::Tensor& tensor,
      const std::vector<std::vector<int64_t>>& inputs_vec) {
    // no use
    tensor.set_name("tensor");
    // shape
    allspark_service::Tensor::Shape* shape = tensor.mutable_shape();
    shape->add_dims(inputs_vec.size());
    shape->add_dims(inputs_vec[0].size());
    int shape_count = inputs_vec.size() * inputs_vec[0].size();
    // only CPU supports now
    allspark_service::DeviceType device_type;
    device_type.set_dev_type(allspark_service::DEVICE_TYPE::CPU);
    tensor.mutable_device_type()->CopyFrom(device_type);
    // only int64_t supports
    tensor.set_data_type(allspark_service::DATA_TYPE::INT64);
    // set data
    for (const auto& row : inputs_vec) {
      tensor.mutable_data()->append(reinterpret_cast<const char*>(row.data()),
                                    row.size() * sizeof(int64_t));
    }
  }
  void toTensorProtoFromDlTensor(allspark_service::Tensor& tensor,
                                 DLManagedTensor* dl_managed_tensor) {
    // no use
    tensor.set_name("tensor");
    // shape
    int shape_count = 1;
    allspark_service::Tensor::Shape* shape = tensor.mutable_shape();
    for (int i = 0; i < dl_managed_tensor->dl_tensor.ndim; i++) {
      shape->add_dims(static_cast<int>(dl_managed_tensor->dl_tensor.shape[i]));
      shape_count *= static_cast<int>(dl_managed_tensor->dl_tensor.shape[i]);
    }
    // only CPU supports now
    allspark_service::DeviceType device_type;
    device_type.set_dev_type(allspark_service::DEVICE_TYPE::CPU);
    tensor.mutable_device_type()->CopyFrom(device_type);
    switch (dl_managed_tensor->dl_tensor.dtype.code) {
      case DLDataTypeCode::kDLFloat:
        switch (dl_managed_tensor->dl_tensor.dtype.bits) {
          case 16:
            tensor.set_data_type(allspark_service::DATA_TYPE::FLOAT16);
            break;
          case 32:
            tensor.set_data_type(allspark_service::DATA_TYPE::FLOAT32);
            break;
        }
        break;
      case kDLUInt:
        switch (dl_managed_tensor->dl_tensor.dtype.bits) {
          case 1:
            tensor.set_data_type(allspark_service::DATA_TYPE::BOOL);
            break;
        }
        break;
      case kDLInt:
        switch (dl_managed_tensor->dl_tensor.dtype.bits) {
          case 8:
            tensor.set_data_type(allspark_service::DATA_TYPE::INT8);
            break;
          case 16:
            tensor.set_data_type(allspark_service::DATA_TYPE::INT16);
            break;
          case 32:
            tensor.set_data_type(allspark_service::DATA_TYPE::INT32);
            break;
          case 64:
            tensor.set_data_type(allspark_service::DATA_TYPE::INT64);
            break;
        }
        break;
      default:
        tensor.set_data_type(allspark_service::DATA_TYPE::DATATYPE_UNDEFINED);
        break;
    }
    // set data
    char* data_ptr = reinterpret_cast<char*>(dl_managed_tensor->dl_tensor.data);
    int data_size = shape_count * dl_managed_tensor->dl_tensor.dtype.bits / 8;
    tensor.set_data(reinterpret_cast<const char*>(data_ptr), data_size);
  }
};

void as_rpc_init_log(const char* tag) {
  google::InitGoogleLogging(tag);
  google::InstallFailureSignalHandler();
  google::EnableLogCleaner(3);

  fLB::FLAGS_timestamp_in_logfile_name = true;
  fLB::FLAGS_alsologtostderr = false;
  fLI::FLAGS_stderrthreshold = google::ERROR;
  fLI::FLAGS_logbuflevel = google::WARNING;
  fLI::FLAGS_logbufsecs = 5;
  fLI::FLAGS_max_log_size = 10;

  const char* log_dir = std::getenv("HIE_LOG_DIR");
  if (not log_dir or std::string(log_dir) == "") {
    fLB::FLAGS_logtostderr = true;
  } else {
    fLS::FLAGS_log_dir = log_dir;
    fLB::FLAGS_logtostderr = false;
  }

  const char* log_level_str = std::getenv("HIE_LOG_LEVEL");
  int log_level = 0;
  if (log_level_str) {
    log_level = atoi(log_level_str);
    log_level = (google::INFO <= log_level and log_level <= google::FATAL)
                    ? log_level
                    : 0;
  }
  fLI::FLAGS_minloglevel = log_level;
}

// make as shared DLTensorManager from proto RunModelParams
void makeInputMapAsFromProto(SharedDLTensorMap& out,
                             const allspark::allspark_service::TensorMap& in) {
  for (auto& in_proto : in.tensor_map()) {
    auto tensor_manage = std::make_shared<DLTensorManager>();
    tensor_manage->ToDlTensor(in_proto.second);
    out.insert(make_pair(in_proto.first, tensor_manage));
  }
}

// make share ptr to ptr
void makeInputMapFromSharedMap(allspark::DLTensorMap& outs,
                               const SharedDLTensorMap& ins) {
  for (auto& in : ins) {
    outs.insert(make_pair(in.first, in.second->GetDlTensor()));
  }
}

// make as GenerateConfig/bad_words_ids from proto
void makeInputCfgAsFromProto(
    allspark::GenerateConfig& gen_cfg,
    const allspark::allspark_service::StartRequestRequest& start_param_proto) {
#define PROTO_CONFIG(src_cfg, config_name, target_config) \
  target_config.config_name = src_cfg.config_name()

  auto& gen_cfg_proto = start_param_proto.config();

  PROTO_CONFIG(gen_cfg_proto, num_beams, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, num_return_sequences, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, temperature, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, do_sample, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, early_stopping, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, repetition_penalty, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, length_penalty, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, min_length, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, max_length, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, no_repeat_ngram_size, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, eos_token_id, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, uuid, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, presence_penalty, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, suppress_repetition_in_generation, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, input_len, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, top_k, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, seed, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, top_p, gen_cfg);

  // bad_words_ids
  std::vector<std::vector<int>> bad_words_ids;
  auto& bad_word_ids_proto = gen_cfg_proto.bad_words_ids();
  for (const auto& array : bad_word_ids_proto.ids()) {
    std::vector<int> array_vector(array.word().begin(), array.word().end());
    bad_words_ids.push_back(std::move(array_vector));
  }
  gen_cfg.bad_words_ids = bad_words_ids;

  // stop_words_ids
  auto& stop_word_ids_proto = gen_cfg_proto.stop_words_ids();
  for (const auto& array : stop_word_ids_proto.ids()) {
    std::vector<int64_t> array_vector(array.word().begin(), array.word().end());
    gen_cfg.stop_words_ids.push_back(std::move(array_vector));
  }
}

// make AS std::vector<SharedDLTensorListMap> from proto
void makeInputExtraAsFromProto(
    std::vector<SharedDLTensorListMap>& extra,
    const allspark::allspark_service::StartRequestRequest& in) {
  for (const auto& tensor_list_map : in.mm_embedding()) {
    SharedDLTensorListMap input_map;
    for (auto& in_array_proto : tensor_list_map.tensor_list_map()) {
      std::vector<std::shared_ptr<DLTensorManager>> dl_tensors;
      for (auto& tensor : in_array_proto.second.tensor()) {
        auto tensor_manage = std::make_shared<DLTensorManager>();
        tensor_manage->ToDlTensor(tensor);
        dl_tensors.push_back(tensor_manage);
      }
      input_map[in_array_proto.first] = std::move(dl_tensors);
    }
    extra.push_back(std::move(input_map));
  }
}

// make share ptr to ptr
void makeInputVecMapFromSharedVecMap(
    std::vector<allspark::DLTensorListMap>& outs,
    const std::vector<SharedDLTensorListMap>& ins) {
  for (auto& in : ins) {
    allspark::DLTensorListMap input_map;
    for (auto& tensor_map : in) {
      std::vector<DLManagedTensor*> dl_tensors;
      for (auto& dl_tensor : tensor_map.second) {
        dl_tensors.push_back(dl_tensor->GetDlTensor());
      }
      input_map[tensor_map.first] = std::move(dl_tensors);
    }
    outs.push_back(std::move(input_map));
  }
}

// make ModelStructConfig proto from AS ModelStructConfig
void makeModelStructConfigProtoFromAs(
    allspark_service::ModelStructConfig& model_struct_proto,
    const allspark::AsModelConfig& model_config) {
  model_struct_proto.set_model_name(model_config.model_name);
  model_struct_proto.set_model_path(model_config.model_path);
  model_struct_proto.set_weights_path(model_config.weights_path);
  model_struct_proto.set_compute_unit(model_config.compute_unit);
  model_struct_proto.set_matmul_precision(model_config.matmul_precision);
  model_struct_proto.set_engine_max_length(model_config.engine_max_length);
  model_struct_proto.set_engine_max_batch(model_config.engine_max_batch);
  model_struct_proto.set_cache_mode(
      static_cast<allspark::allspark_service::AsCacheMode>(
          model_config.cache_mode));
  model_struct_proto.set_prefill_mode(
      static_cast<allspark::allspark_service::AsMHAPrefill>(
          model_config.prefill_mode));
  model_struct_proto.set_text_graph(model_config.text_graph);
  model_struct_proto.set_num_threads(model_config.num_threads);
}

// make AS ModelStructConfig from ModelStructConfig proto
allspark::AsModelConfig makeModelStructConfigAsFromProto(
    const allspark::allspark_service::ModelStructConfig& struct_config) {
  allspark::AsModelConfigBuilder builder;
  auto as_model_config =
      builder.withModelName(struct_config.model_name())
          .withModelPath(struct_config.model_path())
          .withWeightsPath(struct_config.weights_path())
          .withComputeUnit(struct_config.compute_unit())
          .withEngineMaxLength(struct_config.engine_max_length())
          .withEngineMaxBatch(struct_config.engine_max_batch())
          .withTextGraph(struct_config.text_graph())
          .withNumThreads(struct_config.num_threads())
          .withMatmulPrecision(struct_config.matmul_precision())
          .withCacheMode(
              static_cast<allspark::AsCacheMode>(struct_config.cache_mode()))
          .withPrefillMode(
              static_cast<allspark::AsMHAPrefill>(struct_config.prefill_mode()))
          .build();
  return as_model_config;
}

// make TensorListMap proto from AS DLTensorListMap
void makeTensorListMapProtoFromAs(allspark_service::TensorListMap& outs_proto,
                                  const allspark::DLTensorListMap& ins_as) {
  for (auto& dl_tensor_map : ins_as) {
    auto array = (*outs_proto.mutable_tensor_list_map())[dl_tensor_map.first];
    for (auto& dl_tensor : dl_tensor_map.second) {
      auto tensor_proto = array.add_tensor();
      auto tensor_manage = std::make_shared<DLTensorManager>();
      tensor_manage->ToTensorProto(*tensor_proto, dl_tensor);
    }
  }
}

// make GenerateConfig proto from AS GenerateConfig
void makeGenCfgProtoFromAs(allspark_service::GenerateConfig& gen_cfg_proto,
                           const allspark::GenerateConfig& gen_cfg) {
#undef PROTO_CONFIG
#define PROTO_CONFIG(target_cfg, config_name, src_cfg) \
  target_cfg.set_##config_name(src_cfg.config_name)
  PROTO_CONFIG(gen_cfg_proto, num_beams, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, num_return_sequences, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, temperature, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, do_sample, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, early_stopping, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, repetition_penalty, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, length_penalty, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, min_length, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, max_length, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, no_repeat_ngram_size, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, eos_token_id, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, uuid, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, presence_penalty, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, suppress_repetition_in_generation, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, input_len, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, top_k, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, seed, gen_cfg);
  PROTO_CONFIG(gen_cfg_proto, top_p, gen_cfg);

  // bad_words_ids
  auto bad_word_ids_proto = gen_cfg_proto.mutable_bad_words_ids();
  for (auto& bad_word_vec : gen_cfg.bad_words_ids) {
    auto array = bad_word_ids_proto->add_ids();
    for (auto& word_id : bad_word_vec) {
      array->add_word(word_id);
    }
  }

  // stop_words_ids
  auto stop_words_ids_proto = gen_cfg_proto.mutable_stop_words_ids();
  for (auto& stop_word_vec : gen_cfg.stop_words_ids) {
    auto array = stop_words_ids_proto->add_ids();
    for (auto& word_id : stop_word_vec) {
      array->add_word(word_id);
    }
  }
}

// make TensorMap proto from AS DLTensorMap
template <typename T>
void makeTensorMapProtoFromAs(allspark_service::TensorMap& out_tensor_proto,
                              const T& inputs) {
  auto tensor_map_proto = out_tensor_proto.mutable_tensor_map();
  for (auto& dl_tensor : inputs) {
    allspark_service::Tensor tensor_proto;
    auto tensor_manage = std::make_shared<DLTensorManager>();
    tensor_manage->ToTensorProto(tensor_proto, dl_tensor.second);
    (*tensor_map_proto)[dl_tensor.first] = tensor_proto;
  }
}

template void makeTensorMapProtoFromAs<allspark::DLTensorMap>(
    allspark_service::TensorMap& out_tensor_proto,
    const allspark::DLTensorMap& inputs);

template void makeTensorMapProtoFromAs<
    std::map<std::string, std::vector<std::vector<int64_t>>>>(
    allspark_service::TensorMap& out_tensor_proto,
    const std::map<std::string, std::vector<std::vector<int64_t>>>& inputs);

void makeLauchServiceCmd(std::vector<std::string>& cmd, int node_nums,
                         const std::string& daemon_path, int client_pid,
                         int numa_offset = 0) {
  //   const char* args[] = {
  //   "mpirun", "-mca", "plm_rsh_agent", "false", "--allow-run-as-root",
  //   "-n", "1",
  //   "numactl", "-m", "0", "-N", "0", "allspark_daemon", "client_pid",
  //   "rank_id" ":",
  //   "-n", "1",
  //   "numactl", "-m", "1", "-N", "1", "allspark_daemon", "client_pid",
  //   "rank_id" , nullptr
  // };
  cmd.clear();
  cmd.push_back(daemon_path + "/mpirun");
  // allow to run without plm_rsh_agent
  cmd.push_back("-mca");
  cmd.push_back("plm_rsh_agent");
  cmd.push_back("false");
  // allow to run as root
  cmd.push_back("--allow-run-as-root");
  for (int i = 0; i < node_nums; i++) {
    cmd.push_back("-n");
    cmd.push_back("1");
    cmd.push_back("numactl");
    cmd.push_back("-m");
    cmd.push_back(std::to_string(numa_offset + i));
    cmd.push_back("-N");
    cmd.push_back(std::to_string(numa_offset + i));
    cmd.push_back(daemon_path + "/allspark_daemon");
    cmd.push_back(std::to_string(client_pid));
    cmd.push_back(std::to_string(i));
    if (i != node_nums - 1) {
      cmd.push_back(":");
    }
  }
}

// make
void makeGeneratedElementsProtoFromAs(
    allspark_service::GeneratedElements* ele_proto,
    std::shared_ptr<allspark::AsEngine::GeneratedElements>& as_ele) {
  if (as_ele == nullptr) {
    LOG(INFO) << "makeGeneratedElementsProtoFromAs as_ele is nullptr";
    ele_proto->set_empty(1);
    return;
  } else {
    ele_proto->set_empty(0);
  }
  for (auto& id : as_ele->ids_from_generate) {
    ele_proto->add_ids_from_generate(id);
  }
  allspark_service::TensorMap* out_tensor_proto =
      ele_proto->mutable_tensors_from_model_inference();
  makeTensorMapProtoFromAs(*out_tensor_proto,
                           as_ele->tensors_from_model_inference);
}

void makeGeneratedElementsAsFromProto(
    allspark_service::GeneratedElements* ele_proto,
    std::shared_ptr<allspark::AsEngine::GeneratedElements>& as_ele) {
  if (ele_proto->empty() == 1) {
    as_ele = nullptr;
    return;
  }
  for (int i = 0; i < ele_proto->ids_from_generate_size(); ++i) {
    as_ele->ids_from_generate.push_back(ele_proto->ids_from_generate(i));
  }
  // set as dltensor
  for (const auto& tensor_map :
       ele_proto->tensors_from_model_inference().tensor_map()) {
    auto tensor_manage = std::make_shared<DLTensorManager>();
    tensor_manage->ToDlTensor(tensor_map.second);
    as_ele->tensors_from_model_inference[tensor_map.first] =
        tensor_manage->GetDlTensor();
  }
}

void makeRequestParamsAsFromProto(
    std::string& model_name, allspark::AsEngine::RequestContent* req_as,
    const allspark::allspark_service::StartRequestRequest& req_proto,
    allspark_service::SharedDLTensorMap& shared_inputs,
    std::vector<allspark_service::SharedDLTensorListMap>& shared_mm_embedding) {
  makeInputMapAsFromProto(shared_inputs, req_proto.inputs());
  req_as->inputs = std::make_shared<DLTensorMap>();
  makeInputMapFromSharedMap(*req_as->inputs, shared_inputs);
  if (req_proto.mm_embedding_size() > 0) {
    makeInputExtraAsFromProto(shared_mm_embedding, req_proto);
    makeInputVecMapFromSharedVecMap(req_as->mm_embedding, shared_mm_embedding);
  }
  makeInputCfgAsFromProto(req_as->config, req_proto);
  req_as->infer_type =
      static_cast<allspark::AsEngine::RequestInferType>(req_proto.infer_type());
  req_as->mm_type =
      static_cast<allspark::AsEngine::RequestMMType>(req_proto.mm_type());
  model_name = req_proto.model_name();
}

void makeRequestParamsProtoFromAs(
    std::string& model_name, allspark::AsEngine::RequestContent* req_as,
    allspark::allspark_service::StartRequestRequest& req_proto) {
  // set model name
  req_proto.set_model_name(model_name);
  // set infer_type
  req_proto.set_infer_type(
      static_cast<allspark_service::RequestInferType>(req_as->infer_type));
  // set mm_type
  req_proto.set_mm_type(
      static_cast<allspark_service::RequestMMType>(req_as->mm_type));
  // TensorMap
  allspark_service::TensorMap* tensor_map_proto = req_proto.mutable_inputs();
  makeTensorMapProtoFromAs(*tensor_map_proto, *req_as->inputs);
  // TensorListMap
  if (req_as->mm_embedding.size() > 0) {
    for (auto& dllistmap_as : req_as->mm_embedding) {
      allspark_service::TensorListMap* mm_embedding =
          req_proto.add_mm_embedding();
      makeTensorListMapProtoFromAs(*mm_embedding, dllistmap_as);
    }
  }
  // GenerateConfig
  allspark_service::GenerateConfig* gen_cfg_proto = req_proto.mutable_config();
  makeGenCfgProtoFromAs(*gen_cfg_proto, req_as->config);
}

void makeAsEngineStatAsFromProto(
    allspark::AsEngineStat& stat,
    allspark::allspark_service::AsEngineStat& stat_proto) {
#undef PROTO_CONFIG
#define PROTO_CONFIG(src_cfg, config_name, target_config) \
  target_config.config_name = src_cfg.config_name()

  PROTO_CONFIG(stat_proto, model_name, stat);
  PROTO_CONFIG(stat_proto, free_token, stat);
  PROTO_CONFIG(stat_proto, pendding_request, stat);
  PROTO_CONFIG(stat_proto, running_request, stat);
  PROTO_CONFIG(stat_proto, total_device_memory_pool_size, stat);
  PROTO_CONFIG(stat_proto, used_device_memory_pool_size, stat);
  PROTO_CONFIG(stat_proto, total_generated_token, stat);
  PROTO_CONFIG(stat_proto, total_prefill_token, stat);
  PROTO_CONFIG(stat_proto, generate_token_persec, stat);
  PROTO_CONFIG(stat_proto, process_token_persec, stat);
}

void makeAsEngineStatProtoFromAs(
    allspark::allspark_service::AsEngineStat& stat_proto,
    allspark::AsEngineStat& stat) {
#undef PROTO_CONFIG
#define PROTO_CONFIG(target_cfg, config_name, src_cfg) \
  target_cfg.set_##config_name(src_cfg.config_name)

  PROTO_CONFIG(stat_proto, model_name, stat);
  PROTO_CONFIG(stat_proto, free_token, stat);
  PROTO_CONFIG(stat_proto, pendding_request, stat);
  PROTO_CONFIG(stat_proto, running_request, stat);
  PROTO_CONFIG(stat_proto, total_device_memory_pool_size, stat);
  PROTO_CONFIG(stat_proto, used_device_memory_pool_size, stat);
  PROTO_CONFIG(stat_proto, total_generated_token, stat);
  PROTO_CONFIG(stat_proto, total_prefill_token, stat);
  PROTO_CONFIG(stat_proto, generate_token_persec, stat);
  PROTO_CONFIG(stat_proto, process_token_persec, stat);
}

}  // namespace allspark_service
}  // namespace allspark
