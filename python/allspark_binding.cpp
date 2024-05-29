/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_binding.cpp
 */
#include <allspark.h>
#include <allspark_client.h>
#include <allsparkz_util.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <type_traits>

namespace py = pybind11;
using namespace allspark;
using AsClientEngine = allspark::AsClientEngine;
using AsEngine = allspark::AsEngine;
using GeneratedElements = allspark::AsEngine::GeneratedElements;
using ResultQueue = allspark::AsEngine::ResultQueue;
static const char* _c_str_dltensor = "dltensor";
static void _c_dlpack_deleter(PyObject* pycapsule) {
  if (PyCapsule_IsValid(pycapsule, _c_str_dltensor)) {
    void* ptr = PyCapsule_GetPointer(pycapsule, _c_str_dltensor);
    DLManagedTensor* p_dlpack = static_cast<DLManagedTensor*>(ptr);
    p_dlpack->deleter(p_dlpack);
  }
}

#define CHECK_CONFIG(src_cfg, config_name, target_config, type)       \
  if (src_cfg.find(#config_name) != src_cfg.end()) {                  \
    target_config.config_name = (src_cfg[#config_name].cast<type>()); \
  }

static void PyParseConfig(std::map<std::string, py::object>& py_gen_cfg,
                          GenerateConfig& as_gen_cfg) {
  CHECK_CONFIG(py_gen_cfg, num_beams, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, num_return_sequences, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, temperature, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, do_sample, as_gen_cfg, bool);
  CHECK_CONFIG(py_gen_cfg, early_stopping, as_gen_cfg, bool);
  CHECK_CONFIG(py_gen_cfg, repetition_penalty, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, presence_penalty, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, length_penalty, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, suppress_repetition_in_generation, as_gen_cfg, bool);
  CHECK_CONFIG(py_gen_cfg, min_length, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, max_length, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, no_repeat_ngram_size, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, eos_token_id, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, stop_words_ids, as_gen_cfg,
               std::vector<std::vector<int64_t>>);
  CHECK_CONFIG(py_gen_cfg, bad_words_ids, as_gen_cfg,
               std::vector<std::vector<int>>);
  CHECK_CONFIG(py_gen_cfg, logprobs, as_gen_cfg, bool);
  CHECK_CONFIG(py_gen_cfg, top_logprobs, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, top_k, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, top_p, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, seed, as_gen_cfg, unsigned long long);
}
static void PyParseAttribute(std::map<std::string, py::object>& py_attr,
                             TensorAttribute& as_attr) {
  CHECK_CONFIG(py_attr, sparse_type, as_attr, int);
  CHECK_CONFIG(py_attr, split_mode, as_attr, int);
  CHECK_CONFIG(py_attr, shape, as_attr, std::vector<int>);
  CHECK_CONFIG(py_attr, dtype, as_attr, char);
  CHECK_CONFIG(py_attr, word_size, as_attr, int);
  CHECK_CONFIG(py_attr, group_list, as_attr, std::vector<int>);
}
#undef CHECK_CONFIG

static void PyParseInputs(
    std::map<std::string, py::capsule>& py_inputs,  // python dict {"xx":
                                                    // DLTensor}
    std::map<std::string, DLManagedTensor*>& as_inputs) {
  for (auto& v : py_inputs) {
    as_inputs.insert(
        make_pair(v.first, static_cast<DLManagedTensor*>(v.second)));
    // no needs to add reference
    // for now, we hold the gil, and will copy all dltensor's data into
    // own tensor, also no need to move ownership.
  }
}

class PyInputReferenceGuard {
 public:
  std::map<std::string, py::capsule>& py_input_;
  PyInputReferenceGuard(std::map<std::string, py::capsule>& py_input)
      : py_input_(py_input) {
    for (auto& v : py_input_) {
      PyObject* pyobject = v.second.ptr();
      Py_INCREF(pyobject);
    }
  }
  ~PyInputReferenceGuard() {
    for (auto& v : py_input_) {
      PyObject* pyobject = v.second.ptr();
      Py_DECREF(pyobject);
    }
  }
};

static void PyParseInputs(
    std::map<std::string, py::object>& py_inputs,  // python dict {"xx": [list]}
    std::map<std::string, std::vector<std::vector<int64_t>>>& as_inputs) {
  for (auto& pair : py_inputs) {
    assert(py::isinstance<py::list>(pair.second));
    const auto& arr_2d = pair.second;
    for (auto& arr_1d : arr_2d) {
      assert(py::isinstance<py::list>(arr_1d));
      const std::vector<int64_t>& vals = arr_1d.cast<std::vector<int64_t>>();
      as_inputs[pair.first].emplace_back(vals);
    }
  }
}

static void PyParseOutputs(AsStatus status,
                           std::map<std::string, DLManagedTensor*>& as_outputs,
                           std::map<std::string, py::capsule>& py_outputs) {
  AS_CHECK(status);
  for (auto& v : as_outputs) {
    py_outputs.insert(make_pair(
        v.first, py::capsule(v.second, _c_str_dltensor, _c_dlpack_deleter)));
  }
}
PYBIND11_MODULE(_allspark, m) {
  py::enum_<AsStatus>(
      m, "AsStatus")  // 错误类型太多，所以设计为一切错误都会抛出异常给python
      .value("ALLSPARK_SUCCESS", AsStatus::ALLSPARK_SUCCESS)
      .value("ALLSPARK_STREAMING", AsStatus::ALLSPARK_STREAMING);
  py::enum_<AsCacheMode>(m, "AsCacheMode")
      .value("AsCacheDefault", AsCacheMode::AsCacheDefault)
      .value("AsCacheQuantI8", AsCacheMode::AsCacheQuantI8);
  py::enum_<AsMHAPrefill>(m, "AsMHAPrefill")
      .value("AsPrefillDefault", AsMHAPrefill::AsPrefillDefault);
  py::enum_<AsEngine::GenerateRequestStatus>(m, "GenerateRequestStatus")
      .value("Init", AsEngine::GenerateRequestStatus::Init)
      .value("ContextFinished",
             AsEngine::GenerateRequestStatus::ContextFinished)
      .value("Generating", AsEngine::GenerateRequestStatus::Generating)
      .value("GenerateFinished",
             AsEngine::GenerateRequestStatus::GenerateFinished)
      .value("GenerateInterrupted",
             AsEngine::GenerateRequestStatus::GenerateInterrupted);

  py::class_<AsFileInfo>(m, "AsFileInfo")
      .def(py::init<>())
      .def_readwrite("create_version_graph", &AsFileInfo::create_version_graph)
      .def_readwrite("current_version_engine",
                     &AsFileInfo::current_version_engine)
      .def_readwrite("create_version_param", &AsFileInfo::create_version_param);

  py::class_<AsModelConfig>(m, "AsModelConfig")
      .def(py::init<>())
      .def(py::init<>([](std::string model_name, std::string model_path,
                         std::string weights_path, std::string compute_unit,
                         int engine_max_length, int engine_max_batch,
                         bool text_graph, int num_threads,
                         std::string matmul_precision,
                         AsMHAPrefill prefill_mode, AsCacheMode cache_mode) {
             return new AsModelConfig(
                 model_name, model_path, weights_path, compute_unit,
                 engine_max_length, engine_max_batch, text_graph, num_threads,
                 matmul_precision, prefill_mode, cache_mode);
           }),
           py::arg("model_name"), py::arg("model_path"),
           py::arg("weights_path"), py::arg("compute_unit") = "CPU:0",
           py::arg("engine_max_length") = 0, py::arg("engine_max_batch") = 0,
           py::arg("text_graph") = false, py::arg("num_threads") = 0,
           py::arg("matmul_precision") = "highest",
           py::arg("prefill_mode") = AsMHAPrefill::AsPrefillDefault,
           py::arg("cache_mode") = AsCacheMode::AsCacheDefault)
      // required field: model_name / model_path / weights_path
      .def_readwrite("model_name", &AsModelConfig::model_name)
      .def_readwrite("model_path", &AsModelConfig::model_path)
      .def_readwrite("weights_path", &AsModelConfig::weights_path)
      .def_readwrite("text_graph", &AsModelConfig::text_graph)
      .def_readwrite("compute_unit", &AsModelConfig::compute_unit)
      .def_readwrite("engine_max_length", &AsModelConfig::engine_max_length)
      .def_readwrite("engine_max_batch", &AsModelConfig::engine_max_batch)
      .def_readwrite("num_threads", &AsModelConfig::num_threads)
      .def_readwrite("matmul_precision", &AsModelConfig::matmul_precision)
      .def_readwrite("prefill_mode", &AsModelConfig::prefill_mode)
      .def_readwrite("cache_mode", &AsModelConfig::cache_mode);

  py::class_<AsEngineStat>(m, "AsEngineStat")
      .def(py::init<>())
      .def(py::init<>(
          [](std::string model_name) { return new AsEngineStat(model_name); }))
      .def("ToString", [](AsEngineStat* self) { return self->ToString(); })
      .def("dict", [](AsEngineStat* self) { return self->ToMap(); })
      // required field: model_name / model_path / weights_path
      .def_readwrite("model_name", &AsEngineStat::model_name)
      .def_readwrite("free_token", &AsEngineStat::free_token)
      .def_readwrite("total_token", &AsEngineStat::total_token)
      .def_readwrite("pendding_request", &AsEngineStat::pendding_request)
      .def_readwrite("running_request", &AsEngineStat::running_request)
      .def_readwrite("total_generated_token",
                     &AsEngineStat::total_generated_token)
      .def_readwrite("total_prefill_token", &AsEngineStat::total_prefill_token)
      .def_readwrite("generate_token_persec",
                     &AsEngineStat::generate_token_persec)
      .def_readwrite("process_token_persec",
                     &AsEngineStat::process_token_persec)
      .def_readwrite("total_device_memory_pool_size",
                     &AsEngineStat::total_device_memory_pool_size)
      .def_readwrite("used_device_memory_pool_size",
                     &AsEngineStat::used_device_memory_pool_size);

  py::class_<GeneratedElements, std::shared_ptr<GeneratedElements>>(
      m, "GeneratedElements", py::module_local(),
      "Generated Token class, contains token(s) and related information, it "
      "may contains multiple tokens between last get.")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("ids_from_generate", &GeneratedElements::ids_from_generate,
                     "Token(s) from this generation")
      .def_readwrite(
          "log_probs_list", &GeneratedElements::log_probs_list,
          "A probability list for each token, including the top_logprobs "
          "tokens and their probabilities when generated.\n"
          "Dimension: [num_token][top_logprobs], where each token has a "
          "pair [token_id, prob].")
      .def_readwrite("token_logprobs_list",
                     &GeneratedElements::token_logprobs_list,
                     "Stores the probability value for each selected token.")
      .def_readwrite("tensors_from_model_inference",
                     &GeneratedElements::tensors_from_model_inference,
                     "Tensor outputs from model inference.");

  // Allspark API
  // We should make sure interfaces are the same with AsClientEngine
  py::class_<AsEngine>(m, "AsEngine")
      .def(py::init<>())
      .def("_build_model_from_as_model_config",
           [](AsEngine* self, py::object as_model_config_obj) {
             auto as_model_cfg = py::cast<AsModelConfig*>(as_model_config_obj);
             AS_CHECK(self->BuildModelFromConfigStruct(*as_model_cfg));
           })
      .def("get_model_info",
           [](AsEngine* self, const char* model_name) -> std::string {
             std::string model_info;
             AS_CHECK(self->GetModelInformation(model_name, &model_info));
             return model_info;
           })
      .def("_start_model",
           [](AsEngine* self, const char* model_name) {
             AsStatus status = self->StartModel(model_name);
             return status;
           })
      .def("_stop_model",
           [](AsEngine* self, const char* model_name) {
             AsStatus status = self->StopModel(model_name);
             return status;
           })
      .def("_release_model",
           [](AsEngine* self, const char* model_name) {
             AsStatus status = self->ReleaseModel(model_name);
             return status;
           })
      .def("_start_request",
           [](AsEngine* self, const char* model_name,
              std::map<std::string, py::capsule>&
                  inputs,  // python dict {"xx": DLTensor}
              std::map<std::string, py::object>& gen_cfg) {
             GenerateConfig as_gen_cfg{};
             std::map<std::string, DLManagedTensor*> as_inputs;
             std::map<std::string, DLManagedTensor*> as_outputs;
             std::map<std::string, py::capsule> py_outputs;
             PyParseConfig(gen_cfg, as_gen_cfg);
             PyParseInputs(inputs, as_inputs);
             std::shared_ptr<AsEngine::RequestContent> req =
                 std::make_shared<AsEngine::RequestContent>();
             req->config = as_gen_cfg;
             req->infer_type = AsEngine::RequestInferType::Generate;
             req->inputs = std::make_shared<DLTensorMap>(as_inputs);

             RequestHandle_t request_handle;
             AsEngine::ResultQueue_t result_queue = nullptr;
             AsStatus status = self->StartRequest(
                 model_name, req, &request_handle, &result_queue);
             // TODO: check status
             py::tuple ret =
                 py::make_tuple(status, (void*)request_handle, result_queue);
             return ret;
           })
      .def(
          "_get_no_wait",
          [](AsEngine* self, const char* model_name, void* result_queue) {
            auto ele = ((AsEngine::ResultQueue_t)result_queue)->GetNoWait();
            if (ele) {
              std::vector<int64_t> result = ele->ids_from_generate;
              return result;
            } else {
              std::vector<int64_t> emptyVec;
              return emptyVec;
            }
          },
          py::return_value_policy::copy,
          "get output token non-block api, return None if no new token "
          "available"
          "[deprecated function] please use ResultQueue function.")
      .def(
          "_get_wait",
          [](AsEngine* self, const char* model_name, void* result_queue) {
            // release and acquire gil lock in case c++ Get() function block all
            // python runtime threads
            py::gil_scoped_release release;
            auto ele = ((AsEngine::ResultQueue_t)result_queue)->Get();
            py::gil_scoped_acquire acquire;
            if (ele) {
              std::vector<int64_t> result = ele->ids_from_generate;
              return result;
            } else {
              std::vector<int64_t> emptyVec;
              return emptyVec;
            }
          },
          py::return_value_policy::copy,
          "get output token block api, will block until new token or status "
          "change"
          ". [deprecated function] please use ResultQueue function.")
      .def("_get_request_status",
           [](AsEngine* self, const char* model_name, void* result_queue) {
             return ((AsEngine::ResultQueue_t)result_queue)->GenerateStatus();
           })
      .def("_stop_request",
           [](AsEngine* self, const char* model_name,
              void* request_handle) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status =
                 self->StopRequest(model_name, (RequestHandle_t)request_handle);
             return status;
           })
      .def("_release_request",
           [](AsEngine* self, const char* model_name,
              void* request_handle) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = self->ReleaseRequest(
                 model_name, (RequestHandle_t)request_handle);
             return status;
           })
      .def("_sync_request",
           [](AsEngine* self, const char* model_name,
              void* request_handle) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = AsStatus::ALLSPARK_SUCCESS;
             if (request_handle == nullptr) {
               status = self->SyncRequest(model_name, nullptr);
             } else {
               status = self->SyncRequest(model_name,
                                          (RequestHandle_t)request_handle);
             }
             return status;
           })
      .def("get_file_information",
           [](AsEngine* self, const char* as_graph_path,
              const char* as_param_path) -> AsFileInfo {
             return self->GetFileInformation(as_graph_path, as_param_path);
           })

      .def("get_as_engine_stat",
           [](AsEngine* self, const char* model_name) {
             py::gil_scoped_release release;
             return self->GetAsEngineStat(model_name);
           })
      .def("get_op_profiling_info",
           [](AsEngine* self, const char* model_name) -> std::string {
             return self->GetOpProfilingInfo(model_name);
           })
      .def("get_rank_id",
           [](AsEngine* self) -> int { return self->GetRankId(); })
      .def("get_version_full",
           [](AsEngine* self) -> std::string { return self->GetVersionFull(); })
      .def("is_allspark_work_as_service", [](AsEngine* self) -> bool {
        return self->IsAllSparkWorkAsService();
      });

  // Allspark client API to use grpc to interact with allspark daemon service
  // Normally we should make sure interfaces are the same with AsEngine
  py::class_<AsClientEngine>(m, "AsClientEngine")
      .def(py::init<>())
      .def("_build_model_from_as_model_config",
           [](AsClientEngine* self, py::object as_model_config_obj) {
             auto as_model_cfg = py::cast<AsModelConfig*>(as_model_config_obj);
             AS_CHECK(self->BuildModelFromConfigStruct(*as_model_cfg));
           })
      .def("get_model_info",
           [](AsClientEngine* self, const char* model_name) -> std::string {
             std::string model_info;
             AS_CHECK(self->GetModelInformation(model_name, &model_info));
             return model_info;
           })
      .def("_start_model",
           [](AsClientEngine* self, const char* model_name) {
             AsStatus status = self->StartModel(model_name);
             return status;
           })
      .def("_stop_model",
           [](AsClientEngine* self, const char* model_name) {
             AsStatus status = self->StopModel(model_name);
             return status;
           })
      .def("_release_model",
           [](AsClientEngine* self, const char* model_name) {
             AsStatus status = self->ReleaseModel(model_name);
             return status;
           })
      .def("_start_request",
           [](AsClientEngine* self, const char* model_name,
              std::map<std::string, py::capsule>&
                  inputs,  // python dict {"xx": DLTensor}
              std::map<std::string, py::object>& gen_cfg) {
             GenerateConfig as_gen_cfg{};
             std::map<std::string, DLManagedTensor*> as_inputs;
             std::map<std::string, DLManagedTensor*> as_outputs;
             std::map<std::string, py::capsule> py_outputs;
             PyParseConfig(gen_cfg, as_gen_cfg);
             PyParseInputs(inputs, as_inputs);
             std::shared_ptr<AsEngine::RequestContent> req =
                 std::make_shared<AsEngine::RequestContent>();
             req->config = as_gen_cfg;
             req->infer_type = AsEngine::RequestInferType::Generate;
             req->inputs = std::make_shared<DLTensorMap>(as_inputs);

             RequestHandle_t request_handle;
             AsEngine::ResultQueue_t result_queue = nullptr;
             AsStatus status = self->StartRequest(
                 model_name, req, &request_handle, &result_queue);
             // TODO: check status
             py::tuple ret =
                 py::make_tuple(status, (void*)request_handle, result_queue);
             return ret;
           })
      .def(
          "_get_no_wait",
          [](AsClientEngine* self, const char* model_name, void* result_queue) {
            auto ele = ((AsEngine::ResultQueue_t)result_queue)->GetNoWait();
            if (ele) {
              std::vector<int64_t> result = ele->ids_from_generate;
              return result;
            } else {
              std::vector<int64_t> emptyVec;
              return emptyVec;
            }
          },
          py::return_value_policy::copy)
      .def(
          "_get_wait",
          [](AsClientEngine* self, const char* model_name, void* result_queue) {
            // release and acquire gil lock in case c++ Get() function block all
            // python runtime threads
            py::gil_scoped_release release;
            auto ele = ((AsEngine::ResultQueue_t)result_queue)->Get();
            py::gil_scoped_acquire acquire;
            if (ele) {
              std::vector<int64_t> result = ele->ids_from_generate;
              return result;
            } else {
              std::vector<int64_t> emptyVec;
              return emptyVec;
            }
          },
          py::return_value_policy::copy)
      .def(
          "_get_request_status",
          [](AsClientEngine* self, const char* model_name, void* result_queue) {
            return ((AsEngine::ResultQueue_t)result_queue)->GenerateStatus();
          })
      .def("_stop_request",
           [](AsClientEngine* self, const char* model_name,
              void* request_handle) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status =
                 self->StopRequest(model_name, (RequestHandle_t)request_handle);
             return status;
           })
      .def("_release_request",
           [](AsClientEngine* self, const char* model_name,
              void* request_handle) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = self->ReleaseRequest(
                 model_name, (RequestHandle_t)request_handle);
             return status;
           })
      .def("_sync_request",
           [](AsClientEngine* self, const char* model_name,
              void* request_handle) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = AsStatus::ALLSPARK_SUCCESS;
             if (request_handle == nullptr) {
               status = self->SyncRequest(model_name, nullptr);
             } else {
               status = self->SyncRequest(model_name,
                                          (RequestHandle_t)request_handle);
             }
             return status;
           })
      .def("get_file_information",
           [](AsClientEngine* self, const char* as_graph_path,
              const char* as_param_path) -> AsFileInfo {
             return self->GetFileInformation(as_graph_path, as_param_path);
           })

      .def("get_as_engine_stat",
           [](AsClientEngine* self, const char* model_name) {
             py::gil_scoped_release release;
             return self->GetAsEngineStat(model_name);
           })
      .def("get_op_profiling_info",
           [](AsClientEngine* self, const char* model_name) -> std::string {
             return self->GetOpProfilingInfo(model_name);
           })
      .def("get_rank_id",
           [](AsClientEngine* self) -> int { return self->GetRankId(); })
      .def("get_version_full",
           [](AsClientEngine* self) -> std::string {
             return self->GetVersionFull();
           })
      .def("is_allspark_work_as_service", [](AsClientEngine* self) -> bool {
        return self->IsAllSparkWorkAsService();
      });
  m.def("save_allsparky", [](const std::string& bin_data,
                             std::map<std::string, py::object>& py_attr) {
    TensorAttribute as_attr{};
    PyParseAttribute(py_attr, as_attr);
    return py::bytes(util::save_allsparky(bin_data, as_attr));
  });

  m.def("save_allsparky_dltensor_tofile",
        [](const std::string& weights_path, const std::string& name,
           py::capsule data, std::map<std::string, py::object>& py_attr) {
          TensorAttribute as_attr{};
          PyParseAttribute(py_attr, as_attr);
          DLManagedTensor* managed_dltensor =
              static_cast<DLManagedTensor*>(data);
          int64_t nbytes = as_attr.word_size;
          for (int i = 0; i < as_attr.shape.size(); i++) {
            nbytes *= as_attr.shape[i];
          }
          const DLTensor& dltensor = managed_dltensor->dl_tensor;
          util::save_allsparky_tofile(weights_path, name, dltensor.data, nbytes,
                                      as_attr);
        });

  m.def("set_global_header", [](const std::string& weights_path) {
    util::set_global_header(weights_path);
  });

  py::class_<ResultQueue, std::shared_ptr<ResultQueue>>(
      m, "ResultQueue", py::module_local(),
      "The ResultQueue class is designed to generate status and retrieve "
      "results")
      .def(py::init<>())  // Assuming there's a default constructor

      .def("GenerateStatus", &ResultQueue::GenerateStatus,
           "Get generation status, this api will not block.")

      .def("GeneratedLength", &ResultQueue::GeneratedLength,
           "Get current generated length, it's accumulated generate token "
           "number")

      .def(
          "Get",
          [](ResultQueue* self) -> py::object {
            py::gil_scoped_release release;
            auto ele = self->Get();
            py::gil_scoped_acquire acquire;
            if (ele == nullptr) {
              return py::none();
            } else {
              return py::cast(ele);
            }
          },
          "Fetches new tokens(s) from the queue, will be block until new token "
          "generated.")

      .def(
          "GetNoWait",
          [](ResultQueue* self) -> py::object {
            auto ele = self->GetNoWait();
            if (ele == nullptr) {
              return py::none();
            } else {
              return py::cast(ele);
            }
          },
          "Fetches new token(s) from the queue, will return None if no new "
          "tokens, non block api.");
}

#undef CHECK_CONFIG
