/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_client_binding.cpp
 */

#include <allspark.h>
#include <allspark_client.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "allspark_binding_common.h"

namespace py = pybind11;
using namespace allspark;
using AsClientEngine = allspark::AsClientEngine;

PYBIND11_MODULE(_allspark_client, m) {
  bindAsStatus(m);
  bindResultQueue(m);
  bindGeneratedElements(m);

  // Allspark client API to use grpc to interact with allspark daemon service
  // Normally we should make sure interfaces are the same with AsEngine
  py::class_<AsClientEngine>(m, "AsClientEngine")
      .def(py::init<>())
      .def(
          "_build_model_from_as_model_config",
          [](AsClientEngine* self, py::object as_model_config_obj) {
            auto as_model_cfg = py::cast<AsModelConfig*>(as_model_config_obj);
            AS_CHECK(self->BuildModelFromConfigStruct(*as_model_cfg));
          },
          "")
      .def("unload_device_mem",
           [](AsClientEngine* self, const char* model_name) {
             py::gil_scoped_release release;
             AS_CHECK(self->UnloadModelFromDeviceMemory(model_name));
           })
      .def("load_device_mem",
           [](AsClientEngine* self, const char* model_name) {
             py::gil_scoped_release release;
             AS_CHECK(self->ReloadModelToDeviceMemory(model_name));
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
             req->mm_type = AsEngine::RequestMMType::TextInput;

             RequestHandle_t request_handle = nullptr;
             AsEngine::ResultQueue_t result_queue = nullptr;
             AsStatus status = self->StartRequest(
                 model_name, req, &request_handle, &result_queue);
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
      .def("load_lora",
           [](AsClientEngine* self, const char* model_name,
              const char* lora_name) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = AsStatus::ALLSPARK_SUCCESS;
             status = self->LoadLoraByName(model_name, lora_name);
             return status;
           })
      .def("unload_lora",
           [](AsClientEngine* self, const char* model_name,
              const char* lora_name) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = AsStatus::ALLSPARK_SUCCESS;
             status = self->UnloadLoraByName(model_name, lora_name);
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
}
