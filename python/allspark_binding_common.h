/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_binding_common.h
 */

#pragma once
#include <allspark.h>
#include <pybind11/pybind11.h>

#include <map>

namespace py = pybind11;
using namespace allspark;
using ResultQueue = allspark::AsEngine::ResultQueue;
using GeneratedElements = allspark::AsEngine::GeneratedElements;

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
  CHECK_CONFIG(py_gen_cfg, frequency_penalty, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, length_penalty, as_gen_cfg, float);
  CHECK_CONFIG(py_gen_cfg, suppress_repetition_in_generation, as_gen_cfg, bool);
  CHECK_CONFIG(py_gen_cfg, min_length, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, max_length, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, no_repeat_ngram_size, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, eos_token_id, as_gen_cfg, int);

  if (py_gen_cfg.find("uuid") != py_gen_cfg.end()) {
    as_gen_cfg.user_request_id = (py_gen_cfg["uuid"].cast<std::string>());
  }

  // CHECK_CONFIG(py_gen_cfg, prefill_mode, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, stop_words_ids, as_gen_cfg,
               std::vector<std::vector<int64_t>>);
  CHECK_CONFIG(py_gen_cfg, lora_name, as_gen_cfg, std::string);
  CHECK_CONFIG(py_gen_cfg, bad_words_ids, as_gen_cfg,
               std::vector<std::vector<int>>);
  CHECK_CONFIG(py_gen_cfg, top_k, as_gen_cfg, int);
  CHECK_CONFIG(py_gen_cfg, top_p, as_gen_cfg, float);
  // TODO: need support int64_t seed
  CHECK_CONFIG(py_gen_cfg, seed, as_gen_cfg, long);

  CHECK_CONFIG(py_gen_cfg, logprobs, as_gen_cfg, bool);
  CHECK_CONFIG(py_gen_cfg, top_logprobs, as_gen_cfg, int);

  as_gen_cfg.mm_info = nullptr;
  CHECK_CONFIG(py_gen_cfg, mm_info, as_gen_cfg, MultiMediaInfo*);
  if (py_gen_cfg.find("response_format") != py_gen_cfg.end()) {
    as_gen_cfg.response_format =
        (py_gen_cfg["response_format"]
             .cast<std::map<std::string, std::string>>());
  }
  if (py_gen_cfg.find("vocab") != py_gen_cfg.end()) {
    as_gen_cfg.vocab = (py_gen_cfg["vocab"].cast<std::map<std::string, int>>());
  }
  CHECK_CONFIG(py_gen_cfg, vocab_type, as_gen_cfg, VocabType);
  CHECK_CONFIG(py_gen_cfg, enable_tensors_from_model_inference, as_gen_cfg,
               bool);
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

void bindAsStatus(py::module& m) {
  py::enum_<AsStatus>(m, "AsStatus", py::module_local(),
                      "Status Code for allspark api, exception will be "
                      "throw if meets error.")
      .value("ALLSPARK_SUCCESS", AsStatus::ALLSPARK_SUCCESS)
      .value("ALLSPARK_PARAM_ERROR", AsStatus::ALLSPARK_PARAM_ERROR)
      .value("ALLSPARK_EXCEED_LIMIT_ERROR",
             AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR)
      .value("ALLSPARK_ALLSPARK_INVALID_CALL_ERROR",
             AsStatus::ALLSPARK_INVALID_CALL_ERROR)
      .value("ALLSPARK_REQUEST_DENIED", AsStatus::ALLSPARK_REQUEST_DENIED)
      .value("ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR",
             AsStatus::ALLSPARK_LORA_NUM_EXCEED_LIMIT_ERROR)
      .value("ALLSPARK_LORA_RANK_EXCEED_LIMIT_ERROR",
             AsStatus::ALLSPARK_LORA_RANK_EXCEED_LIMIT_ERROR)
      .value("ALLSPARK_LORA_NOT_FOUND", AsStatus::ALLSPARK_LORA_NOT_FOUND)
      .value("ALLSPARK_LORA_ALREADY_LOADED",
             AsStatus::ALLSPARK_LORA_ALREADY_LOADED)
      .value("ALLSPARK_LORA_IN_USE", AsStatus::ALLSPARK_LORA_IN_USE)
      .value("ALLSPARK_STREAMING", AsStatus::ALLSPARK_STREAMING);
}

void bindResultQueue(py::module& m) {
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

      .def("RequestStatInfo", &ResultQueue::RequestStatInfo,
           "Get key-value dict of all the statistic of this request.")

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
          "GetWithTimeout",
          [](ResultQueue* self, int timeout_ms) -> py::object {
            py::gil_scoped_release release;
            auto ele = self->Get(timeout_ms);
            py::gil_scoped_acquire acquire;
            if (ele == nullptr) {
              return py::none();
            } else {
              return py::cast(ele);
            }
          },
          "Fetches new tokens(s) from the queue, will be block until new token "
          "generated or timeout(in ms)")
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
std::vector<std::map<std::string, py::capsule>>
get_tensors_from_model_inference(const GeneratedElements ele) {
  const std::vector<std::unordered_map<std::string, std::shared_ptr<ITensor>>>
      in = ele.tensors_from_model_inference;
  std::vector<std::map<std::string, py::capsule>> output;
  output.reserve(in.size());
  for (const auto& map : in) {
    std::map<std::string, py::capsule> new_map;
    for (const auto& pair : map) {
      const std::string& key = pair.first;
      // std::shared_ptr<ITensor> tensor = (pair.second)->ToDLPack();
      DLManagedTensor* tensor = (pair.second)->ToDLPack();
      // 这里把实际上把所有权交给python了
      // 用 py::capsule 封装 DLManagedTensor*
      new_map.emplace(key,
                      py::capsule(tensor, _c_str_dltensor, _c_dlpack_deleter));
    }
    output.emplace_back((new_map));
  }
  return output;
}
void bindGeneratedElements(py::module& m) {
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
      .def_property_readonly(
          "tensors_from_model_inference",
          //  &GeneratedElements::tensors_from_model_inference,
          &get_tensors_from_model_inference,
          "Tensor outputs from model inference.")
      .def_readwrite("prefix_cache_len", &GeneratedElements::prefix_cache_len,
                     "Cached prefix token length.")
      .def_readwrite("prefix_len_gpu", &GeneratedElements::prefix_len_gpu,
                     "GPU cached prefix token length.")
      .def_readwrite("prefix_len_cpu", &GeneratedElements::prefix_len_cpu,
                     "CPU cached prefix token length.");
}

void bindGenerateRequestStatus(py::module& m) {
  py::enum_<AsEngine::GenerateRequestStatus>(m, "GenerateRequestStatus")
      .value("Init", AsEngine::GenerateRequestStatus::Init)
      .value("ContextFinished",
             AsEngine::GenerateRequestStatus::ContextFinished)
      .value("Generating", AsEngine::GenerateRequestStatus::Generating)
      .value("GenerateFinished",
             AsEngine::GenerateRequestStatus::GenerateFinished)
      .value("GenerateInterrupted",
             AsEngine::GenerateRequestStatus::GenerateInterrupted);
}
