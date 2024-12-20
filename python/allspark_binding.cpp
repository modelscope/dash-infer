/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_binding.cpp
 */

#include <allspark.h>
#include <allsparkz_util.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>

#include "allspark_binding_common.h"

// major and minor name is polluted by macros introduced by sys/sysmacros.h
#ifdef major
#pragma push_macro("major")
#undef major
#endif

#ifdef minor
#pragma push_macro("minor")
#undef minor
#endif

#include "allspark.pb.h"

#pragma pop_macro("minor")
#pragma pop_macro("major")

namespace py = pybind11;
using namespace allspark;
using AsEngine = allspark::AsEngine;
using MultiMediaInfo = allspark::MultiMediaInfo;

class PyInputReferenceGuard {
 public:
  std::map<std::string, py::capsule>& py_input_;
  explicit PyInputReferenceGuard(std::map<std::string, py::capsule>& py_input)
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

static void PyParseOutputs(AsStatus status, DLTensorMap& as_outputs,
                           std::map<std::string, py::capsule>& py_outputs) {
  AS_CHECK(status);
  for (auto& v : as_outputs) {
    py_outputs.insert(make_pair(
        v.first, py::capsule(v.second, _c_str_dltensor, _c_dlpack_deleter)));
  }
}

template <typename Element, typename VE = Element*>
class RepeatedPtrFieldPtr {
 public:
  RepeatedPtrFieldPtr(::google::protobuf::RepeatedPtrField<Element>* ptr)
      : ptr_(ptr) {
    iter_ = ptr_->end();
  }
  ~RepeatedPtrFieldPtr() = default;
  Element* Mutable(int index) {
    // convert python negative index to positive index
    index = index >= 0 ? index : ptr_->size() + index;
    assert(index >= 0);
    return ptr_->Mutable(index);
  }
  void Erase(int index) {
    auto iter = ptr_->begin();
    std::advance(iter, index);
    ptr_->erase(iter);
  }
  RepeatedPtrFieldPtr<Element, VE>* Iterator() {
    iter_ = ptr_->begin();
    return this;
  }
  Element* Next() {
    if (iter_ == ptr_->end()) {
      throw py::stop_iteration();
    }
    Element* e = &*iter_;
    std::advance(iter_, 1);
    return e;
  }
  void Extend(const std::vector<VE>& elements) {
    for (auto& eptr : elements) {
      Append(eptr);
    }
  }

  void Append(const Element* element) {
    Element* new_element_ptr = ptr_->Add();
    *new_element_ptr = *element;
  }

  void Append(const std::string& element) {
    std::string* new_element_ptr = ptr_->Add();
    *new_element_ptr = element;
  }

  void Insert(int index, const Element& element) {
    Append(&element);
    std::rotate(ptr_->pointer_begin() + index,
                ptr_->pointer_begin() + ptr_->size() - 1, ptr_->pointer_end());
  }

 private:
  ::google::protobuf::RepeatedPtrField<Element>* ptr_;
  typename ::google::protobuf::RepeatedPtrField<Element>::iterator iter_;
};

template <typename Element>
class RepeatedFieldPtr {
 public:
  RepeatedFieldPtr(::google::protobuf::RepeatedField<Element>* ptr)
      : ptr_(ptr) {}
  ~RepeatedFieldPtr() = default;
  Element* Mutable(int index) {
    // convert python negative index to positive index
    index = index >= 0 ? index : ptr_->size() + index;
    assert(index >= 0);
    return ptr_->Mutable(index);
  }
  void Erase(int index) {
    auto iter = ptr_->begin();
    std::advance(iter, index);
    ptr_->erase(iter);
  }
  void Extend(const std::vector<Element>& elements) {
    ptr_->Add(elements.begin(), elements.end());
  }

  void Append(Element element) { ptr_->Add(element); }

 private:
  ::google::protobuf::RepeatedField<Element>* ptr_;
};

#define PYCLASS_REPEATED_PTR_PROTO_BIND(m, pyclass, T, VT)                     \
  py::class_<RepeatedPtrFieldPtr<T, VT>>(m, pyclass)                           \
      .def(                                                                    \
          "__getitem__",                                                       \
          [](RepeatedPtrFieldPtr<T, VT>* self, int index) {                    \
            return self->Mutable(index);                                       \
          },                                                                   \
          py::return_value_policy::reference)                                  \
      .def("__setitem__",                                                      \
           [](RepeatedPtrFieldPtr<T, VT>* self, int index, const T& element) { \
             *(self->Mutable(index)) = element;                                \
           })                                                                  \
      .def("__delitem__", [](RepeatedPtrFieldPtr<T, VT>* self,                 \
                             int index) { self->Erase(index); })               \
      .def(                                                                    \
          "__iter__",                                                          \
          [](RepeatedPtrFieldPtr<T, VT>* self) { return self->Iterator(); },   \
          py::return_value_policy::reference)                                  \
      .def(                                                                    \
          "__next__",                                                          \
          [](RepeatedPtrFieldPtr<T, VT>* self) { return self->Next(); },       \
          py::return_value_policy::reference)                                  \
      .def("extend",                                                           \
           [](RepeatedPtrFieldPtr<T, VT>* self,                                \
              const std::vector<VT>& elements) { self->Extend(elements); })    \
      .def("append", [](RepeatedPtrFieldPtr<T, VT>* self,                      \
                        const T* element) { self->Append(element); })          \
      .def("insert", [](RepeatedPtrFieldPtr<T, VT>* self, int index,           \
                        const T& element) { self->Insert(index, element); });

#define PYCLASS_REPEATED_PROTO_BIND(m, pyclass, T)                           \
  py::class_<RepeatedFieldPtr<T>>(m, pyclass)                                \
      .def(                                                                  \
          "__getitem__",                                                     \
          [](RepeatedFieldPtr<T>* self, int index) {                         \
            assert(index >= 0);                                              \
            return self->Mutable(index);                                     \
          },                                                                 \
          py::return_value_policy::reference)                                \
      .def("__setitem__",                                                    \
           [](RepeatedFieldPtr<T>* self, int index, const T& element) {      \
             assert(index >= 0);                                             \
             *(self->Mutable(index)) = element;                              \
           })                                                                \
      .def("__delitem__",                                                    \
           [](RepeatedFieldPtr<T>* self, int index) { self->Erase(index); }) \
      .def("extend",                                                         \
           [](RepeatedFieldPtr<T>* self, const std::vector<T>& elements) {   \
             self->Extend(elements);                                         \
           })                                                                \
      .def("append",                                                         \
           [](RepeatedFieldPtr<T>* self, T tensor) { self->Append(tensor); });

template <typename Key, typename T>
class MapPtr {
 public:
  MapPtr(::google::protobuf::Map<Key, T>* ptr) : ptr_(ptr) {}
  ~MapPtr() = default;
  T* Mutable(const Key& key) { return &((*ptr_)[key]); }

 private:
  ::google::protobuf::Map<Key, T>* ptr_;
};
#define PYCLASS_MAP_PTR_PROTO_BIND(m, pyclass, KeyType, ValueType)   \
  py::class_<MapPtr<KeyType, ValueType>>(m, pyclass)                 \
      .def(                                                          \
          "__getitem__",                                             \
          [](MapPtr<KeyType, ValueType>* self, const KeyType& key) { \
            return self->Mutable(key);                               \
          },                                                         \
          py::return_value_policy::reference)                        \
      .def("__setitem__",                                            \
           [](MapPtr<KeyType, ValueType>* self, KeyType key,         \
              ValueType value) { *(self->Mutable(key)) = value; });

PYBIND11_MODULE(_allspark, m) {
  // =================================== allspark.proto =======================
  m.attr("DEVICETYPE_UNDEFINED") =
      py::int_(int(DeviceType::DEVICETYPE_UNDEFINED));
  m.attr("CPU") = py::int_(int(DeviceType::CPU));
  m.attr("CUDA") = py::int_(int(DeviceType::CUDA));
  m.attr("COMPILE_TIME_MAX_DEVICE_TYPES") =
      py::int_(int(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES));
  m.attr("CPU_PINNED") = py::int_(int(DeviceType::CPU_PINNED));

  m.attr("DENSE") = py::int_(int(DataMode::DENSE));
  m.attr("CSC") = py::int_(int(DataMode::CSC));
  m.attr("ELL") = py::int_(int(DataMode::ELL));

  m.attr("NOSPLIT") = py::int_(int(SplitMode::NOSPLIT));
  m.attr("VSPLIT") = py::int_(int(SplitMode::VSPLIT));
  m.attr("HSPLIT") = py::int_(int(SplitMode::HSPLIT));
  m.attr("QKVSPLIT") = py::int_(int(SplitMode::QKVSPLIT));
  m.attr("KVSPLIT") = py::int_(int(SplitMode::KVSPLIT));
  m.attr("HSPLIT_QUANTIZE") = py::int_(int(SplitMode::HSPLIT_QUANTIZE));
  m.attr("GROUP_VSPLIT") = py::int_(int(SplitMode::GROUP_VSPLIT));
  m.attr("MQA_VSPLIT") = py::int_(int(SplitMode::MQA_VSPLIT));
  m.attr("BATCH_VSPLIT") = py::int_(int(SplitMode::BATCH_VSPLIT));
  m.attr("BATCH_HSPLIT") = py::int_(int(SplitMode::BATCH_HSPLIT));
  m.attr("BATCH_KVSPLIT") = py::int_(int(SplitMode::BATCH_KVSPLIT));

  m.attr("DATATYPE_UNDEFINED") = py::int_(int(DataType::DATATYPE_UNDEFINED));
  m.attr("FLOAT32") = py::int_(int(DataType::FLOAT32));
  m.attr("FLOAT16") = py::int_(int(DataType::FLOAT16));
  m.attr("INT8") = py::int_(int(DataType::INT8));
  m.attr("INT16") = py::int_(int(DataType::INT16));
  m.attr("INT32") = py::int_(int(DataType::INT32));
  m.attr("INT64") = py::int_(int(DataType::INT64));
  m.attr("STRING") = py::int_(int(DataType::STRING));
  m.attr("BOOL") = py::int_(int(DataType::BOOL));
  m.attr("BFLOAT16") = py::int_(int(DataType::BFLOAT16));
  m.attr("UINT8") = py::int_(int(DataType::UINT8));
  m.attr("POINTER") = py::int_(int(DataType::POINTER));

  m.attr("HIGHEST") = py::int_(int(PrecisionLevel::HIGHEST));
  m.attr("HIGH") = py::int_(int(PrecisionLevel::HIGH));
  m.attr("MEDIUM_BF16") = py::int_(int(PrecisionLevel::MEDIUM_BF16));
  m.attr("MEDIUM_FP16") = py::int_(int(PrecisionLevel::MEDIUM_FP16));

  m.attr("BINARYTYPE_UNDEFINED") =
      py::int_(int(BinaryType::BINARYTYPE_UNDEFINED));
  m.attr("ADD") = py::int_(int(BinaryType::ADD));
  m.attr("MUL") = py::int_(int(BinaryType::MUL));
  m.attr("FUSED_MUL_ADD_1") = py::int_(int(BinaryType::FUSED_MUL_ADD_1));
  m.attr("GEGLU") = py::int_(int(BinaryType::GEGLU));
  m.attr("SWIGLU") = py::int_(int(BinaryType::SWIGLU));

  m.attr("UNARYTYPE_UNDEFINED") = py::int_(int(UnaryType::UNARYTYPE_UNDEFINED));
  m.attr("TANH") = py::int_(int(UnaryType::TANH));
  m.attr("GELU_ERF") = py::int_(int(UnaryType::GELU_ERF));
  m.attr("GELU_TANH") = py::int_(int(UnaryType::GELU_TANH));
  m.attr("SILU") = py::int_(int(UnaryType::SILU));
  m.attr("SIGMOID") = py::int_(int(UnaryType::SIGMOID));

  m.attr("base_rotary") = py::int_(int(RotaryInvFreqType::base_rotary));
  m.attr("chatglm_v2") = py::int_(int(RotaryInvFreqType::chatglm_v2));
  m.attr("chatglm_v3") = py::int_(int(RotaryInvFreqType::chatglm_v3));
  m.attr("yarn") = py::int_(int(RotaryInvFreqType::yarn));

  py::class_<ConfigProto>(m, "ConfigProto")
      .def(py::init<>())
      .def_property(
          "dtype", [](ConfigProto* self) { return int(self->dtype()); },
          [](ConfigProto* self, int dtype) {
            self->set_dtype(DataType(dtype));
          })
      .def_property("ln_eps", &ConfigProto::ln_eps, &ConfigProto::set_ln_eps)
      .def_property("num_heads", &ConfigProto::num_heads,
                    &ConfigProto::set_num_heads)
      .def_property("with_weights", &ConfigProto::with_weights,
                    &ConfigProto::set_with_weights)
      .def_property("enc_layer", &ConfigProto::enc_layer,
                    &ConfigProto::set_enc_layer)
      .def_property("dec_layer", &ConfigProto::dec_layer,
                    &ConfigProto::set_dec_layer)
      .def_property("is_generate", &ConfigProto::is_generate,
                    &ConfigProto::set_is_generate)
      .def_property("start_dec_id", &ConfigProto::start_dec_id,
                    &ConfigProto::set_start_dec_id)
      .def_property("end_dec_id", &ConfigProto::end_dec_id,
                    &ConfigProto::set_end_dec_id)
      .def_property("num_beam", &ConfigProto::num_beam,
                    &ConfigProto::set_num_beam)
      .def_property("data_mode", &ConfigProto::data_mode,
                    &ConfigProto::set_data_mode)
      .def_property(
          "activation",
          [](ConfigProto* self) { return int(self->activation()); },
          [](ConfigProto* self, int activation) {
            self->set_activation(UnaryType(activation));
          })
      .def_property("d_model", &ConfigProto::d_model, &ConfigProto::set_d_model)
      .def_property("enc_num_heads", &ConfigProto::enc_num_heads,
                    &ConfigProto::set_enc_num_heads)
      .def_property("dec_num_heads", &ConfigProto::dec_num_heads,
                    &ConfigProto::set_dec_num_heads)
      .def_property("multi_query_group_num",
                    &ConfigProto::multi_query_group_num,
                    &ConfigProto::set_multi_query_group_num)
      .def_property("kv_channels", &ConfigProto::kv_channels,
                    &ConfigProto::set_kv_channels)
      .def_property("size_per_head", &ConfigProto::size_per_head,
                    &ConfigProto::set_size_per_head)
      .def_property("hidden_size", &ConfigProto::hidden_size,
                    &ConfigProto::set_hidden_size)
      .def_property("num_experts", &ConfigProto::num_experts,
                    &ConfigProto::set_num_experts)
      .def_property("num_experts_per_tok", &ConfigProto::num_experts_per_tok,
                    &ConfigProto::set_num_experts_per_tok)
      .def_property("intermediate_size", &ConfigProto::intermediate_size,
                    &ConfigProto::set_intermediate_size);

// major, minor is polluted by macro introduced by sys/sysmacros.h
#ifdef major
#pragma push_macro("major")
#undef major
#endif

#ifdef minor
#pragma push_macro("minor")
#undef minor
#endif
  py::class_<BuildVersion>(m, "BuildVersion")
      .def(py::init<>())
      .def_property("major", &BuildVersion::major, &BuildVersion::set_major)
      .def_property("minor", &BuildVersion::minor, &BuildVersion::set_minor)
      .def_property("patch", &BuildVersion::patch, &BuildVersion::set_patch)
      .def_property("git_commit", &BuildVersion::git_commit,
                    &BuildVersion::set_git_commit<std::string>)
      .def_property("git_tag", &BuildVersion::git_tag,
                    &BuildVersion::set_git_tag<std::string>);

#pragma pop_macro("minor")
#pragma pop_macro("major")

  PYCLASS_REPEATED_PTR_PROTO_BIND(m, "RepeatedTensorProto", TensorProto,
                                  TensorProto*);
  PYCLASS_REPEATED_PTR_PROTO_BIND(m, "RepeatedOperatorProto", OperatorProto,
                                  OperatorProto*);
  PYCLASS_REPEATED_PTR_PROTO_BIND(m, "RepeatedString", std::string,
                                  std::string);
  PYCLASS_REPEATED_PROTO_BIND(m, "RepeatedInt64", int64_t);
  PYCLASS_MAP_PTR_PROTO_BIND(m, "MapStringString", std::string, std::string);
  PYCLASS_MAP_PTR_PROTO_BIND(m, "MapStringTensorProto", std::string,
                             TensorProto);
  PYCLASS_MAP_PTR_PROTO_BIND(m, "MapStringGraphProto", std::string, GraphProto);

  bindAsStatus(m);
  bindResultQueue(m);
  bindGeneratedElements(m);
  bindGenerateRequestStatus(m);

  py::class_<WeightHash>(m, "WeightHash")
      .def(py::init<>())
      .def_property("algorithm", &WeightHash::algorithm,
                    &WeightHash::set_algorithm<std::string>)
      .def_property_readonly(
          "hash_length",
          [](WeightHash* self) {
            return RepeatedFieldPtr<int64_t>(self->mutable_hash_length());
          })
      .def_property_readonly("hash", [](WeightHash* self) {
        return RepeatedPtrFieldPtr<std::string, std::string>(
            self->mutable_hash());
      });

  py::class_<BuildMetaProto>(m, "BuildMetaProto")
      .def(py::init<>())
      .def_property_readonly("version", &BuildMetaProto::mutable_version,
                             py::return_value_policy::reference)
      .def_property_readonly("weight_hash",
                             &BuildMetaProto::mutable_weight_hash,
                             py::return_value_policy::reference)
      .def_property_readonly("torch_build_config", [](BuildMetaProto* self) {
        return MapPtr<std::string, std::string>(
            self->mutable_torch_build_config());
      });

  py::class_<TransformerProto>(m, "TransformerProto")
      .def(py::init<>())
      .def_property("model_type", &TransformerProto::model_type,
                    &TransformerProto::set_model_type<std::string>)
      .def_property_readonly("model_conf",
                             &TransformerProto::mutable_model_conf,
                             py::return_value_policy::reference)
      .def_property_readonly(
          "inputs",
          [](TransformerProto* self) {
            return RepeatedPtrFieldPtr<TensorProto>(self->mutable_inputs());
          })
      .def_property_readonly(
          "outputs",
          [](TransformerProto* self) {
            ;
            return RepeatedPtrFieldPtr<TensorProto>(self->mutable_outputs());
          })
      .def_property_readonly(
          "weights",
          [](TransformerProto* self) {
            return MapPtr<std::string, TensorProto>(self->mutable_weights());
          })
      .def_property_readonly(
          "graphs",
          [](TransformerProto* self) {
            return MapPtr<std::string, GraphProto>(self->mutable_graphs());
          })
      .def_property_readonly(
          "graph_names",
          [](TransformerProto* self) {
            return RepeatedPtrFieldPtr<std::string, std::string>(
                self->mutable_graph_names());
          })
      .def_property_readonly("build_meta",
                             &TransformerProto::mutable_build_meta,
                             py::return_value_policy::reference)
      .def("SerializeToString", [](TransformerProto* self) {
        std::string s;
        self->SerializeToString(&s);
        return py::bytes(s);
      });

  py::class_<GraphProto>(m, "GraphProto")
      .def(py::init<>())
      .def_property_readonly(
          "inputs",
          [](GraphProto* self) {
            return RepeatedPtrFieldPtr<TensorProto>(self->mutable_inputs());
          })
      .def_property_readonly(
          "outputs",
          [](GraphProto* self) {
            return RepeatedPtrFieldPtr<TensorProto>(self->mutable_outputs());
          })
      .def_property_readonly("ops", [](GraphProto* self) {
        return RepeatedPtrFieldPtr<OperatorProto>(self->mutable_ops());
      });

  py::class_<OperatorProto>(m, "OperatorProto")
      .def(py::init<>())
      .def_property("op_type", &OperatorProto::op_type,
                    &OperatorProto::set_op_type<std::string>)
      .def_property("op_name", &OperatorProto::op_name,
                    &OperatorProto::set_op_name<std::string>)
      .def_property_readonly(
          "attr",
          [](OperatorProto* self) {
            return MapPtr<std::string, std::string>(self->mutable_attr());
          })
      .def_property_readonly(
          "inputs",
          [](OperatorProto* self) {
            return RepeatedPtrFieldPtr<TensorProto>(self->mutable_inputs());
          })
      .def_property_readonly(
          "outputs",
          [](OperatorProto* self) {
            return RepeatedPtrFieldPtr<TensorProto>(self->mutable_outputs());
          })
      .def_property_readonly("weights", [](OperatorProto* self) {
        return RepeatedPtrFieldPtr<TensorProto>(self->mutable_weights());
      });
  py::class_<TensorProto>(m, "TensorProto")
      .def(py::init<>())
      .def_property("name", &TensorProto::name,
                    &TensorProto::set_name<std::string>)
      .def_property("data", &TensorProto::data,
                    &TensorProto::set_data<std::string>)
      .def("CopyFrom", [](TensorProto* self, const TensorProto* other) {
        self->CopyFrom(*other);
      });

  py::enum_<AsCacheMode>(m, "AsCacheMode")
      .value("AsCacheDefault", AsCacheMode::AsCacheDefault)
      .value("AsCacheQuantI8", AsCacheMode::AsCacheQuantI8)
      .value("AsCacheQuantU4", AsCacheMode::AsCacheQuantU4);
  py::enum_<AsMHAPrefill>(m, "AsMHAPrefill")
      .value("AsPrefillDefault", AsMHAPrefill::AsPrefillDefault)
      .value("AsPrefillFlashV2", AsMHAPrefill::AsPrefillFlashV2)
      .value("AsPrefillXformer", AsMHAPrefill::AsPrefillXformer);
  py::enum_<AsEvictionStrategy>(m, "AsEvictionStrategy")
      .value("MaxLength", AsEvictionStrategy::MaxLength)
      .value("Random", AsEvictionStrategy::Random);
  py::enum_<AsSchedulingStrategy>(m, "AsSchedulingStrategy")
      .value("ContextPriority", AsSchedulingStrategy::ContextPriority)
      .value("Balance", AsSchedulingStrategy::Balance);

  py::enum_<VocabType>(m, "VocabType")
      .value("VOCAB_TYPE_WPM", VocabType::VOCAB_TYPE_WPM)
      .value("VOCAB_TYPE_SPM", VocabType::VOCAB_TYPE_SPM)
      .value("VOCAB_TYPE_UGM", VocabType::VOCAB_TYPE_UGM)
      .value("VOCAB_TYPE_BPE", VocabType::VOCAB_TYPE_BPE);

  py::class_<AsFileInfo>(m, "AsFileInfo")
      .def(py::init<>())
      .def_readwrite("create_version_graph", &AsFileInfo::create_version_graph)
      .def_readwrite("current_version_engine",
                     &AsFileInfo::current_version_engine)
      .def_readwrite("create_version_param", &AsFileInfo::create_version_param);

  py::class_<AsModelConfig>(
      m, "AsModelConfig", py::module_local(),
      "This class define how engine handle the model including runtime "
      "configuration and optimize related configuration")
      .def(py::init<>())
      .def(py::init<>([](std::string model_name, std::string model_path,
                         std::string weights_path, std::string compute_unit,
                         int engine_max_length, int engine_max_batch,
                         int engine_max_prefill_length, int64_t swap_threshold,
                         bool text_graph, int num_threads,
                         std::string matmul_precision,
                         std::vector<std::string> lora_names,
                         int cache_span_size, int cache_span_num_init,
                         int cache_span_num_grow, bool enable_prefix_cache,
                         int prefix_cache_ttl, AsMHAPrefill prefill_mode,
                         AsCacheMode cache_mode,
                         AsEvictionStrategy eviction_strategy,
                         AsSchedulingStrategy scheduling_strategy,
                         bool enable_sparsity_matmul) {
             return new AsModelConfig(
                 model_name, model_path, weights_path, compute_unit,
                 engine_max_length, engine_max_batch, engine_max_prefill_length,
                 swap_threshold, text_graph, num_threads, matmul_precision,
                 lora_names, cache_span_size, cache_span_num_init,
                 cache_span_num_grow, enable_prefix_cache, prefix_cache_ttl,
                 prefill_mode, cache_mode, eviction_strategy,
                 scheduling_strategy, enable_sparsity_matmul);
           }),
           py::arg("model_name"), py::arg("model_path"),
           py::arg("weights_path"), py::arg("compute_unit") = "CUDA:0",
           py::arg("engine_max_length") = 0, py::arg("engine_max_batch") = 0,
           py::arg("engine_max_prefill_length") = 0,
           py::arg("swap_threshold") = -1, py::arg("text_graph") = false,
           py::arg("num_threads") = 0, py::arg("matmul_precision") = "highest",
           py::arg("lora_names") = std::vector<std::string>(),
           py::arg("cache_span_size") = AsModelConfig::default_span_size,
           py::arg("cache_span_num_init") = 0,
           py::arg("cache_span_num_grow") = 0,
           py::arg("enable_prefix_cache") = true,
           py::arg("prefix_cache_ttl") = 300,
           py::arg("prefill_mode") = AsMHAPrefill::AsPrefillDefault,
           py::arg("cache_mode") = AsCacheMode::AsCacheDefault,
           py::arg("eviction_strategy") = AsEvictionStrategy::MaxLength,
           py::arg("scheduling_strategy") =
               AsSchedulingStrategy::ContextPriority,
           py::arg("enable_sparsity_matmul") = false)
      .def(
          "__eq__",
          [](const AsModelConfig& self, const AsModelConfig& other) {
            return self == other;
          },
          py::is_operator())
      .def("__repr__", &AsModelConfig::ToString)
      // required field: model_name / model_path / weights_path
      .def_readwrite("model_name", &AsModelConfig::model_name)
      .def_readwrite("model_path", &AsModelConfig::model_path)
      .def_readwrite("weights_path", &AsModelConfig::weights_path)
      .def_readwrite("text_graph", &AsModelConfig::text_graph)
      .def_readwrite("compute_unit", &AsModelConfig::compute_unit)
      .def_readwrite("engine_max_length", &AsModelConfig::engine_max_length)
      .def_readwrite("engine_max_batch", &AsModelConfig::engine_max_batch)
      .def_readwrite("engine_max_prefill_length",
                     &AsModelConfig::engine_max_prefill_length)
      .def_readwrite("swap_threshold", &AsModelConfig::swap_threshold)
      .def_readwrite("num_threads", &AsModelConfig::num_threads)
      .def_readwrite("matmul_precision", &AsModelConfig::matmul_precision)
      .def_readwrite("lora_names", &AsModelConfig::lora_names)
      .def_readwrite("cache_span_size", &AsModelConfig::cache_span_size)
      .def_readwrite("cache_span_num_init", &AsModelConfig::cache_span_num_init)
      .def_readwrite("cache_span_num_grow", &AsModelConfig::cache_span_num_grow)
      .def_readwrite("enable_prefix_cache", &AsModelConfig::enable_prefix_cache)
      .def_readwrite("prefix_cache_ttl", &AsModelConfig::prefix_cache_ttl)
      .def_readwrite("prefill_mode", &AsModelConfig::prefill_mode)
      .def_readwrite("cache_mode", &AsModelConfig::cache_mode)
      .def_readwrite("eviction_strategy", &AsModelConfig::eviction_strategy)
      .def_readwrite("scheduling_strategy", &AsModelConfig::scheduling_strategy)
      .def_readwrite("enable_sparsity_matmul",
                     &AsModelConfig::enable_sparsity_matmul)
      .def_readwrite("lora_max_rank", &AsModelConfig::lora_max_rank)
      .def_readwrite("lora_max_num", &AsModelConfig::lora_max_num);

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
                     &AsEngineStat::used_device_memory_pool_size)
      .def_readwrite("prefix_cache_hit_token",
                     &AsEngineStat::prefix_cache_hit_token)
      .def_readwrite("prefix_cache_miss_token",
                     &AsEngineStat::prefix_cache_miss_token)
      .def_readwrite("prefix_cache_hit_rate",
                     &AsEngineStat::prefix_cache_hit_rate)
      .def_readwrite("prefix_cache_miss_rate",
                     &AsEngineStat::prefix_cache_miss_rate);

  py::class_<MultiMediaInfo>(m, "MultiMediaInfo")
      .def(py::init<>())
      // .def_readwrite("multimedia_type", &MultiMediaInfo::multimedia_type)
      .def("set_multimedia_type", &MultiMediaInfo::set_multimedia_type)
      .def("add_multimedia_content",
           [](MultiMediaInfo* self, std::string key,
              const std::vector<py::capsule>& dl_tensor_py_list) {
             std::vector<DLManagedTensor*> dl_tensor_list;
             for (const auto& dl_tensor_py : dl_tensor_py_list) {
               auto* dl_tensor = static_cast<DLManagedTensor*>(dl_tensor_py);
               dl_tensor_list.push_back(dl_tensor);
             }
             return self->add_multimedia_content(key, dl_tensor_list);
           });

  // Allspark API
  // We should make sure interfaces are the same with AsClientEngine
  py::class_<AsEngine>(m, "AsEngine")
      .def(py::init<>())
      .def("_build_model_from_as_model_config",
           [](AsEngine* self, py::object as_model_config_obj) {
             auto as_model_cfg = py::cast<AsModelConfig*>(as_model_config_obj);
             AS_CHECK(self->BuildModelFromConfigStruct(*as_model_cfg));
           })
      .def("unload_device_mem",
           [](AsEngine* self, const char* model_name) {
             py::gil_scoped_release release;
             AS_CHECK(self->UnloadModelFromDeviceMemory(model_name));
           })
      .def("load_device_mem",
           [](AsEngine* self, const char* model_name) {
             py::gil_scoped_release release;
             AS_CHECK(self->ReloadModelToDeviceMemory(model_name));
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
             DLTensorMap as_inputs;
             DLTensorMap as_outputs;
             std::map<std::string, py::capsule> py_outputs;
             PyParseConfig(gen_cfg, as_gen_cfg);
             PyParseInputs(inputs, as_inputs);
             PyInputReferenceGuard ref_guard(inputs);
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
      .def("load_lora",
           [](AsEngine* self, const char* model_name,
              const char* lora_name) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = AsStatus::ALLSPARK_SUCCESS;
             status = self->LoadLoraByName(model_name, lora_name);
             return status;
           })
      .def("unload_lora",
           [](AsEngine* self, const char* model_name,
              const char* lora_name) -> AsStatus {
             py::gil_scoped_release release;
             AsStatus status = AsStatus::ALLSPARK_SUCCESS;
             status = self->UnloadLoraByName(model_name, lora_name);
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
}

#undef CHECK_CONFIG
