/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    handle.hpp
 */

#pragma once

#include <span_attn.h>

#include "attn/span_attention.hpp"
#include "common/data_type.h"
#include "common/enums.h"

namespace span {

struct SpanAttnHandle {
  SpanAttnHandle(DataType dataType, QuantMode kvQuantMode, int batchSize,
                 int nHeads, int nGroups, int headSize, int spanLen,
                 int nSpansPerRequest, const int* seqLen,
                 const cudaDeviceProp& deviceProp, bool forceSimt);
  ~SpanAttnHandle();

  void* Object() const { return spanAttnObj_; }
  DataType Type() const { return dataType_; }

  template <typename Func, typename... Args>
  void DispatchType(Func&& func, Args&&... args) {
    dispatchTypeImpl(Type(), Object(), std::forward<Func>(func),
                     std::forward<Args>(args)...);
    return;
  }

 private:
  DataType dataType_;
  void* spanAttnObj_;

  template <typename Func, typename... Args>
  static void dispatchTypeImpl(DataType dtype, void* obj, Func&& func,
                               Args&&... args) {
    switch (dtype) {
      case DataType::FP32: {
        using FType = TypeAdapter<(DataType::FP32)>::Type;
        SpanAttn<FType>* typedObj = static_cast<SpanAttn<FType>*>(obj);
        std::forward<Func>(func).template operator()<FType>(
            typedObj, std::forward<Args>(args)...);
        break;
      }
#ifdef ENABLE_FP16
      case DataType::FP16: {
        using FType = TypeAdapter<(DataType::FP16)>::Type;
        SpanAttn<FType>* typedObj = static_cast<SpanAttn<FType>*>(obj);
        std::forward<Func>(func).template operator()<FType>(
            typedObj, std::forward<Args>(args)...);
        break;
      }
#endif
#ifdef ENABLE_BF16
      case DataType::BF16: {
        using FType = TypeAdapter<(DataType::BF16)>::Type;
        SpanAttn<FType>* typedObj = static_cast<SpanAttn<FType>*>(obj);
        std::forward<Func>(func).template operator()<FType>(
            typedObj, std::forward<Args>(args)...);
        break;
      }
#endif
      default: {
        LOG(ERROR) << "SpanAttnHandle: unsupported data type: "
                   << to_string(dtype) << std::endl;
        throw SpanAttnError(SaStatus::PARAM_ERROR,
                            "unsupported data type: " + to_string(dtype));
      }
    }
    return;
  }
};

}  // namespace span
