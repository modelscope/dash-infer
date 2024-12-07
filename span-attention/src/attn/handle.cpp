/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    handle.cpp
 */

#include "attn/handle.hpp"

#include "common/data_type.h"
#include "common/enums.h"
#include "common/logger.h"

namespace span {

SpanAttnHandle::SpanAttnHandle(DataType dataType, QuantMode kvQuantMode,
                               int batchSize, int nHeads, int nGroups,
                               int headSize, int spanLen, int nSpansPerRequest,
                               const int* seqLen,
                               const cudaDeviceProp& deviceProp, bool forceSimt)
    : dataType_(dataType), spanAttnObj_(nullptr) {
  /*
   * It is tricky that we actually do NOT need typedObj in the lambda.
   * So we explicitly pass nullptr to make the compiler happy without confusing
   * maintainers.
   */
  dispatchTypeImpl(
      dataType_, nullptr, [=]<typename FType>(SpanAttn<FType>* /* typedObj */) {
        this->spanAttnObj_ = new SpanAttn<FType>(
            headSize, nHeads, nGroups, seqLen, spanLen, nSpansPerRequest,
            batchSize, kvQuantMode, deviceProp, forceSimt);
      });
}

SpanAttnHandle::~SpanAttnHandle() {
  dispatchTypeImpl(dataType_, spanAttnObj_,
                   [=]<typename FType>(SpanAttn<FType>* typedObj) {
                     delete typedObj;
                     this->spanAttnObj_ = nullptr;
                   });
}

}  // namespace span
