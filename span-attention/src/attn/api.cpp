/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    api.cpp
 */

#include "attn/handle.hpp"
#include "attn/span_attention.hpp"
#include "common/data_type.h"
#include "common/enums.h"
#include "common/error.h"
#include "common/logger.h"

namespace span {

namespace {

// func should take NO parameter, and returns SaStatus
template <typename Func>
SaStatus api_body(Func&& func, const char* caller_name = __builtin_FUNCTION()) {
  SaStatus ret = SaStatus::SUCCESS;
  try {
    ret = func();
  } catch (const SpanAttnError& e) {
    LOG(ERROR) << caller_name << " catches " << GetErrorName(e.code()) << ": "
               << e.what() << std::endl;
    return e.code();
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << caller_name << " catches runtime error: " << e.what()
               << std::endl;
    return SaStatus::RUNTIME_ERROR;
  } catch (const std::exception& e) {
    LOG(ERROR) << caller_name << " catches exception: " << e.what()
               << std::endl;
    return SaStatus::UNKNOWN_ERROR;
  }
  return ret;
}

}  // namespace

SaStatus CreateHandle(SpanAttnHandle** handle, DataType dataType,
                      QuantMode kvQuantMode, int batchSize, int nHeads,
                      int nGroups, int headSize, int spanLen,
                      int nSpansPerRequest, const int* seqLen,
                      const cudaDeviceProp& deviceProp /* , bool forceSimt */) {
  if (batchSize <= 0 || nHeads <= 0 || nGroups <= 0 || headSize <= 0 ||
      spanLen <= 0 || nSpansPerRequest <= 0) {
    LOG(ERROR) << "CreateHandle: invalid parameter, got batchSize=" << batchSize
               << " nHeads=" << nHeads << " nGroups=" << nGroups
               << " headSize=" << headSize << " spanLen=" << spanLen
               << " nSpansPerRequest=" << nSpansPerRequest << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (nHeads % nGroups != 0) {
    LOG(ERROR)
        << "CreateHandle: nHeads should be a multiple of nGroups, got nHeads="
        << nHeads << " nGroups=" << nGroups << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (seqLen == nullptr) {
    LOG(ERROR) << "CreateHandle: seqLen must not be null" << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (handle == nullptr) {
    LOG(ERROR) << "CreateHandle: pointer to the handle must not be null"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  return api_body([=]() -> SaStatus {
    *handle = new SpanAttnHandle(dataType, kvQuantMode, batchSize, nHeads,
                                 nGroups, headSize, spanLen, nSpansPerRequest,
                                 seqLen, deviceProp, false);
    return SaStatus::SUCCESS;
  });
}

SaStatus DestroyHandle(SpanAttnHandle* handle) {
  if (handle == nullptr) {
    LOG(ERROR) << "DestroyHandle: handle must not be null" << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  return api_body([=]() -> SaStatus {
    delete handle;
    return SaStatus::SUCCESS;
  });
}

SaStatus GetHostWorkspaceSize(size_t* wsInBytes, SpanAttnHandle* handle) {
  if (handle == nullptr) {
    LOG(ERROR) << "GetHostWorkspaceSize: handle must not be null" << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (wsInBytes == nullptr) {
    LOG(ERROR) << "GetHostWorkspaceSize: pointer to the workspace size must "
                  "not be null"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  return api_body([=]() -> SaStatus {
    handle->DispatchType([=]<typename FType>(SpanAttn<FType>* typedObj) {
      *wsInBytes = typedObj->GetHostWorkspaceSize();
    });
    return SaStatus::SUCCESS;
  });
}

SaStatus GetDeviceWorkspaceSize(size_t* wsInBytes, SpanAttnHandle* handle) {
  if (handle == nullptr) {
    LOG(ERROR) << "GetDeviceWorkspaceSize: handle must not be null"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (wsInBytes == nullptr) {
    LOG(ERROR) << "GetDeviceWorkspaceSize: pointer to the workspace size must "
                  "not be null"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  return api_body([=]() -> SaStatus {
    handle->DispatchType([=]<typename FType>(SpanAttn<FType>* typedObj) {
      *wsInBytes = typedObj->GetDeviceWorkspaceSize();
    });
    return SaStatus::SUCCESS;
  });
}

SaStatus Run(void* output, const void* query, const void* const* kSpanArray,
             const void* const* vSpanArray, void* deviceWorkspace,
             size_t deviceWsSizeInBytes, void* hostWorkspace,
             size_t hostWsSizeInBytes, float QKScale, SpanAttnHandle* handle,
             cudaStream_t stream) {
  if (handle == nullptr) {
    LOG(ERROR) << "Run: handle must not be null" << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (output == nullptr || query == nullptr || kSpanArray == nullptr ||
      vSpanArray == nullptr) {
    LOG(ERROR) << "Run: input and output pointers must not be null"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (deviceWorkspace == nullptr && deviceWsSizeInBytes > 0) {
    LOG(ERROR) << "Run: device workspace pointer must not be null if device "
                  "workspace size is non-zero"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  if (hostWorkspace == nullptr && hostWsSizeInBytes > 0) {
    LOG(ERROR) << "Run: host workspace pointer must not be null if host "
                  "workspace size is non-zero"
               << std::endl;
    return SaStatus::PARAM_ERROR;
  }

  return api_body([=]() -> SaStatus {
    SaStatus ret = SaStatus::SUCCESS;
    handle->DispatchType([=, &ret]<typename FType>(SpanAttn<FType>* typedObj) {
      ret = typedObj->Run(static_cast<const FType*>(query), kSpanArray,
                          vSpanArray, static_cast<FType*>(output),
                          deviceWorkspace, deviceWsSizeInBytes, hostWorkspace,
                          hostWsSizeInBytes, QKScale, stream);
    });
    return ret;
  });
}

const char* GetErrorName(SaStatus code) { return to_string(code).c_str(); }

const char* GetErrorString(SaStatus code) {
  switch (code) {
    case SaStatus::SUCCESS:
      return "success";
    case SaStatus::CUDA_ERROR:
      return "CUDA runtime API error";
    case SaStatus::RUNTIME_ERROR:
      return "runtime error";
    case SaStatus::PARAM_ERROR:
      return "invalid parameter";
    case SaStatus::EXCEED_LIMIT_ERROR:
      return "parameter exceeding numerical limits";
    case SaStatus::INTERNAL_ERROR:
      return "internal error";
    case SaStatus::UNKNOWN_ERROR:
      /* fall through */
    default:
      return "unrecognized error";
  }
}

}  // namespace span
