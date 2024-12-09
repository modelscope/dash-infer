/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_cpp.cpp
 */

#include <hiednn.h>
#include <hiednn_cpp.h>

#include <cpp/cpp_handle.hpp>

hiednnStatus_t
hiednnCreateCppHandle(HiednnCppHandle **handle, hiednnAsmOpt_t asmOpt) {
    *handle = new HiednnCppHandle();
    (*handle)->asmOpt = asmOpt;
    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnDestroyCppHandle(HiednnCppHandle *handle) {
    if (handle != nullptr) {
        delete handle;
        return HIEDNN_STATUS_SUCCESS;
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }
}

