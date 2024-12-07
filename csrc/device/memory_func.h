/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    memory_func.h
 */

#ifndef MEMORY_FUNC_H
#define MEMORY_FUNC_H

/**
 * @file   memory_func.h
 *
 * @brief This file collects the device memory related function
 *
 */

#include <common/common.h>
#include <common/device_context.h>

namespace allspark {

void MemsetZero(void* dst_data, DeviceType dst_device, int64_t nbytes);

}

#endif
