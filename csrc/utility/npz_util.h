/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    npz_util.h
 */

#pragma once
#include <core/tensor/tensor.h>

namespace allspark {
namespace util {

// 仅支持numpy中row major数据的加载，即 order = 'C'
// 仅支持npz中未被压缩的数据解析，不支持np.savez_compress()保存的npz文件
void npz_load(const std::string& file_path, TensorMap& data,
              DeviceType device_type);
void npz_loads(const std::string& bin_data, TensorMap& data,
               DeviceType device_type);
}  // namespace util
}  // namespace allspark