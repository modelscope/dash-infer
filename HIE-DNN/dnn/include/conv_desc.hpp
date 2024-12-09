/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    conv_desc.hpp
 */

#ifndef DNN_INCLUDE_CONV_DESC_HPP_
#define DNN_INCLUDE_CONV_DESC_HPP_

#include <hiednn.h>
#include <utils.hpp>

constexpr int CONV_DIM_MAX = HIEDNN_CONV_DIM_MAX;

struct HiednnConvolutionDesc {
    int convNDims;
    int64_t group;
    int64_t pad[CONV_DIM_MAX];
    int64_t stride[CONV_DIM_MAX];
    int64_t dilation[CONV_DIM_MAX];
    hiednnConvMode_t mode;
    hiednnDataType_t computeType;

    HiednnConvolutionDesc() :
            group(1), mode(HIEDNN_CONV_CROSS_CORRELATION) {}
};


#endif  // DNN_INCLUDE_CONV_DESC_HPP_
