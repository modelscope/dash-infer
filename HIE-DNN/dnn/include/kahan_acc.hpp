/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kahan_acc.hpp
 */

#ifndef DNN_INCLUDE_KAHAN_ACC_HPP_
#define DNN_INCLUDE_KAHAN_ACC_HPP_

#include <device_function_modifier.hpp>

namespace hiednn {

template <typename AccT>
class KahanAcc {
 private:
    AccT err_;

 public:
    AccT sum;

    DEVICE_FUNCTION KahanAcc() : err_(0), sum(0) {}

    template <typename T>
    DEVICE_FUNCTION void Acc(const T &x) {
        AccT y = x - err_;
        AccT t = sum + y;
        err_ = (t - sum) - y;
        sum = t;
    }
};

}  // namespace hiednn

#endif  // DNN_INCLUDE_KAHAN_ACC_HPP_

