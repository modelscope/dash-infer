/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    copy_var_array.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T>
void CopyToArrayKernelLauncher(T* array, const T** vars, int count) {
  for (int i = 0; i < count; i++) {
    array[i] = *vars[i];
  }
}

template void CopyToArrayKernelLauncher<int>(int* array, const int** vars,
                                             int count);

template void CopyToArrayKernelLauncher<int64_t>(int64_t* array,
                                                 const int64_t** vars,
                                                 int count);

template void CopyToArrayKernelLauncher<float>(float* array, const float** vars,
                                               int count);

//

template <typename T>
void CopyToVarsKernelLauncher(T** vars, const T* array, int count) {
  for (int i = 0; i < count; i++) {
    *vars[i] = array[i];
  }
}

template void CopyToVarsKernelLauncher<int>(int** vars, const int* array,
                                            int count);

template void CopyToVarsKernelLauncher<int64_t>(int64_t** vars,
                                                const int64_t* array,
                                                int count);

template void CopyToVarsKernelLauncher<float>(float** vars, const float* array,
                                              int count);

}  // namespace cpu
}  // namespace allspark
