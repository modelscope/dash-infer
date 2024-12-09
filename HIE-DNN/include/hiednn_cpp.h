/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_cpp.h
 */

/**
 * @file
 * HIE-DNN Operator API for CPP (Standard C++) Backend.
 *
 * @warning
 * CPP backend is at the early stage, only few operators are supported,
 * and the operators are not optimized.
 */

#ifndef HIEDNN_CPP_H_
#define HIEDNN_CPP_H_

#include "hiednn.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @enum hiednnAsmOpt_t
 * Enumerated type to enable assembly optimized kernel for specific platform.
 *
 * @var hiednnAsmOpt_t HIEDNN_ASM_OPTIMIZE_NONE
 * Disable assembly optimized kernel.
 *
 * @var hiednnAsmOpt_t HIEDNN_ASM_OPTIMIZE_X86
 * Enable assembly optimized kernel for x86 CPU.
 *
 * @var hiednnAsmOpt_t HIEDNN_ASM_OPTIMIZE_ARM
 * Enable assembly optimized kernel for ARM CPU.
 */
typedef enum {
    HIEDNN_ASM_OPTIMIZE_NONE  = 0,
    HIEDNN_ASM_OPTIMIZE_X86   = 1,
    HIEDNN_ASM_OPTIMIZE_ARM   = 2,
} hiednnAsmOpt_t;

/**
 * @typedef hiednnCppHandle_t
 * A pointer to an opaque structure holding HIE-DNN CPP operator context.
 */
typedef struct HiednnCppHandle *hiednnCppHandle_t;

/**
 * @brief
 * Create HIE-DNN CPP operator context.
 *
 * This function initialize the library and create an opaque structure holding
 * the HIE-DNN CPP operator context. It also allocate necessary resources of
 * context. This function must be called before any other HIE-DNN CPP functions
 * with a #hiednnCppHandle_t parameter. @n
 *
 * This Function is not thread safe.
 *
 * @warning
 * @c asmOpt option has no effect, because there's no assembly optimized kernel
 * for CPP backend so far.
 *
 * For operator functions with #hiednnCppHandle_t parameter, the context is
 * thread safe, but #hiednnCreateCppHandle and #hiednnDestroyCppHandle is not
 * thread safe.
 *
 * @param[in,out] cppHandle
 * Pointer to pointer to the HIE-DNN CPP operator context to be created.
 *
 * @param[in] asmOpt
 * Enumerator to enable assembly optimized kernel for specific platform.
 */
hiednnStatus_t
hiednnCreateCppHandle(hiednnCppHandle_t *cppHandle,
                      hiednnAsmOpt_t asmOpt);  

/**
 * @brief
 * Destroy HIE-DNN CPP operator context.
 *
 * This function release the resources hold by HIE-DNN CPP operator context,
 * and should be called after any other function calls associated with the
 * handle. @n
 *
 * This Function is not thread safe.
 *
 * @param[in] cppHandle
 * Pointer to the HIE-DNN CPP operator context to be destroyed.
 */
hiednnStatus_t
hiednnDestroyCppHandle(hiednnCppHandle_t cppHandle);

/**
 * @brief
 * Initialize the data of a tensor.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cppHandle
 * Previously created HIE-DNN CPP handle.
 *
 * @param[in] mode
 * Operation mode, refer to #hiednnSetTensorValueMode_t for more information.
 *
 * @param[in] p0, p1
 * Pointers to parameters for tensor initialization, refer to
 * #hiednnSetTensorValueMode_t for more information. @n
 *
 * For #HIEDNN_SETTENSOR_CONST mode, parameter passed by p0 should have same
 * data type as tensor y; @n
 *
 * For #HIEDNN_SETTENSOR_RANGE mode, parameters passed by p0 and p1 should both
 * have same data type as tensor y; @n
 *
 * For #HIEDNN_SETTENSOR_DIAGONAL mode, parameter passed by p0 should be a
 * int32 value and parameter passed by p1 should have same data type as
 * tensor y.
 *
 * @param[in] yDesc
 * Tensor descriptor of tensor @c y.
 *
 * @param[out] y
 * Pointer to the data of output tensor.
 */
hiednnStatus_t
hiednnCppSetTensorValue(hiednnCppHandle_t cppHandle,
                        hiednnSetTensorValueMode_t mode,
                        const void *p0,
                        const void *p1,
                        hiednnTensorDesc_t yDesc,
                        void *y);

/**
 * @brief
 * Unary map (1 input tensor, 1 output tensor) elementwise operator function.
 *
 * This function performs @c <tt>y = alpha * op(x) + y * beta</tt>, where @c op
 * was defined by @c unaryEltwiseOp. @n
 *
 * @c alpha and @c beta are scaling factors for numerical operators. For
 * operators with BOOL output, scaling factors are senseless and can pass
 * @a nullptr or @a NULL to @c alpha and @c beta in the parameter list.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cppHandle
 * Previously created HIE-DNN CPP handle.
 *
 * @param[in] unaryEltwiseOp
 * Enumerator to specify the operator, refer to #hiednnUnaryEltwiseOp_t for
 * more information.
 *
 * @param[in] alpha, beta
 * Scaling factors.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Pointer to the data of input tensor.
 *
 * @param[in] extParam1, extParam2
 * Optional parameters for operator, refer to #hiednnUnaryEltwiseOp_t for more
 * information.
 *
 * @param[out] y
 * Pointer to the data of output tensor.
 */
hiednnStatus_t
hiednnCppUnaryElementwiseOp(hiednnCppHandle_t cppHandle,
                            hiednnUnaryEltwiseOp_t unaryEltwiseOp,
                            const void *alpha,
                            hiednnTensorDesc_t xDesc,
                            const void *x,
                            const void *extParam1,
                            const void *extParam2,
                            const void *beta,
                            hiednnTensorDesc_t yDesc,
                            void *y);

/**
 * @brief
 * Data type conversion operator function.
 *
 * This function convert a tensor from data type specified by @c xDesc to the
 * data type specified by @c yDesc. @n
 *
 * Information in @c xDesc and @c yDesc should be same except data type.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cppHandle
 * Previously created HIE-DNN CPP handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Pointer to the data of input tensor.
 *
 * @param[out] y
 * Pointer to the data of output tensor.
 */
hiednnStatus_t
hiednnCppCast(hiednnCppHandle_t cppHandle,
              hiednnTensorDesc_t xDesc,
              const void *x,
              hiednnTensorDesc_t yDesc,
              void *y);

/**
 * @brief
 * Broadcast the input tensor to the dimension of output tensor follow the
 * Numpy-style broadcast rule.
 *
 * Dimension of @c xDesc should be broadcastable to that of @c yDesc. @n
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cppHandle
 * Previously created HIE-DNN CPP handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Pointer to the data of input tensor.
 *
 * @param[out] y
 * Pointer to the data of output tensor.
 */
hiednnStatus_t
hiednnCppExpand(hiednnCppHandle_t cppHandle,
                hiednnTensorDesc_t xDesc,
                const void *x,
                hiednnTensorDesc_t yDesc,
                void *y);

#if defined(__cplusplus)
}
#endif

#endif  // HIEDNN_CPP_H_


