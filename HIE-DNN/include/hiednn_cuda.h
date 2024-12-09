/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_cuda.h
 */

/**
 * @file
 * HIE-DNN Operator API for CUDA Backend.
 */

#ifndef HIEDNN_CUDA_H_
#define HIEDNN_CUDA_H_

#include "hiednn.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @typedef hiednnCudaHandle_t
 * A pointer to an opaque structure holding HIE-DNN CUDA operator context.
 */
typedef struct HiednnCudaHandle *hiednnCudaHandle_t;

/**
 * @brief
 * Create HIE-DNN CUDA operator context.
 *
 * This function initialize the library and create an opaque structure holding
 * the HIE-DNN CUDA operator context. It also allocate necessary resources on
 * host and device. This function must be called before any other HIE-DNN CUDA
 * functions with a #hiednnCudaHandle_t parameter.
 *
 * HIE-DNN handle is tied to the current device. You can call @b cudaSetDevice
 * or @b cuCtxSetCurrent before #hiednnCreateCudaHandle to create a handle on a
 * specific devicve. For multi device(nvidia GPU) platform, #hiednnCudaHandle_t
 * should be created for each GPU.
 *
 * For a give device, multiple handles can be created (for example, create
 * multiple handles and bind to various CUDA streams). To release the resources,
 * #hiednnDestroyCudaHandle should be called and @b cudaDeviceSynchronize is
 * implicitly called in #hiednnDestroyCudaHandle, so #hiednnCreateCudaHandle and
 * #hiednnDestroyCudaHandle should not be called in hot-spot code paths.
 *
 * The context is not thread-safe, for a multithread program with the same
 * device shared by threads, it's recommended to create a handle for each thread.
 *
 * For programs with the same device shared by multi CUDA streams, it's
 * recommended to create a handle for each CUDA stream.
 *
 * @param[in,out] cudaHandle
 * Pointer to pointer to the HIE-DNN CUDA operator context to be created.
 */
hiednnStatus_t
hiednnCreateCudaHandle(hiednnCudaHandle_t *cudaHandle);

/**
 * @brief
 * Destroy HIE-DNN CUDA operator context.
 *
 * This function release the resources hold by HIE-DNN CUDA operator context,
 * and should be called after any other function calls associated with the
 * handle.
 *
 * @b cudaDeviceSynchronize is implicitly called in #hiednnDestroyCudaHandle,
 * so #hiednnDestroyCudaHandle should not be called in hot-spot code paths.
 *
 * @param[in] cudaHandle
 * Pointer to the HIE-DNN CUDA operator context to be destroyed.
 */
hiednnStatus_t
hiednnDestroyCudaHandle(hiednnCudaHandle_t cudaHandle);

/**
 * @brief
 * Bind CUDA handle to a specific CUDA stream
 *
 * Bind CUDA handle to a specific CUDA stream, and the stream will be used to
 * launch CUDA kernels in operator functions. If the HIE-DNN stream is not set
 * by #hiednnSetCudaStream, all kernels use the default stream.
 *
 * Operators from the same handle may be issued in-order, even if they are
 * attached with different streams by #hiednnSetCudaStream. For multiple stream
 * concurrency, it's recommended to create a handle for each stream.
 *
 * It's not recommended to call this function in hot-spot code paths.
 *
 * @param cudaHandle
 * Pointer to HIE-DNN CUDA operator context.
 *
 * @param cudaStream
 * CUDA stream bound to the handle.
 */
hiednnStatus_t
hiednnSetCudaStream(hiednnCudaHandle_t cudaHandle, cudaStream_t cudaStream);

/**
 * @brief
 * Initialize the data of a tensor.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] mode
 * Operation mode, refer to #hiednnSetTensorValueMode_t for more information.
 *
 * @param[in] p0, p1
 * Host memory pointers to parameters for tensor initialization, refer to
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
 * Pointer to GPU memory associated with output tensor.
 */
hiednnStatus_t
hiednnCudaSetTensorValue(hiednnCudaHandle_t cudaHandle,
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
 * was defined by @c unaryEltwiseOp.
 *
 * @c alpha and @c beta are scaling factors for numerical operators. For
 * operators with BOOL output, scaling factors are senseless and can pass
 * @a nullptr or @a NULL to @c alpha and @c beta in the parameter list.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
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
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] extParam1, extParam2
 * Optional parameters for operator, refer to #hiednnUnaryEltwiseOp_t for more
 * information.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaUnaryElementwiseOp(hiednnCudaHandle_t cudaHandle,
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
 * Binary map (2 input tensor, 1 output tensor) elementwise operator function.
 *
 * This function performs @c <tt>z = alpha * op(x, y) + z * beta</tt>, where
 * @c op was defined by @c binaryEltwiseOp.
 *
 * @c alpha and @c beta are scaling factors for numerical operators. For
 * operators with BOOL input or output, scaling factors are senseless and can
 * pass @a nullptr or @a NULL to @c alpha and @c beta in the parameter list.
 *
 * This function supports Numpy-style broadcast.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] binaryEltwiseOp
 * Enumerator to specify the operator, refer to #hiednnBinaryEltwiseOp_t for
 * more information.
 *
 * @param[in] alpha, beta
 * Scaling factors.
 *
 * @param[in] xDesc, yDesc, zDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x, y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] extParam
 * Optional parameter for operator, refer to #hiednnBinaryEltwiseOp_t for more
 * information.
 *
 * @param[out] z
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaBinaryElementwiseOp(hiednnCudaHandle_t cudaHandle,
                              hiednnBinaryEltwiseOp_t binaryEltwiseOp,
                              const void *alpha,
                              hiednnTensorDesc_t xDesc,
                              const void *x,
                              hiednnTensorDesc_t yDesc,
                              const void *y,
                              const void *extParam,
                              const void *beta,
                              hiednnTensorDesc_t zDesc,
                              void *z);

/**
 * @brief
 * Data type conversion operator function.
 *
 * This function convert a tensor from data type specified by @c xDesc to the
 * data type specified by @c yDesc.
 *
 * Information in @c xDesc and @c yDesc should be same except data type.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaCast(hiednnCudaHandle_t cudaHandle,
               hiednnTensorDesc_t xDesc,
               const void *x,
               hiednnTensorDesc_t yDesc,
               void *y);

/**
 * @brief
 * Broadcast the input tensor to the dimension of output tensor follow the
 * Numpy-style broadcast rule.
 *
 * Dimension of @c xDesc should be broadcastable to that of @c yDesc.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaExpand(hiednnCudaHandle_t cudaHandle,
                 hiednnTensorDesc_t xDesc,
                 const void *x,
                 hiednnTensorDesc_t yDesc,
                 void *y);

/**
 * @brief
 * Perform elements cumulative sum for the input tensor along specific axis.
 *
 * This function support inclusive/exclusive, prefix/suffix scan.
 *
 * This function only support tensors with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] axis
 * Specify the scanning(accumulation) axis.
 *
 * @param[in] exclusive
 * @c exclusive can be 0 or 1 to indicate this function perform inclusive or
 * exclusive accumulation. @n
 *
 * For prefix scan, if @c exclusive is 0, @c <tt>y[i] = x[0] + x[1] + ... +
 * x[i]</tt>, if @c exclusive is 1, @c <tt>y[i] = x[0] + x[1] + ... +
 * x[i-1]</tt>.
 *
 * @param[in] reverse
 * @c reverse can be 0 or 1 to indicate this function perform prefix or suffix
 * scan>
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaPrefixSum(hiednnCudaHandle_t cudaHandle,
                    hiednnTensorDesc_t xDesc,
                    const void *x,
                    int axis,
                    int exclusive,
                    int reverse,
                    hiednnTensorDesc_t yDesc,
                    void *y);

/**
 * @brief
 * Create a copy of input tensor and update values at specific positions.
 *
 * Output tensor @c y (rank @a r) is a copy of input tensor @c x, and update
 * values of tensor @c y at position specified by tensor @c indices (rank @a q)
 * to values specified by tensor @c updates (rank <em>q + r - indices.dim[-1] -
 * 1</em>).
 *
 * Tensor @c indices should not have duplicated values. or the output is
 * undefined.
 *
 * This function performs an inplace operation if pointer @c x and @c y is the
 * same.
 *
 * This function only support tensors with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] indicesDesc
 * Tensor descriptor for tensor @c indices. This function support INT32, UINT32,
 * INT64 indices.
 *
 * @param[in] updatesDesc
 * Tensor descriptor for tensor @c updates. @c updates should have same data
 * type with tensor @c x and @c y.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] indices
 * Device memory (or device addressable page locked memory) pointer to the data
 * of tensor @c indices.
 *
 * @param[in] updates
 * Device memory (or device addressable page locked memory) pointer to the data
 * of tensor @c updates.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaScatterND(hiednnCudaHandle_t cudaHandle,
                    hiednnTensorDesc_t xDesc,
                    const void *x,
                    hiednnTensorDesc_t indicesDesc,
                    const void *indices,
                    hiednnTensorDesc_t updatesDesc,
                    const void *updates,
                    hiednnTensorDesc_t yDesc,
                    void *y);

/**
 * @brief
 * Resize (upsampling or downsampling) the input tensor to the dimension of
 * output tensor by linear interpolation.
 *
 * This function support 1-, 2- or 3-linear interpolation.
 *
 * For n-linear interpolation, the rank of input and output tensor should be
 * @c n+1, where the outermost dimension is batch size. For example, perform a
 * batched 2-linear(bilinear) interpolation on @c M0*N0 sized 2D image, and
 * resize it to size @c M1*N1, the dimension of tensor @c x should be
 * @c <tt>{batch, M0, N0}</tt> and that of tensor @c y should be @c <tt>{batch,
 * M1, N1}</tt>.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout
 * and floating point data types (FP64, FP32, FP16, BF16..).
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] coordMode
 * Enumerator to indicate how to transform coordinate in the output tensor to
 * input tensor, refer to #hiednnInterpCoordMode_t for more information.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] scale
 * Host memory pointer to interpolation scaling factor array. For n-linear
 * interpolation, array @c scale should contain @c n non-negative floating
 * point numbers. Scale value greater than 1 means upsampling, otherwise
 * means downsampling. @n
 *
 * For FP64 tensor, @c scale should be 64-bit floating point (double). For
 * FP32/FP16/BF16 tensor, @c scale should be 32-bit floating point (float).
 *
 * @param[in] scaleSize
 * Size of @c scale array.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaLinearInterpolation(hiednnCudaHandle_t cudaHandle,
                              hiednnInterpCoordMode_t coordMode,
                              hiednnTensorDesc_t xDesc,
                              const void *x,
                              const void *scale,
                              int scaleSize,
                              hiednnTensorDesc_t yDesc,
                              void *y);

/**
 * @brief
 * Resize (upsampling or downsampling) the input tensor to the dimension of
 * output tensor by nearest interpolation.
 *
 * This function support 1D, 2D or 3D nearest interpolation.
 *
 * For nD interpolation, the rank of input and output tensor should be
 * @c n+1, where the outermost dimension is batch size. For example, perform a
 * batched 2D nearest interpolation on @c M0*N0 sized 2D image, and resize it
 * to size @c M1*N1, the dimension of tensor @c x should be @c <tt>{batch, M0,
 * N0}</tt> and that of tensor @c y should be @c <tt>{batch, M1, N1}</tt>.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] coordMode
 * Enumerator to indicate how to transform coordinate in the output tensor to
 * input tensor, refer to #hiednnInterpCoordMode_t for more information.
 *
 * @param[in] nearestMode
 * Enumerator to indicate how to get the index of nearest pixel. For a pixel
 * in output tensor with index @a idx, the index of nearest pixel in input
 * tensor can calculated by <em>nearest(coordinate(idx))</em>. For more
 * information about @a coordinate and @a nearest, refer to
 * #hiednnInterpCoordMode_t and #hiednnInterpNearestMode_t.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] scale
 * Host memory pointer to interpolation scaling factor array. For nD nearest
 * interpolation, array @c scale should contain @c n non-negative 32-bit
 * floating point numbers (float). Scale value greater than 1 means upsampling,
 * otherwise means downsampling.
 *
 * @param[in] scaleSize
 * Size of @c scale array.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaNearestInterpolation(hiednnCudaHandle_t cudaHandle,
                               hiednnInterpCoordMode_t coordMode,
                               hiednnInterpNearestMode_t nearestMode,
                               hiednnTensorDesc_t xDesc,
                               const void *x,
                               const float *scale,
                               int scaleSize,
                               hiednnTensorDesc_t yDesc,
                               void *y);

/**
 * @brief
 * Resize (upsampling or downsampling) the input tensor to the dimension of
 * output tensor by cubic interpolation.
 *
 * This function support 1D, 2D or 3D cubic interpolation.
 *
 * For nD cubic interpolation, the rank of input and output tensor should be
 * @c n+1, where the outermost dimension is batch size. For example, perform a
 * batched 2D cubic interpolation on @c M0*N0 sized 2D image, and resize it to
 * size @c M1*N1, the dimension of tensor @c x should be @c <tt>{batch, M0,
 * N0}</tt> and that of tensor @c y should be @c <tt>{batch, M1, N1}</tt>.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout
 * and floating point data types (FP64, FP32, FP16, BF16..).
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] coordMode
 * Enumerator to indicate how to transform coordinate in the output tensor to
 * input tensor, refer to #hiednnInterpCoordMode_t for more information.
 *
 * @param[in] cubicCoefficient
 * Host memory pointer to a floating point value. This is a parameter for cubic
 * interpolation weights calculation, two common values are -0.5 (in some cases
 * of TensorFlow) and -0.75 (in PyTorch). Refer to
 * https://ieeexplore.ieee.org/document/1163711 for more information. @n
 *
 * For FP64 tensor, @c cubicCoefficient should be 64-bit floating point
 * (double). For FP32/FP16/BF16 tensor, @c cubicCoefficient should be 32-bit
 * floating point (float).
 *
 * @param[in] excludeOutside
 * @c excludeOutside can be 0 or 1. If set to 1, the weight of sampling
 * locations outside the tensor @c x will be set to 0 and the weight will be
 * renormalized so that their sum is 1.0.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] scale
 * Host memory pointer to interpolation scaling factor array. For nD cubic
 * interpolation, array @c scale should contain @c n non-negative floating
 * point numbers. Scale value greater than 1 means upsampling, otherwise
 * means downsampling. @n
 *
 * For FP64 tensor, @c scale should be 64-bit floating point (double). For
 * FP32/FP16/BF16 tensor, @c scale should be 32-bit floating point (float).
 *
 * @param[in] scaleSize
 * Size of @c scale array.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaCubicInterpolation(hiednnCudaHandle_t cudaHandle,
                             hiednnInterpCoordMode_t coordMode,
                             const void *cubicCoefficient,
                             int excludeOutside,
                             hiednnTensorDesc_t xDesc,
                             const void *x,
                             const void *scale,
                             int scaleSize,
                             hiednnTensorDesc_t yDesc,
                             void *y);

/**
 * @brief
 * Perform a reduction along a specific axis on input tensor.
 * @c <tt>y = alpha * reduce(x)</tt> where @c reduce was defined by parameter
 * @c reduceOp.
 *
 * Supported data type configuration: @n
 * @n
 * SUM/PROD/SUM_ABS/SUM_SQUARE: @n
 * [TOC]
 * x types | compute precision | y types
 * --------|-------------------|---------------------------------------
 * INT8    | INT32             | INT8, INT16, INT32, FP16, BF16, FP32
 * UINT8   | UINT32            | UINT8, UINT16, UINT32, FP16, BF16, FP32
 * INT16   | INT32             | INT16, INT32
 * UINT16  | UINT32            | UINT16, UINT32
 * INT32   | INT32             | INT32
 * UINT32  | UINT32            | UINT32
 * INT64   | INT64             | INT64
 * UIN64   | UINT64            | UINT64
 * FP16    | FP32              | FP16, FP32
 * BF16    | FP32              | BF16, FP32
 * FP32    | FP32              | FP32
 * FP64    | FP64              | FP64
 *
 * @n
 * SQRT_SUM_SQUARE/LOG_SUM/LOG_SUM_EXP: @n
 * [TOC]
 * x types                                     | compute precision | y types
 * --------------------------------------------|-------------------|--------
 * INT8, UINT8                                 | FP32 | FP16, BF16, FP32
 * INT16, UINT16, INT32, UINT32, INT64, UINT64 | FP32 | FP32
 * FP16                                        | FP32 | FP16, FP32
 * BF16                                        | FP32 | BF16, FP32
 * FP32                                        | FP32 | FP32
 * FP64                                        | FP64 | FP64
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @n
 * MAX/MIN: @n
 * Support any data types except BOOL, input and output tensor should have same
 * data type.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] reduceOp
 * Enumerator to indicate the reduce operator, refer to #hiednnReduceOp_t for
 * more information.
 *
 * @param[in] alpha
 * Scaling factor, have same data type with tensor @c y.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors. @n
 * Each dimension of @c yDesc must match the corresponding dimension of @c xDesc
 * except the @c axis dimension, the @c axis dimension of @c yDesc must be 1.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] axis
 * Specify the reduced axis.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 *
 * @param[in] indicesType (optional)
 * Only work for MAX/MIN, specify the data type of @c indices. @n
 * This function support UINT8, UINT16, UINT32, UINT64 indices.
 *
 * @param[out] indices (optional)
 * Device memory (or device addressable page locked memory) pointer to indices,
 * whitch containing the index of the maximum/minimum element relative to the
 * reduced dimension. @n
 * Pass @a nullptr or @a NULL to @c indices indicate do not generate indices.
 */
hiednnStatus_t
hiednnCudaReduce(hiednnCudaHandle_t cudaHandle,
                 hiednnReduceOp_t reduceOp,
                 const void *alpha,
                 hiednnTensorDesc_t xDesc,
                 const void *x,
                 int axis,
                 hiednnTensorDesc_t yDesc,
                 void *y,
                 hiednnDataType_t indicesType,
                 void *indices);

/**
 * @brief
 * Perform a maximum or minimum reduction along a specific axis on input tensor
 * and return the index of the maximum or minimum element.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] reduceOp
 * Enumerator to indicate the reduce operator, refer to #hiednnReduceOp_t for
 * more information. @n
 * This function only support #HIEDNN_REDUCE_MAX and #HIEDNN_REDUCE_MIN.
 *
 * @param[in] xDesc
 * Tensor descriptor for input tensor.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] axis
 * Specify the reduced axis.
 *
 * @param[in] indicesDesc
 * Tensor descriptor for output tensor (index tensor). @n
 * Each dimension of @c indicesDesc must match the corresponding dimension of
 * @c xDesc except the @c axis dimension, the @c axis dimension of @c yDesc must
 * be 1. @n
 * This function support UINT64, UINT32, UINT16 and UINT8 indices.
 *
 * @param[out] indices
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output index tensor.
 */
hiednnStatus_t
hiednnCudaReduceIndex(hiednnCudaHandle_t cudaHandle,
                      hiednnReduceOp_t reduceOp,
                      hiednnTensorDesc_t xDesc,
                      const void *x,
                      int axis,
                      hiednnTensorDesc_t indicesDesc,
                      void *indices);

/**
 * @brief
 * Produces a slice of the input tensor @c x along multiple @c axes
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] starts
 * Host memory pointer to start indices of corresponding axis in @c axes.
 *
 * @param[in] ends
 * Host memory pointer to end indices (exclusive) of corresponding axis
 * in @c axes.
 *
 * @param[in] steps (optional)
 * Host memory pointer to slice step of corresponding axis in @c axes. @n
 * Negative value means slicing backward. @n
 * @c steps cannot be 0. @n
 * Defaults to 1's.
 *
 * @param[in] axes (optional)
 * Host memory pointer to axes that @c starts and @c ends apply to. @n
 * Negative value means counting dimensions from the back. @n
 * Accepted range is [-r, r-1] where r = rank( @c x ). @n
 * Repeated axis is not allowed.
 *
 * @param[in] nParams
 * Length of @c starts (as well as @c ends , @c steps , and @c axes ,
 * if applicable)
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaSlice(hiednnCudaHandle_t cudaHandle,
                hiednnTensorDesc_t xDesc,
                const void *x,
                const int64_t *starts,
                const int64_t *ends,
                const int64_t *steps,
                const int *axes,
                int nParams,
                hiednnTensorDesc_t yDesc,
                void *y);

/**
 * @brief
 * Pad the input tensor according to the parameter @c mode @c pads.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle
 *
 * @param[in] mode
 * Padding mode, refer to #hiednnPadMode_t for more information.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for input and output tensors.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] pads
 * Host memory pointer to the number of elements to add at the beginning and
 * end of each axis. For a N-Dim input tensor, @c pads should contain 2*N
 * elements and the format should be {dim0_begin, dim1_begin, ..., dim0_end,
 * dim1_end, ...}. Values pointed by @c pads should be nonnegative.
 *
 * @param[in] nPads
 * Number of elements in @c pads. For a N-Dim input tensor, @c nPads should be
 * 2*N.
 *
 * @param[in] param
 * Reversed parameter for some padding modes, refer to #hiednnPadMode_t for
 * more information.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaPad(hiednnCudaHandle_t cudaHandle,
              hiednnPadMode_t mode,
              hiednnTensorDesc_t xDesc,
              const void *x,
              const int64_t *pads,
              int nPads,
              const void *param,
              hiednnTensorDesc_t yDesc,
              void *y);

/**
 * @brief
 * Index into the input @c x tensor at index positions determined by elements
 * of the @c indices tensor.
 *
 * GatherElements takes two inputs @c x and @c indices of the same rank r >= 1,
 * and an attribute @c axis that identifies an axis of @c x .
 * Its output shape is the same as the shape of @c indices and consists of one
 * value (gathered from @c x ) for each element in indices.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc
 * Tensor descriptor for input tensor @c x whose rank r >= 1.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] indicesDesc
 * Tensor descriptor for int32/int64 indexing tensor @c indices which is of the
 * same rank r as the input @c x .
 *
 * @param[in] indices
 * Device memory (or device addressable page locked memory) pointer to the data
 * of indexing tensor. All index values are expected to be within bounds
 * [-s, s-1] along axis of size s. If any of the index values are out of bounds,
 * the behavior is undefined.
 *
 * @param[in] yDesc
 * Tensor descriptor for output tensor @c y which is of the same shape as
 * @c indices .
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 *
 * @param[in] axis
 * Which axis to gather on. Negative value means counting dimensions from the
 * back. Accepted range is [-r, r-1] where r = rank( @c x ).
 */
hiednnStatus_t
hiednnCudaGatherElements(hiednnCudaHandle_t cudaHandle,
                         hiednnTensorDesc_t xDesc,
                         const void *x,
                         hiednnTensorDesc_t indicesDesc,
                         const void *indices,
                         hiednnTensorDesc_t yDesc,
                         void *y,
                         int axis);

/**
 * @brief
 * Given a 2-D matrix or batches of 2-D matrices (passed in as a 3-D tensor),
 * returns the upper or lower triangular part of the tensor(s).
 *
 * Trilu takes a @a 3-D input tensor of shape [*, N, M], where * is one (1) or
 * batch dimension. The upper triangular part consists of the elements on and
 * above the given diagonal (k). The lower triangular part consists of elements
 * on and below the diagonal. All other elements in the matrix are set to zero.
 *
 * If k = 0, the triangular part on and above/below the main diagonal is
 * retained. A positive k means the k-th diagonal above the main diagonal, and
 * a negative k means the k-th diagonal below the main diagonal.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc
 * Tensor descriptor for input @a 3-D tensor @c x (the outer-most dim is 1 for
 * one 2-D matrix, or the size of the batch for 2-D matrices).
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] yDesc
 * Tensor descriptor for output tensor @c y which is of the same shape and type
 * as @c x .
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 *
 * @param[in] k
 * An integer corresponding to the number diagonals above or below the main
 * diagonal to exclude or include.
 *
 * @param[in] triluOp
 * Indicates whether upper (#HIEDNN_TRILU_UPPER) or lower (#HIEDNN_TRILU_LOWER)
 * part of matrix is retained.
 */
hiednnStatus_t
hiednnCudaTrilu(hiednnCudaHandle_t cudaHandle,
                hiednnTensorDesc_t xDesc,
                const void *x,
                hiednnTensorDesc_t yDesc,
                void *y,
                int64_t k,
                hiednnTriluOp_t triluOp);

/**
 * @brief
 * Return elements from tensor @c x or @c y depending on boolean elements in
 * tensor @c condition. If elements in @c condition is true, return elements
 * from @c x, otherwise from @c y.
 *
 * This function supports Numpy-style broadcast.
 *
 * This function only support tensor with #HIEDNN_TENSORFORMAT_NORMAL layout.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc, yDesc
 * Tensor descriptors for the input tensor @c x and @c y.
 *
 * @param[in] x, y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of the input tensor @c x and @c y.
 *
 * @param[in] conditionDesc
 * Tensor descriptors for the input tensor @c condition, data type of
 * @c conditionDesc should be #HIEDNN_DATATYPE_BOOL.
 *
 * @param[in] condition
 * Device memory (or device addressable page locked memory) pointer to the data
 * of the input tensor @c condition.
 *
 * @param[in] zDesc
 * Tensor descriptors for the output tensor @c z.
 *
 * @param[out] z
 * Device memory (or device addressable page locked memory) pointer to the data
 * of the output tensor @c z.
 */
hiednnStatus_t
hiednnCudaWhere(hiednnCudaHandle_t cudaHandle,
                hiednnTensorDesc_t xDesc,
                const void *x,
                hiednnTensorDesc_t yDesc,
                const void *y,
                hiednnTensorDesc_t conditionDesc,
                const char *condition,
                hiednnTensorDesc_t zDesc,
                void *z);

/**
 * @brief
 * Creating a copy of the input @c x tensor, and then updating its value to
 * values specified by @c updates tensor at specific index positions specified
 * by @c indices tensor .
 *
 * ScatterElements takes three inputs @c x , @c updates , and @c indices of the
 * same rank r >= 1, and an attribute @c axis that identifies an axis of @c x .
 * Its output shape is the same as the shape of @c x .
 *
 * Another attribute @c reduction allows specification of an optional reduction
 * operation, which is applied to all values in @c updates tensor
 * into output @c y tensor at the specified @c indices .
 * Refer to #hiednnScatterElemReduce_t for more information.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc
 * Tensor descriptor for input tensor @c x whose rank r >= 1.
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] indicesDesc
 * Tensor descriptor for int32/int64 indexing tensor @c indices which is of the
 * same rank r as the input @c x .
 *
 * @param[in] indices
 * Device memory (or device addressable page locked memory) pointer to the data
 * of indexing tensor. All index values are expected to be within bounds
 * [-s, s-1] along @c axis of size s. If any of the index values are out of
 * bounds, the behavior is undefined.
 *
 * @param[in] updatesDesc
 * Tensor descriptor for tensor @c updates which is of the same rank and shape
 * as @c indices .
 *
 * @param[in] updates
 * Device memory (or device addressable page locked memory) pointer to the data
 * of updates tensor.
 *
 * @param[in] yDesc
 * Tensor descriptor for output tensor @c y which is of the same rank and shape
 * as @c x .
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 *
 * @param[in] axis
 * Which axis to scatter on. Negative value means counting dimensions from the
 * back. Accepted range is [-r, r-1] where r = rank( @c x ).
 *
 * @param[in] reduction
 * Type of reduction to apply. Refer to #hiednnScatterElemReduce_t for more
 * information. In cases where @c reduction is set to
 * #HIEDNN_SCATTERELEM_REDUCE_NONE, @c indices should not have duplicate
 * entries, otherwise the behavior is undefined.
 */
hiednnStatus_t
hiednnCudaScatterElements(hiednnCudaHandle_t cudaHandle,
                          hiednnTensorDesc_t xDesc,
                          const void *x,
                          hiednnTensorDesc_t indicesDesc,
                          const void *indices,
                          hiednnTensorDesc_t updatesDesc,
                          const void *updates,
                          hiednnTensorDesc_t yDesc,
                          void *y,
                          int axis,
                          hiednnScatterElemReduce_t reduction);

/**
 * @brief
 * Returns the minimum size in bytes of the workspace that #hiednnCudaNonZero
 * requires.
 *
 * @param[in] xDesc
 * Tensor descriptor for input tensor.
 *
 * @return
 * Minimum size in bytes of the workspace.
 */
size_t
hiednnCudaNonZeroGetWorkspaceSize(hiednnTensorDesc_t xDesc);

/**
 * @brief
 * Returns the indices of the elements that are non-zero (in row-major order).
 * When the number of non-zero elements is large, consider using
 * #hiednnCudaFastNonZero for better performance.
 *
 * Returns a @b flattened tensor, each row of the original @b 2-D tensor
 * corresponding to one dimension of input tensor @c x ,
 * containing the indices of the non-zero elements in that dimension.
 * The number of non-zero elements @a M will be written to the memory that
 * the pointer @c countPtr refers to. A user-allocated workspace is required,
 * and its size should be computed with #hiednnCudaNonZeroGetWorkspaceSize.
 *
 * @note
 * Suppose the input tensor @c x is an @a N -Dim tensor of size @a S, containing
 * @a M non-zero elements. Then the shape of the output tensor @c y should
 * be ( @a N * @a S ,) when passed into this operator to ensure enough space.
 * After this operator returns, @c y is a @b flattened tensor whose original
 * shape is ( @a N , @a M ). The value of an element beyond index @a N * @a M
 * in tensor @c y is undefined.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc
 * Tensor descriptor for input tensor @c x .
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] yDesc
 * Tensor descriptor for output tensor @c y whose data type is uint32 or uint64.
 * To ensure enough space, the shape of @c y is expected to be
 * ( @a N * @a S ,) , where @a N is the rank of @c x and @a S is
 * the size of @c x .
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 *
 * @param[out] nonZeroCount
 * Device memory (or device addressable page locked memory) pointer where the
 * count of non-zero elements will be stored.
 *
 * @param[in] workspace
 * Device memory (or device addressable page locked memory) pointer to the
 * user-allocated workspace.
 *
 * @param[in] wsSizeInBytes
 * Size of the workspace in bytes, no less than the size computed by the
 * function #hiednnCudaNonZeroGetWorkspaceSize.
 */
hiednnStatus_t
hiednnCudaNonZero(hiednnCudaHandle_t cudaHandle,
                  hiednnTensorDesc_t xDesc,
                  const void *x,
                  hiednnTensorDesc_t yDesc,
                  void *y,
                  size_t *nonZeroCount,
                  void *workspace,
                  size_t wsSizeInBytes);

/**
 * @brief
 * Returns the indices of the elements that are non-zero (in row-major order).
 * This is a faster version of #hiednnCudaNonZero operator, especially when the
 * number of non-zero elements is large.
 *
 * Returns a @b tuple of arrays, one for each dimension of input tensor @c x ,
 * containing the indices of the non-zero elements in that dimension.
 * The number of non-zero elements will be written to the memory that the
 * pointer @c countPtr refers to.
 *
 * @note
 * Suppose the input tensor @c x is an @a N -Dim tensor of size @a S, containing
 * @a M non-zero elements. Then the shape of the output tensor @c y should be
 * ( @a N , @a S ) when passed into this operator. After this operator returns,
 * the first @a M elements of each row of @c y are valid, containing the indices
 * of the non-zero elements in that dimension. The value of an element beyond
 * index @a M in each row is undefined.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDesc
 * Tensor descriptor for input tensor @c x .
 *
 * @param[in] x
 * Device memory (or device addressable page locked memory) pointer to the data
 * of input tensor.
 *
 * @param[in] yDesc
 * Tensor descriptor for output tensor @c y whose data type is uint32 or uint64.
 * To ensure enough space, the shape of @c y is expected to be ( @a N , @a S )
 * where @a N is the rank of @c x and @a S is the size of @c x .
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 *
 * @param[out] nonZeroCount
 * Device memory (or device addressable page locked memory) pointer where the
 * count of non-zero elements will be stored, indicating the valid length of
 * each row in the output tensor @c y .
 */
hiednnStatus_t
hiednnCudaFastNonZero(hiednnCudaHandle_t cudaHandle,
                      hiednnTensorDesc_t xDesc,
                      const void *x,
                      hiednnTensorDesc_t yDesc,
                      void *y,
                      size_t *nonZeroCount);

/**
 * @brief
 * Concatenate a series of tensors into a single tensor.
 *
 * @param[in] cudaHandle
 * Previously created HIE-DNN CUDA handle.
 *
 * @param[in] xDescs
 * Array of pointers to the input tensor descriptor. All of the input tensors
 * should have the same shape, except for the dimension @c axis.
 *
 * @param[in] x
 * Array of pointers to the input tensor data. The array should locate in host
 * memory and pointers in the array should point to device memory (or device
 * addressable page locked memory)
 *
 * @param[in] inputCount
 * Number of input tensors, should not exceed 128.
 *
 * @param[in] axis
 * Which axis to concat on. A negative value means counting dimensions from
 * the back. Accepted range is [-r, r-1] where r = rank(inputs)..
 *
 * @param[in] yDesc
 * Tensor descriptor for the output tensor. The output tensor should have same
 * shape with input tensors, except for the dimension @c axis. Size of the
 * dimension @c axis of the output tensor should equal to the sum of that of the
 * input tensors.
 *
 * @param[out] y
 * Device memory (or device addressable page locked memory) pointer to the data
 * of output tensor.
 */
hiednnStatus_t
hiednnCudaConcat(hiednnCudaHandle_t cudaHandle,
                 hiednnTensorDesc_t const *xDescs,
                 const void * const *x,
                 int inputCount,
                 int axis,
                 hiednnTensorDesc_t yDesc,
                 void *y);

#if defined(__cplusplus)
}
#endif

#endif  // HIEDNN_CUDA_H_
