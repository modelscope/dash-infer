/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn.h
 */

/**
 * @file
 * HIE-DNN Basic API.
 */

#ifndef HIEDNN_H_
#define HIEDNN_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @enum hiednnStatus_t
 * Enumerated type for function status returns.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_SUCCESS
 * Function was done successfully.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_INVALID_DATATYPE
 * Data type of tensor is invalid or not supported by the operator.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_INVALID_OPTYPE
 * Operator type or mode (such as #hiednnReduceOp_t or #hiednnInterpCoordMode_t)
 * is invalid or not supported by the operator.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_INVALID_PARAMETER
 * Parameters of functions are invalid or not compatible.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_TENSOR_OVERSIZE
 * Size of tensor exceed operator's limition.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_RUNTIME_ERROR
 * Runtime error (such as CUDA runtime error) occurred in function.
 *
 * @var hiednnStatus_t HIEDNN_STATUS_INTERNAL_ERROR
 * Other exceptions except all above occured in function.
 */
typedef enum  {
    HIEDNN_STATUS_SUCCESS               = 0,
    HIEDNN_STATUS_INVALID_DATATYPE      = 1,
    HIEDNN_STATUS_INVALID_OPTYPE        = 2,
    HIEDNN_STATUS_INVALID_PARAMETER     = 3,
    HIEDNN_STATUS_TENSOR_OVERSIZE       = 4,
    HIEDNN_STATUS_RUNTIME_ERROR         = 5,
    HIEDNN_STATUS_INTERNAL_ERROR        = 6,
} hiednnStatus_t;

/**
 * @typedef hiednnTensorDesc_t
 * A pointer to an opaque structure holding the information
 * (such as dimension, datatype, etc.) of a tensor.
 */
typedef struct HiednnTensorDesc *hiednnTensorDesc_t;

/**
 * @def HIEDNN_DIM_MAX
 * #hiednnTensorDesc_t support at most #HIEDNN_DIM_MAX dimensions.
 */
#define HIEDNN_DIM_MAX 8

/**
 * @enum hiednnDataType_t
 * Enumerated type to indicate the data type of tensor.
 *
 * @var HIEDNN_DATATYPE_FP32
 * 32-bit single-precision floating-point (float).
 *
 * @var HIEDNN_DATATYPE_FP64
 * 64-bit double-precision floating-point (double).
 *
 * @var HIEDNN_DATATYPE_FP16
 * 16-bit half-precision floating-point.
 *
 * @var HIEDNN_DATATYPE_BF16
 * 16-bit floating-point with 1-bit sign, 8-bit exponent and 7-bit mantissa.
 *
 * @var HIEDNN_DATATYPE_INT8
 * 8-bit signed integer.
 *
 * @var HIEDNN_DATATYPE_INT16
 * 16-bit signed integer.
 *
 * @var HIEDNN_DATATYPE_INT32
 * 32-bit signed integer.
 *
 * @var HIEDNN_DATATYPE_INT64
 * 64-bit signed integer.
 *
 * @var HIEDNN_DATATYPE_UINT8
 * 8-bit unsigned integer.
 *
 * @var HIEDNN_DATATYPE_UINT16
 * 16-bit unsigned integer.
 *
 * @var HIEDNN_DATATYPE_UINT32
 * 32-bit unsigned integer.
 *
 * @var HIEDNN_DATATYPE_UINT64
 * 64-bit unsigned integer.
 *
 * @var HIEDNN_DATATYPE_BOOL
 * 8-bit boolean type (can be 0 or 1).
 */
typedef enum {
    HIEDNN_DATATYPE_FP32    = 0,
    HIEDNN_DATATYPE_FP64    = 1,
    HIEDNN_DATATYPE_FP16    = 2,
    HIEDNN_DATATYPE_BF16    = 3,
    HIEDNN_DATATYPE_INT8    = 4,
    HIEDNN_DATATYPE_INT16   = 5,
    HIEDNN_DATATYPE_INT32   = 6,
    HIEDNN_DATATYPE_INT64   = 7,
    HIEDNN_DATATYPE_UINT8   = 8,
    HIEDNN_DATATYPE_UINT16  = 9,
    HIEDNN_DATATYPE_UINT32  = 10,
    HIEDNN_DATATYPE_UINT64  = 11,
    HIEDNN_DATATYPE_BOOL    = 12,
} hiednnDataType_t;

/**
 * @enum hiednnTensorFormat_t
 * Enumerated type to indicate the data layout of a tensor.
 *
 * @var HIEDNN_TENSORFORMAT_NORMAL
 * Tensor is dense and the innermost dimension is contiguous in memory. @n
 * For example, a tensor's dim is @c <tt>{D0, D1, D2}</tt>, the stride will be
 * @c <tt>{D1*D2, D2, 1}</tt>.
 *
 * @var HIEDNN_TENSORFORMAT_NHWC
 * Only work for 4D dense tensor and the data is laid in the following order:
 * batch size, height, width, channels. @n
 * For example, a 4D tensor's dim is @c <tt>{N, C, H, W}</tt>, the stride will
 * be @c <tt>{H*W*C, 1, W*C, C}</tt>.
 *
 * @var HIEDNN_TENSORFORMAT_NcHWc4
 * Only work for 4D dense tensor, the data is laid in the following order:
 * batch size, height, width, channels, and every 4 channels are packed at
 * the innermost dimension. @n
 * A 4D tensor with dim @c <tt>{N, C, H, W}</tt> can be interpreted as a
 * 5D tensor with dim @c <tt>{N, C/4, H, W, 4}</tt>, each @c W is a 4-C
 * packed vector.
 *
 * @var HIEDNN_TENSORFORMAT_NcHWc32
 * Only work for 4D dense tensor, the data is laid in the following order:
 * batch size, height, width, channels, and every 32 channels are packed at
 * the innermost dimension. @n
 * A 4D tensor with dim @c <tt>{N, C, H, W}</tt> can be interpreted as a
 * 5D tensor with dim @c <tt>{N, C/32, H, W, 32}</tt>, each @c W is a 32-C
 * packed vector.
 *
 * @var HIEDNN_TENSORFORMAT_CUSTOMIZED
 * Tensor layout is non of the above and defined by dim-stride.
 */
typedef enum {
    HIEDNN_TENSORFORMAT_NORMAL      = 0,
    HIEDNN_TENSORFORMAT_NHWC        = 1,
    HIEDNN_TENSORFORMAT_NcHWc4      = 2,
    HIEDNN_TENSORFORMAT_NcHWc32     = 3,
    HIEDNN_TENSORFORMAT_CUSTOMIZED  = 4,
} hiednnTensorFormat_t;

/**
 * @enum hiednnSetTensorValueMode_t
 * Enumerated type to indicate the mode of function: @n
 * #hiednnCppSetTensorValue @n
 * #hiednnCudaSetTensorValue
 *
 * @var hiednnSetTensorValueMode_t HIEDNN_SETTENSOR_CONST
 * Set all elements of the tensor to value indicated by parameter @c p0.
 *
 * @var hiednnSetTensorValueMode_t HIEDNN_SETTENSOR_RANGE
 * Only work for 1D tensor, set the element with index @c i to @c p0+p1*i.
 *
 * @var hiednnSetTensorValueMode_t HIEDNN_SETTENSOR_DIAGONAL
 * Only work for 2D tensor, set the element with index @c <tt>(i,i+p0)</tt> to
 * value @c p1, others to zero.@n
 * @c p0 can be negative.
 */
typedef enum {
    HIEDNN_SETTENSOR_CONST      = 0,
    HIEDNN_SETTENSOR_RANGE      = 1,
    HIEDNN_SETTENSOR_DIAGONAL   = 2,
} hiednnSetTensorValueMode_t;

/**
 * @enum hiednnUnaryEltwiseOp_t
 * Enumerated type to indicate the operator of function: @n
 * #hiednnCppUnaryElementwiseOp @n
 * #hiednnCudaUnaryElementwiseOp
 *
 * Support all data types, input and output should have same data type: @n
 * #HIEDNN_UNARY_MATH_ABS @n
 * #HIEDNN_UNARY_MATH_ADD @n
 * #HIEDNN_UNARY_MATH_MUL @n
 * #HIEDNN_UNARY_MATH_DIV @n
 * #HIEDNN_UNARY_MATH_SHRINK @n
 * #HIEDNN_UNARY_MATH_CLIP @n
 * #HIEDNN_UNARY_MATH_SIGN @n
 * @n
 * Support floating point types (FP64/FP32/FP16/BF16...), input and output
 * should have same data type: @n
 * #HIEDNN_UNARY_MATH_POW @n
 * #HIEDNN_UNARY_MATH_SQRT @n
 * #HIEDNN_UNARY_MATH_CBRT @n
 * #HIEDNN_UNARY_MATH_EXP @n
 * #HIEDNN_UNARY_MATH_ERF @n
 * #HIEDNN_UNARY_MATH_LOG @n
 * #HIEDNN_UNARY_MATH_SIN @n
 * #HIEDNN_UNARY_MATH_COS @n
 * #HIEDNN_UNARY_MATH_TAN @n
 * #HIEDNN_UNARY_MATH_ASIN @n
 * #HIEDNN_UNARY_MATH_ACOS @n
 * #HIEDNN_UNARY_MATH_ATAN @n
 * #HIEDNN_UNARY_MATH_SINH @n
 * #HIEDNN_UNARY_MATH_COSH @n
 * #HIEDNN_UNARY_MATH_TANH @n
 * #HIEDNN_UNARY_MATH_ASINH @n
 * #HIEDNN_UNARY_MATH_ACOSH @n
 * #HIEDNN_UNARY_MATH_ATANH @n
 * #HIEDNN_UNARY_MATH_RECIPROCAL @n
 * #HIEDNN_UNARY_MATH_CEIL @n
 * #HIEDNN_UNARY_MATH_FLOOR @n
 * #HIEDNN_UNARY_MATH_ROUND @n
 * #HIEDNN_UNARY_MATH_SIGMOID @n
 * #HIEDNN_UNARY_MATH_LEAKYRELU @n
 * #HIEDNN_UNARY_MATH_ELU @n
 * #HIEDNN_UNARY_MATH_SELU @n
 * #HIEDNN_UNARY_MATH_CELU @n
 * #HIEDNN_UNARY_MATH_SOFTPLUS @n
 * #HIEDNN_UNARY_MATH_SOFTSIGN @n
 * #HIEDNN_UNARY_MATH_HARDSIGMOID @n
 * #HIEDNN_UNARY_MATH_HARDSWISH @n
 * @n
 * Support signed types (INT32, FP32...), input and output should have same
 * data type: @n
 * #HIEDNN_UNARY_MATH_NEG @n
 * #HIEDNN_UNARY_MATH_THRESHOLDRELU @n
 * @n
 * Support floating point input and BOOL output: @n
 * #HIEDNN_UNARY_MATH_ISINF @n
 * #HIEDNN_UNARY_MATH_ISNAN @n
 * @n
 * Both input and output should be BOOL: @n
 * #HIEDNN_UNARY_LOGICAL_NOT
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ABS
 * @c <tt>if x[i] < 0, op(x[i]) = -x[i]</tt>; @n
 * @c <tt>otherwise, op(x[i]) = x[i]</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ADD
 * @c <tt>op(x[i]) = x[i] + extParam1</tt>. @n
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_MUL
 * <tt>op(x[i]) = x[i] * extParam1</tt>. @n
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_DIV
 * <tt>op(x[i]) = x[i] / extParam1</tt>. @n
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SHRINK
 * @c <tt>if x[i] > extParam1, op(x[i]) = x[i] - extParam2</tt>; @n
 * @c <tt>if x[i] < -extParam1, op(x[i]) = x[i] + extParam2</tt>; @n
 * @c <tt>otherwise, op(x[i]) = 0</tt>. @n
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_CLIP
 * @c <tt>if x[i] < extParam1, op(x[i]) = extParam1</tt>; @n
 * @c <tt>if x[i] > extParam2, op(x[i]) = extParam2</tt>; @n
 * @c <tt>otherwise, op(x[i]) = x[i]</tt>. @n
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SIGN
 * @c <tt>if x[i] > 0, op(x[i]) = 1</tt>; @n
 * @c <tt>if x[i] = 0, op(x[i]) = 0</tt>; @n
 * @c <tt>if x[i] < 0, op(x[i]) = -1</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_POW
 * @c <tt>op(x[i]) = pow(x[i], extParam1)</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SQRT
 * @c <tt>op(x[i]) = sqrt(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_CBRT
 * @c <tt>op(x[i]) = cbrt(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_EXP
 * @c <tt>op(x[i]) = exp(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ERF
 * @c <tt>op(x[i]) = erf(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_LOG
 * @c <tt>op(x[i]) = log(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SIN
 * @c <tt>op(x[i]) = sin(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_COS
 * @c <tt>op(x[i]) = cos(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_TAN
 * @c <tt>op(x[i]) = tan(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ASIN
 * @c <tt>op(x[i]) = asin(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ACOS
 * @c <tt>op(x[i]) = acos(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ATAN
 * @c <tt>op(x[i]) = atan(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SINH
 * @c <tt>op(x[i]) = sinh(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_COSH
 * @c <tt>op(x[i]) = cosh(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_TANH
 * @c <tt>op(x[i]) = tanh(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ASINH
 * @c <tt>op(x[i]) = asinh(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ACOSH
 * @c <tt>op(x[i]) = acosh(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ATANH
 * @c <tt>op(x[i]) = atanh(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_RECIPROCAL
 * @c <tt>op(x[i]) = 1 / x[i]</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_CEIL
 * @c <tt>op(x[i]) = ceil(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_FLOOR
 * @c <tt>op(x[i]) = floor(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ROUND
 * @c <tt>op(x[i]) = round(x[i])</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SIGMOID
 * @c <tt>op(x[i]) = 1 / (1 + exp(x[i]))</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_LEAKYRELU
 * @c <tt>if x[i] < 0, op(x[i]) = x[i] * extParam1</tt>; @n
 * @c <tt>otherwise, op(x[i]) = x</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ELU
 * @c <tt>if x[i] < 0, op(x[i]) = extParam1 * (exp(x[i]) - 1)</tt>; @n
 * @c <tt>otherwise, op(x[i]) = x</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SELU
 * @c <tt>if x[i] > 0, op(x[i]) = extParam2 * x[i]</tt>; @n
 * @c <tt>otherwise, op(x[i]) =
 *    extParam2 * (extParam1 * exp(x[i]) - extParam1)</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_CELU
 * @c <tt>op(x[i]) = max(0, x[i]) +
 *                   min(0, extParam1 * (exp(x / extParam1) - 1))</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SOFTPLUS
 * @c <tt>op(x[i]) = log1p(exp(x[i]))</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_SOFTSIGN
 * @c <tt>op(x[i]) = x[i] / (1 + abs(x[i]))</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_HARDSIGMOID
 * @c <tt>op(x[i]) = max(0, min(1, extParam1 * x[i] + extParam2))</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_HARDSWISH
 * @c <tt>op(x[i]) = x[i] * max(0, min(1, extParam1 * x[i] + extParam2))</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_NEG
 * @c <tt>op(x[i]) = -x[i]</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_THRESHOLDRELU
 * @c <tt>if x[i] > extParam1, op(x[i]) = x</tt>; @n
 * @c <tt>otherwise, op(x[i]) = 0</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ISINF
 * @c <tt>if x[i] is INF or -INF, op(x[i]) = 1</tt>; @n
 * @c <tt>otherwise, op(x[i]) = 0</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_MATH_ISNAN
 * @c <tt>if x[i] is NAN or -NAN, op(x[i]) = 1</tt>; @n
 * @c <tt>otherwise, op(x[i]) = 0</tt>.
 *
 * @var hiednnUnaryEltwiseOp_t HIEDNN_UNARY_LOGICAL_NOT
 * @c <tt>op(x[i]) = !x[i]</tt>.
 */
typedef enum {
    // support all datatype
    HIEDNN_UNARY_MATH_ABS           = 0,
    HIEDNN_UNARY_MATH_ADD           = 1,
    HIEDNN_UNARY_MATH_MUL           = 2,
    HIEDNN_UNARY_MATH_DIV           = 3,
    HIEDNN_UNARY_MATH_SHRINK        = 4,
    HIEDNN_UNARY_MATH_CLIP          = 5,
    HIEDNN_UNARY_MATH_SIGN          = 6,

    // support floating point
    HIEDNN_UNARY_MATH_POW           = 7,
    HIEDNN_UNARY_MATH_SQRT          = 8,
    HIEDNN_UNARY_MATH_CBRT          = 9,
    HIEDNN_UNARY_MATH_EXP           = 10,
    HIEDNN_UNARY_MATH_ERF           = 11,
    HIEDNN_UNARY_MATH_LOG           = 12,
    HIEDNN_UNARY_MATH_SIN           = 13,
    HIEDNN_UNARY_MATH_COS           = 14,
    HIEDNN_UNARY_MATH_TAN           = 15,
    HIEDNN_UNARY_MATH_ASIN          = 16,
    HIEDNN_UNARY_MATH_ACOS          = 17,
    HIEDNN_UNARY_MATH_ATAN          = 18,
    HIEDNN_UNARY_MATH_SINH          = 19,
    HIEDNN_UNARY_MATH_COSH          = 20,
    HIEDNN_UNARY_MATH_TANH          = 21,
    HIEDNN_UNARY_MATH_ASINH         = 22,
    HIEDNN_UNARY_MATH_ACOSH         = 23,
    HIEDNN_UNARY_MATH_ATANH         = 24,
    HIEDNN_UNARY_MATH_RECIPROCAL    = 25,
    HIEDNN_UNARY_MATH_CEIL          = 26,
    HIEDNN_UNARY_MATH_FLOOR         = 27,
    HIEDNN_UNARY_MATH_ROUND         = 28,
    HIEDNN_UNARY_MATH_SIGMOID       = 29,
    HIEDNN_UNARY_MATH_LEAKYRELU     = 30,
    HIEDNN_UNARY_MATH_ELU           = 31,
    HIEDNN_UNARY_MATH_SELU          = 32,
    HIEDNN_UNARY_MATH_CELU          = 33,
    HIEDNN_UNARY_MATH_SOFTPLUS      = 34,
    HIEDNN_UNARY_MATH_SOFTSIGN      = 35,
    HIEDNN_UNARY_MATH_HARDSIGMOID   = 36,
    HIEDNN_UNARY_MATH_HARDSWISH     = 37,

    // support signed number
    HIEDNN_UNARY_MATH_NEG           = 38,
    HIEDNN_UNARY_MATH_THRESHOLDRELU = 39,

    // floating point input, bool output
    HIEDNN_UNARY_MATH_ISINF         = 40,
    HIEDNN_UNARY_MATH_ISNAN         = 41,

    // only support bool
    HIEDNN_UNARY_LOGICAL_NOT        = 42,
} hiednnUnaryEltwiseOp_t;

/**
 * @enum hiednnBinaryEltwiseOp_t
 * Enumerated type to indicate the operator of function: @n
 * #hiednnCudaBinaryElementwiseOp
 *
 * Support all data types, input and output should have same data type: @n
 * #HIEDNN_BINARY_MATH_ADD @n
 * #HIEDNN_BINARY_MATH_SUB @n
 * #HIEDNN_BINARY_MATH_MUL @n
 * #HIEDNN_BINARY_MATH_DIV @n
 * #HIEDNN_BINARY_MATH_MAX @n
 * #HIEDNN_BINARY_MATH_MIN @n
 * #HIEDNN_BINARY_MATH_MOD @n
 * #HIEDNN_BINARY_MATH_PRELU @n
 * @n
 * Input support all data types, output is BOOL: @n
 * #HIEDNN_BINARY_COMPARE_EQ @n
 * #HIEDNN_BINARY_COMPARE_GT @n
 * #HIEDNN_BINARY_COMPARE_GE @n
 * #HIEDNN_BINARY_COMPARE_LT @n
 * #HIEDNN_BINARY_COMPARE_LE @n
 * @n
 * Both input and output should be BOOL: @n
 * #HIEDNN_BINARY_LOGICAL_AND @n
 * #HIEDNN_BINARY_LOGICAL_OR @n
 * #HIEDNN_BINARY_LOGICAL_XOR @n
 * @n
 * Both input and output should be unsigned integer: @n
 * #HIEDNN_BINARY_MATH_BITSHIFT
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_ADD
 * @c <tt>op(x[i], y[i]) = x[i] + y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_SUB
 * @c <tt>op(x[i], y[i]) = x[i] - y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_MUL
 * @c <tt>op(x[i], y[i]) = x[i] * y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_DIV
 * @c <tt>op(x[i], y[i]) = x[i] / y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_MAX
 * @c <tt>op(x[i], y[i]) = max(x[i], y[i])</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_MIN
 * @c <tt>op(x[i], y[i]) = min(x[i], y[i])</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_MOD
 * @c extParam is int32. @n
 * @c <tt>op(x[i], y[i]) = x[i] mod y[i]</tt>. @n
 * If @c extParam is 0, the sign of remainder is same as that of dividend, @n
 * and the input and output tensor can only be integer; @n
 * If @c extParam is 1, the sign of remainder is same as that of divisor, @n
 * and the input and output tensor can be any types.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_PRELU
 * @c <tt>if x[i] > 0, op(x[i], y[i]) = x[i]</tt>; @n
 * @c <tt>otherwise, op(x[i], y[i]) = x[i] * y[i]</tt>. @n
 * Tensor @c y should be broadcastable to tensor x.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_COMPARE_EQ
 * @c <tt>op(x[i], y[i]) = x[i] == y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_COMPARE_GT
 * @c <tt>op(x[i], y[i]) = x[i] > y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_COMPARE_GE
 * @c <tt>op(x[i], y[i]) = x[i] >= y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_COMPARE_LT
 * @c <tt>op(x[i], y[i]) = x[i] < y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_COMPARE_LE
 * @c <tt>op(x[i], y[i]) = x[i] <= y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_LOGICAL_AND
 * @c <tt>op(x[i], y[i]) = x[i] && y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_LOGICAL_OR
 * @c <tt>op(x[i], y[i]) = x[i] || y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_LOGICAL_XOR
 * @c <tt>op(x[i], y[i]) = x[i] xor y[i]</tt>.
 *
 * @var hiednnBinaryEltwiseOp_t HIEDNN_BINARY_MATH_BITSHIFT
 * @c extParam is int32 @n
 * @c <tt>if extParam is 0, op(x[i], y[i]) = x[i] << y[i]</tt>; @n
 * @c <tt>if extParam is 1, op(x[i], y[i]) = x[i] >> y[i]</tt>. @n
 */
typedef enum {
    // support all datatype
    HIEDNN_BINARY_MATH_ADD          = 0,
    HIEDNN_BINARY_MATH_SUB          = 1,
    HIEDNN_BINARY_MATH_MUL          = 2,
    HIEDNN_BINARY_MATH_DIV          = 3,
    HIEDNN_BINARY_MATH_MAX          = 4,
    HIEDNN_BINARY_MATH_MIN          = 5,
    HIEDNN_BINARY_MATH_MOD          = 6,
    HIEDNN_BINARY_MATH_PRELU        = 7,

    // all datatype input, bool output
    HIEDNN_BINARY_COMPARE_EQ        = 8,
    HIEDNN_BINARY_COMPARE_GT        = 9,
    HIEDNN_BINARY_COMPARE_GE        = 10,
    HIEDNN_BINARY_COMPARE_LT        = 11,
    HIEDNN_BINARY_COMPARE_LE        = 12,

    // only support bool
    HIEDNN_BINARY_LOGICAL_AND       = 13,
    HIEDNN_BINARY_LOGICAL_OR        = 14,
    HIEDNN_BINARY_LOGICAL_XOR       = 15,

    // only support unsigned number
    HIEDNN_BINARY_MATH_BITSHIFT     = 16,
} hiednnBinaryEltwiseOp_t;

/**
 * @enum hiednnReduceOp_t
 * Enumerated type to indicate the operator of function: @n
 * #hiednnCudaReduce
 *
 * Support all data types: @n
 * #HIEDNN_REDUCE_SUM @n
 * #HIEDNN_REDUCE_PROD @n
 * #HIEDNN_REDUCE_MAX @n
 * #HIEDNN_REDUCE_MIN @n
 * #HIEDNN_REDUCE_SUM_SQUARE @n
 * @n
 * Only support signed types (FP32, INT32, ...): @n
 * #HIEDNN_REDUCE_SUM_ABS @n
 * @n
 * Only support floating point types (FP64, FP32, FP16, ...): @n
 * #HIEDNN_REDUCE_SQRT_SUM_SQUARE @n
 * #HIEDNN_REDUCE_LOG_SUM @n
 * #HIEDNN_REDUCE_LOG_SUM_EXP
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_SUM
 * @c <tt>op(x[*]) = sum(x[*])</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_PROD
 * @c <tt>op(x[*]) = x[0] * x[1] * x[2] ..</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_MAX
 * @c <tt>op(x[*]) = max(x[*])</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_MIN
 * @c <tt>op(x[*]) = min(x[*])</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_SUM_SQUARE
 * @c <tt>op(x[*]) = sum(x[*] ^ 2)</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_SUM_ABS
 * @c <tt>op(x[*]) = sum(abs(x[*]))</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_SQRT_SUM_SQUARE
 * @c <tt>op(x[*]) = sqrt(sum(x[*] ^ 2))</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_LOG_SUM
 * @c <tt>op(x[*]) = log(sum(x[*]))</tt>.
 *
 * @var hiednnReduceOp_t HIEDNN_REDUCE_LOG_SUM_EXP
 * @c <tt>op(x[*]) = log(sum(exp(x[*])))</tt>.
 */
typedef enum {
    // support all types
    HIEDNN_REDUCE_SUM               = 0,
    HIEDNN_REDUCE_PROD              = 1,
    HIEDNN_REDUCE_MAX               = 2,
    HIEDNN_REDUCE_MIN               = 3,
    HIEDNN_REDUCE_SUM_SQUARE        = 4,

    // only support signed types
    HIEDNN_REDUCE_SUM_ABS           = 5,

    // only support floating point types
    HIEDNN_REDUCE_SQRT_SUM_SQUARE   = 6,
    HIEDNN_REDUCE_LOG_SUM           = 7,
    HIEDNN_REDUCE_LOG_SUM_EXP       = 8,
} hiednnReduceOp_t;

/**
 * @enum hiednnInterpCoordMode_t
 * Enumerated type to indicate the coordinate mode of function: @n
 * #hiednnCudaLinearInterpolation @n
 * #hiednnCudaNearestInterpolation @n
 * #hiednnCudaCubicInterpolation
 *
 * @var hiednnInterpCoordMode_t HIEDNN_INTERP_COORD_HALF_PIXEL
 * @c <tt>coordinate(idx[i]) = (idx[i] + 0.5) / scale - 0.5</tt>.
 *
 * @var hiednnInterpCoordMode_t HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL
 * @c <tt>if yDim[i] > 1, coordinate(idx[i]) =
 *                        (idx[i] + 0.5) / scale - 0.5</tt>. @n
 * @c <tt>otherwise, coordinate(idx[i]) = 0</tt>
 *
 * @var hiednnInterpCoordMode_t HIEDNN_INTERP_COORD_ALIGN_CORNER
 * @c <tt>coordinate(idx[i]) = idx[i] * (xDim[i] - 1) / (yDim[i] - 1)</tt>.
 *
 * @var hiednnInterpCoordMode_t HIEDNN_INTERP_COORD_ASYMMETRIC
 * @c <tt>coordinate(idx[i]) = idx[i] / scale[i]</tt>.
 */
typedef enum {
    HIEDNN_INTERP_COORD_HALF_PIXEL          = 0,
    HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL  = 1,
    HIEDNN_INTERP_COORD_ALIGN_CORNER        = 2,
    HIEDNN_INTERP_COORD_ASYMMETRIC          = 3,
} hiednnInterpCoordMode_t;

/**
 * @enum hiednnInterpNearestMode_t
 * Enumerated type to indicate the nearest mode of function: @n
 * #hiednnCudaNearestInterpolation
 *
 * @var hiednnInterpNearestMode_t HIEDNN_INTERP_NEAREST_HALF_DOWN
 * @c <tt>nearest(idx[i]) = ceil(idx[i] - 0.5)</tt>.
 *
 * @var hiednnInterpNearestMode_t HIEDNN_INTERP_NEAREST_HALF_UP
 * @c <tt>nearest(idx[i]) = floor(idx[i] + 0.5)</tt>.
 *
 * @var hiednnInterpNearestMode_t HIEDNN_INTERP_NEAREST_FLOOR
 * @c <tt>nearest(idx[i]) = floor(idx[i])</tt>.
 *
 * @var hiednnInterpNearestMode_t HIEDNN_INTERP_NEAREST_CEIL
 * @c <tt>nearest(idx[i]) = ceil(idx[i])</tt>.
 */
typedef enum {
    HIEDNN_INTERP_NEAREST_HALF_DOWN = 0,
    HIEDNN_INTERP_NEAREST_HALF_UP   = 1,
    HIEDNN_INTERP_NEAREST_FLOOR     = 2,
    HIEDNN_INTERP_NEAREST_CEIL      = 3,
} hiednnInterpNearestMode_t;

/**
 * @enum hiednnPadMode_t
 * Enumerated type to indicate the padding mode of function: @n
 * #hiednnCudaPad
 *
 * @var hiednnPadMode_t HIEDNN_PAD_CONST
 * Pad with a constant value specified by the parameter @c param of
 * the pad function.
 *
 * @var hiednnPadMode_t HIEDNN_PAD_EDGE
 * Pad with the element on the edge of each axis.
 *
 * @var hiednnPadMode_t HIEDNN_PAD_REFLECT
 * Pad with the mirrored element along each axis.
 */
typedef enum {
    HIEDNN_PAD_CONST    = 0,
    HIEDNN_PAD_EDGE     = 1,
    HIEDNN_PAD_REFLECT  = 2,
} hiednnPadMode_t;

/**
 * @enum hiednnTriluOp_t
 * Enumerated type to indicate the type of function: @n
 * #hiednnCudaTrilu
 *
 * @var hiednnTriluOp_t HIEDNN_TRILU_UPPER
 * The upper part of matrix is retained.
 *
 * @var hiednnTriluOp_t HIEDNN_TRILU_LOWER
 * The lower part of matrix is retained.
 */
typedef enum {
    HIEDNN_TRILU_UPPER  = 0,
    HIEDNN_TRILU_LOWER  = 1,
} hiednnTriluOp_t;

/**
 * @enum hiednnScatterElemReduce_t
 * Enumerated type to indicate the reduction type of function: @n
 * #hiednnCudaScatterElements
 *
 * Note that in the following definitions, the convertion of an index
 * specified in @c indices into the corresponding index in @c y is
 * omitted and simply represented with the star (*) symbol.
 *
 * @var hiednnScatterElemReduce_t HIEDNN_SCATTERELEM_REDUCE_NONE
 * @c <tt>y[indices[*]] = updates[*]</tt>.
 *
 * @var hiednnScatterElemReduce_t HIEDNN_SCATTERELEM_REDUCE_ADD
 * @c <tt>y[indices[*]] = (y[indices[*]] + updates[*])</tt>.
 *
 * @var hiednnScatterElemReduce_t HIEDNN_SCATTERELEM_REDUCE_MUL
 * @c <tt>y[indices[*]] = (y[indices[*]] * updates[*])</tt>.
 *
 * @var hiednnScatterElemReduce_t HIEDNN_SCATTERELEM_REDUCE_MAX
 * @c <tt>y[indices[*]] = max(y[indices[*]], updates[*])</tt>.
 *
 * @var hiednnScatterElemReduce_t HIEDNN_SCATTERELEM_REDUCE_MIN
 * @c <tt>y[indices[*]] = min(y[indices[*]], updates[*])</tt>.
 */
typedef enum {
    HIEDNN_SCATTERELEM_REDUCE_NONE  = 0,
    HIEDNN_SCATTERELEM_REDUCE_ADD   = 1,
    HIEDNN_SCATTERELEM_REDUCE_MUL   = 2,
    HIEDNN_SCATTERELEM_REDUCE_MAX   = 3,
    HIEDNN_SCATTERELEM_REDUCE_MIN   = 4,
} hiednnScatterElemReduce_t;

/**
 * @brief
 * Create an opaque structure to hold the information of a tensor.
 *
 * @param[in,out] tensorDesc
 * Pointer to pointer to the tensor information opaque structure to be created.
 */
hiednnStatus_t
hiednnCreateTensorDesc(hiednnTensorDesc_t *tensorDesc);

/**
 * @brief
 * Destroy a tensor descriptor.
 *
 * @param[in] tensorDesc
 * Pointer to the tensor information opaque structure to be destroyed.
 */
hiednnStatus_t
hiednnDestroyTensorDesc(hiednnTensorDesc_t tensorDesc);

/**
 * @brief
 * Generic function to initialize a previously created tensor descriptor.
 *
 * @param[in,out] tensorDesc
 * Previously created tensor descriptor to be initialized.
 *
 * @param[in] dataType
 * Data type.
 *
 * @param[in] nDims
 * Number of dimensions of the tensor.
 *
 * @param[in] dim
 * Array of @c nDims elements that contain the size of the tensor for every
 * dimension.
 *
 * @param[in] stride
 * Array of @c nDims elements that contain the stride of the tensor for every
 * dimension.
 */
hiednnStatus_t
hiednnSetTensorDesc(hiednnTensorDesc_t tensorDesc,
                    hiednnDataType_t dataType,
                    int nDims,
                    const int64_t *dim,
                    const int64_t *stride);

/**
 * @brief
 * Initialize a tensor descriptor for a tensor with #HIEDNN_TENSORFORMAT_NORMAL
 * data layout.
 *
 * @param[in,out] tensorDesc
 * Previously created tensor descriptor to be initialized.
 *
 * @param[in] dataType
 * Data type.
 *
 * @param[in] nDims
 * Number of dimensions of the tensor.
 *
 * @param[in] dim
 * Array of @c nDims elements that contain the size of the tensor for every
 * dimension.
 */
hiednnStatus_t
hiednnSetNormalTensorDesc(hiednnTensorDesc_t tensorDesc,
                          hiednnDataType_t dataType,
                          int nDims,
                          const int64_t *dim);

/**
 * @brief
 * Initialize a tensor descriptor for a 4D dense tensor.
 *
 * @param[in,out] tensorDesc
 * Previously created tensor descriptor to be initialized.
 *
 * @param[in] dataType
 * Data type.
 *
 * @param[in] tensorFormat
 * Data layout.
 *
 * @param[in] n, c, h, w
 * Size of batch size, channel, height, width.
 */
hiednnStatus_t
hiednnSet4dTensorDesc(hiednnTensorDesc_t tensorDesc,
                      hiednnDataType_t dataType,
                      hiednnTensorFormat_t tensorFormat,
                      int64_t n,
                      int64_t c,
                      int64_t h,
                      int64_t w);

/**
 * @brief
 * Get the parameters of a previously initialized 4D dense tensor descriptor.
 *
 * @param[in] tensorDesc
 * Previously initialized 4D dense tensor descriptor.
 *
 * @param[out] dataType
 * Data type.
 *
 * @param[out] tensorFormat
 * Data layout.
 *
 * @param[out] n, c, h, w
 * Size of batch size, channel, height, width.
 */
hiednnStatus_t
hiednnGet4dTensorDesc(const hiednnTensorDesc_t tensorDesc,
                      hiednnDataType_t *dataType,
                      hiednnTensorFormat_t *tensorFormat,
                      int64_t *n,
                      int64_t *c,
                      int64_t *h,
                      int64_t *w);

/**
 * @brief
 * Get the parameters of a previously initialized tensor descriptor.
 *
 * @b ATTENTION: For a tensor descriptor with channel-packed layout
 * (such as #HIEDNN_TENSORFORMAT_NcHWc4), the returned stride is senseless
 * and set to all-zero.
 *
 * @param[in] tensorDesc
 * Previously initialized tensor descriptor.
 *
 * @param[out] dataType
 * Data type.
 *
 * @param[out] tensorFormat
 * Data layout.
 *
 * @param[in] requestDims
 * Number of dimensions to extract from the tensor descriptor. If
 * @c requestDims is greater than @c <tt>nDims[0]</tt>, only
 * @c <tt>nDims[0]</tt> dimensions are returned.
 *
 * @param[out] nDims
 * Number of returned dimensions.
 *
 * @param[out] dim
 * Array to hold the size of each returned dimension of the tensor descriptor,
 * the size of array should be at least @c requestDims.
 *
 * @param[out] stride
 * Array to hold the stride of each returned dimension of the tensor descriptor,
 * the size of array should be at least @c requestDims.
 */
hiednnStatus_t
hiednnGetTensorDesc(const hiednnTensorDesc_t tensorDesc,
                    hiednnDataType_t *dataType,
                    hiednnTensorFormat_t *tensorFormat,
                    int requestDims,
                    int *nDims,
                    int64_t *dim,
                    int64_t *stride);

#if defined(__cplusplus)
}
#endif

#endif  // HIEDNN_H_


