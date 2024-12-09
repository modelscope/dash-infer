/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <scalar_functor.hpp>
#include <cuda/cuda_handle.hpp>

#include "reduce_dispatch.hpp"
#include "row_reduce.cuh"
#include "column_reduce.cuh"

namespace hiednn {

namespace cuda {

namespace reduce {

template <template <typename> class PreFunc,
          template <typename> class ReduceFunc,
          template <typename> class PostFunc>
struct ReduceImpl {
    template <typename ST,
              typename DT,
              typename CompT>
    hiednnStatus_t Run(
            const HiednnCudaHandle &handle,
            const int64_t &outerDim,
            const int64_t &reduceDim,
            const int64_t &innerDim,
            const void *alphaPtr,
            const void *x,
            void *y) {
        DT alpha = *static_cast<const DT *>(alphaPtr);
        const ST *xPtr = static_cast<const ST *>(x);
        DT *yPtr = static_cast<DT *>(y);

        using PreFuncT = PreFunc<CompT>;
        using ReduceFuncT = ReduceFunc<CompT>;
        using PostFuncT = PostFunc<CompT>;

        // type of offset, associated with max supported tensor size
        using OffsetT = uint32_t;

        if (innerDim == 1) {
            return RowReduce<PreFuncT, ReduceFuncT, PostFuncT,
                             ST, DT, CompT, CompT, OffsetT>(
                handle, outerDim, reduceDim, alpha, xPtr, yPtr);
        } else {
            return ColumnReduce<PreFuncT, ReduceFuncT, PostFuncT,
                                ST, DT, CompT, CompT, OffsetT>(
                handle, reduceDim, innerDim, outerDim, alpha, xPtr, yPtr);
        }
    }
};

// Max/Min reduce functor
template <typename T>
struct CmpReduceImpl {
    template <typename PreFuncT,
              typename ReduceFuncT,
              typename PostFuncT>
    hiednnStatus_t Reduce(
            const HiednnCudaHandle &handle,
            const int64_t &outerDim,
            const int64_t &reduceDim,
            const int64_t &innerDim,
            const T &alpha,
            const T *x,
            T *y,
            const PreFuncT &preFunc,
            const ReduceFuncT &reduceFunc,
            const PostFuncT &postFunc) {
        // type of offset, associated with max supported tensor size
        using OffsetT = uint32_t;

        if (innerDim == 1) {
            return RowReduce<PreFuncT, ReduceFuncT, PostFuncT,
                             T, T, T, T, OffsetT>(
                handle, outerDim, reduceDim, alpha, x, y);
        } else {
            return ColumnReduce<PreFuncT, ReduceFuncT, PostFuncT,
                                T, T, T, T, OffsetT>(
                handle, reduceDim, innerDim, outerDim, alpha, x, y);
        }
    }

    template <typename GetPreFunc,
              typename GetReduceFunc,
              typename GetPostFunc>
    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            const int64_t &outerDim,
            const int64_t &reduceDim,
            const int64_t &innerDim,
            const void *alphaPtr,
            const void *x,
            void *y,
            GetPreFunc getPreFunc,
            GetReduceFunc getReduceFunc,
            GetPostFunc getPostFunc) {
        T alpha = *static_cast<const T *>(alphaPtr);
        const T *xPtr = static_cast<const T *>(x);
        T *yPtr = static_cast<T *>(y);

        auto preFunc = getPreFunc.template get<T>();
        auto reduceFunc = getReduceFunc.template get<T>();
        auto postFunc = getPostFunc.template get<T>();

        return Reduce(handle, outerDim, reduceDim, innerDim, alpha,
                      xPtr, yPtr, preFunc, reduceFunc, postFunc);
    }
};

// Max/Min with index reduce functor
template <typename T>
struct CmpReduceIndexImpl {
    template <typename IdxT,
              typename PreFuncT,
              typename ReduceFuncT,
              typename PostFuncT>
    hiednnStatus_t Reduce(
            const HiednnCudaHandle &handle,
            const int64_t &outerDim,
            const int64_t &reduceDim,
            const int64_t &innerDim,
            const T &alpha,
            const T *x,
            T *y,
            void *indices,
            const PreFuncT &preFunc,
            const ReduceFuncT &reduceFunc,
            const PostFuncT &postFunc) {
        // type of offset, associated with max supported tensor size
        using OffsetT = uint32_t;

        IdxT *idxPtr = static_cast<IdxT *>(indices);

        if (innerDim == 1) {
            return RowReduce<
                    PreFuncT, ReduceFuncT, PostFuncT, T, T, T,
                    IndexReduceT<T, IdxT>, OffsetT, IdxT, true, true>(
                handle, outerDim, reduceDim, alpha, x, y, idxPtr);
        } else {
            return ColumnReduce<
                    PreFuncT, ReduceFuncT, PostFuncT, T, T, T,
                    IndexReduceT<T, IdxT>, OffsetT, IdxT, true, true>(
                handle, reduceDim, innerDim, outerDim, alpha, x, y, idxPtr);
        }
    }

    template <typename GetPreFunc,
              typename GetReduceFunc,
              typename GetPostFunc>
    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            const int64_t &outerDim,
            const int64_t &reduceDim,
            const int64_t &innerDim,
            const void *alphaPtr,
            const void *x,
            void *y,
            hiednnDataType_t indicesType,
            void *indices,
            GetPreFunc getPreFunc,
            GetReduceFunc getReduceFunc,
            GetPostFunc getPostFunc) {
        if (indices == nullptr) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        T alpha = *static_cast<const T *>(alphaPtr);
        const T *xPtr = static_cast<const T *>(x);
        T *yPtr = static_cast<T *>(y);

        auto preFunc = getPreFunc.template get<T>();
        auto reduceFunc = getReduceFunc.template get<T>();
        auto postFunc = getPostFunc.template get<T>();

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;

        switch (indicesType) {
            case HIEDNN_DATATYPE_UINT8:
                ret = Reduce<uint8_t>(
                    handle, outerDim, reduceDim, innerDim, alpha,
                    xPtr, yPtr, indices, preFunc, reduceFunc, postFunc);
                break;
            case HIEDNN_DATATYPE_UINT16:
                ret = Reduce<uint16_t>(
                    handle, outerDim, reduceDim, innerDim, alpha,
                    xPtr, yPtr, indices, preFunc, reduceFunc, postFunc);
                break;
            case HIEDNN_DATATYPE_UINT32:
                ret = Reduce<uint32_t>(
                    handle, outerDim, reduceDim, innerDim, alpha,
                    xPtr, yPtr, indices, preFunc, reduceFunc, postFunc);
                break;
            case HIEDNN_DATATYPE_UINT64:
                ret = Reduce<uint64_t>(
                    handle, outerDim, reduceDim, innerDim, alpha,
                    xPtr, yPtr, indices, preFunc, reduceFunc, postFunc);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
                break;
        }

        return ret;
    }
};

hiednnStatus_t Reduce(
        const HiednnCudaHandle &handle,
        hiednnReduceOp_t reduceOp,
        const void *alpha,
        const HiednnTensorDesc &xDesc,
        const void *x,
        int axis,
        const HiednnTensorDesc &yDesc,
        void *y,
        hiednnDataType_t indicesType,
        void *indices) {
    int64_t d0, d1, d2;
    ReduceDimsFold(xDesc, axis, &d0, &d1, &d2);

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (reduceOp) {
        case HIEDNN_REDUCE_SUM:
            ret = ReduceDispatch<ReduceImpl<
                      scalar_functor::Pass,
                      scalar_functor::Add,
                      scalar_functor::Pass>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_PROD:
            ret = ReduceDispatch<ReduceImpl<
                      scalar_functor::Pass,
                      scalar_functor::Mul,
                      scalar_functor::Pass>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_SUM_ABS:
            ret = ReduceDispatchSigned<ReduceImpl<
                      scalar_functor::Abs,
                      scalar_functor::Add,
                      scalar_functor::Pass>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_SUM_SQUARE:
            ret = ReduceDispatch<ReduceImpl<
                      scalar_functor::Square,
                      scalar_functor::Add,
                      scalar_functor::Pass>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_SQRT_SUM_SQUARE:
            ret = ReduceDispatchFP<ReduceImpl<
                      scalar_functor::Square,
                      scalar_functor::Add,
                      scalar_functor::Sqrt>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_LOG_SUM:
            ret = ReduceDispatchFP<ReduceImpl<
                      scalar_functor::Pass,
                      scalar_functor::Add,
                      scalar_functor::Log>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_LOG_SUM_EXP:
            ret = ReduceDispatchFP<ReduceImpl<
                      scalar_functor::Exp,
                      scalar_functor::Add,
                      scalar_functor::Log>>(
                      xDesc.dataType, yDesc.dataType,
                      handle, d0, d1, d2, alpha, x, y);
            break;
        case HIEDNN_REDUCE_MAX:
            if (indices) {
                ret = DispatchAll<CmpReduceIndexImpl>(
                          xDesc.dataType, handle, d0, d1, d2, alpha,
                          x, y, indicesType, indices,
                          GetScalarOp<scalar_functor::Pass>(),
                          GetScalarOp<MaxWithIndex>(),
                          GetScalarOp<scalar_functor::Pass>());
            } else {
                ret = DispatchAll<CmpReduceImpl>(
                          xDesc.dataType, handle, d0, d1, d2, alpha, x, y,
                          GetScalarOp<scalar_functor::Pass>(),
                          GetScalarOp<scalar_functor::Max>(),
                          GetScalarOp<scalar_functor::Pass>());
            }
            break;
        case HIEDNN_REDUCE_MIN:
            if (indices) {
                ret = DispatchAll<CmpReduceIndexImpl>(
                          xDesc.dataType, handle, d0, d1, d2, alpha,
                          x, y, indicesType, indices,
                          GetScalarOp<scalar_functor::Pass>(),
                          GetScalarOp<MinWithIndex>(),
                          GetScalarOp<scalar_functor::Pass>());
            } else {
                ret = DispatchAll<CmpReduceImpl>(
                          xDesc.dataType, handle, d0, d1, d2, alpha, x, y,
                          GetScalarOp<scalar_functor::Pass>(),
                          GetScalarOp<scalar_functor::Min>(),
                          GetScalarOp<scalar_functor::Pass>());
            }
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_OPTYPE;
            break;
    }

    return ret;
}

}  // namespace reduce

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaReduce(HiednnCudaHandle *cudaHandle,
                 hiednnReduceOp_t reduceOp,
                 const void *alpha,
                 HiednnTensorDesc *xDesc,
                 const void *x,
                 int axis,
                 HiednnTensorDesc *yDesc,
                 void *y,
                 hiednnDataType_t indicesType,
                 void *indices) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->nDims != yDesc->nDims ||
        axis < 0 || axis >= xDesc->nDims ||
        yDesc->dims[axis] != 1 ||
        alpha == nullptr) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    for (int i = 0; i < xDesc->nDims; ++i) {
        if (i != axis && yDesc->dims[i] != xDesc->dims[i]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::cuda::reduce::Reduce(
        *cudaHandle, reduceOp, alpha, *xDesc, x, axis,
        *yDesc, y, indicesType, indices);
}


