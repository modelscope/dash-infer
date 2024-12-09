/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce_index.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <cuda/cuda_handle.hpp>

#include "row_reduce.cuh"
#include "column_reduce.cuh"

namespace hiednn {

namespace cuda {

namespace reduce {

template <typename T>
struct ReduceIndexImpl {
    template <typename IdxT>
    struct Impl {
        template <typename PreFuncT,
                  typename ReduceFuncT,
                  typename PostFuncT>
        hiednnStatus_t operator()(
                const HiednnCudaHandle &handle,
                const int64_t &outerDim,
                const int64_t &reduceDim,
                const int64_t &innerDim,
                const T *x,
                void *indices,
                const PreFuncT &preFunc,
                const ReduceFuncT &reduceFunc,
                const PostFuncT &postFunc) {
            // type of offset, associated with max supported tensor size
            using OffsetT = uint32_t;

            IdxT *idxPtr = static_cast<IdxT *>(indices);
            T alpha = 1;

            if (innerDim == 1) {
                return RowReduce<
                        PreFuncT, ReduceFuncT, PostFuncT, T, T, T,
                        IndexReduceT<T, IdxT>, OffsetT, IdxT, false, true>(
                    handle, outerDim, reduceDim, alpha, x, nullptr, idxPtr);
            } else {
                return ColumnReduce<
                        PreFuncT, ReduceFuncT, PostFuncT, T, T, T,
                        IndexReduceT<T, IdxT>, OffsetT, IdxT, false, true>(
                    handle, reduceDim, innerDim, outerDim, alpha, x,
                    nullptr, idxPtr);
            }
        }
    };

    template <typename GetPreFunc,
              typename GetReduceFunc,
              typename GetPostFunc>
    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            const int64_t &outerDim,
            const int64_t &reduceDim,
            const int64_t &innerDim,
            const void *x,
            hiednnDataType_t indicesType,
            void *indices,
            GetPreFunc getPreFunc,
            GetReduceFunc getReduceFunc,
            GetPostFunc getPostFunc) {
        const T *xPtr = static_cast<const T *>(x);

        auto preFunc = getPreFunc.template get<T>();
        auto reduceFunc = getReduceFunc.template get<T>();
        auto postFunc = getPostFunc.template get<T>();

        return DispatchUnsigned<Impl>(
            indicesType, handle, outerDim, reduceDim,
            innerDim, xPtr, indices, preFunc, reduceFunc, postFunc);
    }
};

hiednnStatus_t ReduceIndex(
        const HiednnCudaHandle &handle,
        hiednnReduceOp_t reduceOp,
        const HiednnTensorDesc &xDesc,
        const void *x,
        int axis,
        const HiednnTensorDesc &indicesDesc,
        void *indices) {
    int64_t d0, d1, d2;
    ReduceDimsFold(xDesc, axis, &d0, &d1, &d2);

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (reduceOp) {
        case HIEDNN_REDUCE_MAX:
            ret = DispatchAll<ReduceIndexImpl>(
                xDesc.dataType, handle, d0, d1, d2, x,
                indicesDesc.dataType, indices,
                GetScalarOp<scalar_functor::Pass>(),
                GetScalarOp<MaxWithIndex>(),
                GetScalarOp<scalar_functor::Pass>());
            break;
        case HIEDNN_REDUCE_MIN:
            ret = DispatchAll<ReduceIndexImpl>(
                xDesc.dataType, handle, d0, d1, d2, x,
                indicesDesc.dataType, indices,
                GetScalarOp<scalar_functor::Pass>(),
                GetScalarOp<MinWithIndex>(),
                GetScalarOp<scalar_functor::Pass>());
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
hiednnCudaReduceIndex(HiednnCudaHandle *cudaHandle,
                      hiednnReduceOp_t reduceOp,
                      HiednnTensorDesc *xDesc,
                      const void *x,
                      int axis,
                      HiednnTensorDesc *indicesDesc,
                      void *indices) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, indicesDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *indicesDesc, indices)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *indicesDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->nDims != indicesDesc->nDims ||
        axis < 0 || axis >= xDesc->nDims ||
        indicesDesc->dims[axis] != 1) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    for (int i = 0; i < xDesc->nDims; ++i) {
        if (i != axis && indicesDesc->dims[i] != xDesc->dims[i]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }

    if (indicesDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::cuda::reduce::ReduceIndex(
        *cudaHandle, reduceOp, *xDesc, x, axis,
        *indicesDesc, indices);
}


