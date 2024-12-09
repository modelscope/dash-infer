/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce_utils.cuh
 */

#ifndef DNN_CUDA_REDUCE_REDUCE_UTILS_CUH_
#define DNN_CUDA_REDUCE_REDUCE_UTILS_CUH_

#include <limits>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <scalar_functor.hpp>
#include <cmath_wrapper.hpp>
#include <datatype_extension/datatype_extension.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace reduce {

inline void ReduceDimsFold(
        const HiednnTensorDesc &desc, int axis,
        int64_t *outerDim, int64_t *reduceDim, int64_t *innerDim) {
    int64_t d0 = 1;
    int64_t d1 = 1;

    for (int i = 0; i < axis; ++i) {
        d0 *= desc.dims[i];
    }
    for (int i = axis + 1; i < desc.nDims; ++i) {
        d1 *= desc.dims[i];
    }

    *outerDim = d0;
    *reduceDim = desc.dims[axis];
    *innerDim = d1;
}

//-------------------------------------------------------------
// ReduceT for reduction with indices
//-------------------------------------------------------------
template <typename DT, typename IdxT>
struct alignas(2 * (sizeof(DT) > sizeof(IdxT) ? sizeof(DT) : sizeof(IdxT)))
IndexReduceT {
    DT data;
    IdxT idx;

    __device__ __forceinline__
    IndexReduceT() {}

    __device__ __forceinline__
    explicit IndexReduceT(DT x) : data(x) {}

    __device__ __forceinline__
    IndexReduceT(DT x, IdxT y) : data(x), idx(y) {}
};

//-------------------------------------------------------------
// ReduceFunc for reduction with indices
//-------------------------------------------------------------
template <typename DT>
struct MaxWithIndex {
    template <typename ReduceT>
    __device__ __forceinline__
    ReduceT operator()(const ReduceT &x, const ReduceT &y) {
        return x.data > y.data ? x : y;
    }
};

template <typename DT>
struct MinWithIndex {
    template <typename ReduceT>
    __device__ __forceinline__
    ReduceT operator()(const ReduceT &x, const ReduceT &y) {
        return x.data < y.data ? x : y;
    }
};

//-------------------------------------------------------------
// value for reduction accumulator initialization.
//-------------------------------------------------------------
template <typename ReduceFunc, typename T>
struct ReduceFuncInit;

template <typename T>
struct ReduceFuncInit<scalar_functor::Add<T>, T> {
    static constexpr T v = 0;
};

template <typename T>
struct ReduceFuncInit<scalar_functor::Mul<T>, T> {
    static constexpr T v = 1;
};

// Max functor initialization for integer
template <typename T>
struct ReduceFuncInit<scalar_functor::Max<T>, T> {
    static constexpr T v = std::numeric_limits<T>::has_infinity ?
                           -std::numeric_limits<T>::infinity() :
                           std::numeric_limits<T>::min();
};

template <typename T>
struct ReduceFuncInit<MaxWithIndex<T>, T> {
    static constexpr T v = std::numeric_limits<T>::has_infinity ?
                           -std::numeric_limits<T>::infinity() :
                           std::numeric_limits<T>::min();
};

#ifdef HIEDNN_USE_FP16
template <>
struct ReduceFuncInit<scalar_functor::Max<half>, half> {
    static constexpr float v = -std::numeric_limits<float>::infinity();
};

template <>
struct ReduceFuncInit<MaxWithIndex<half>, half> {
    static constexpr float v = -std::numeric_limits<float>::infinity();
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct ReduceFuncInit<scalar_functor::Max<bfloat16>, bfloat16> {
    static constexpr float v = -std::numeric_limits<float>::infinity();
};

template <>
struct ReduceFuncInit<MaxWithIndex<bfloat16>, bfloat16> {
    static constexpr float v = -std::numeric_limits<float>::infinity();
};
#endif

// Min functor initialization for integer
template <typename T>
struct ReduceFuncInit<scalar_functor::Min<T>, T> {
    static constexpr T v = std::numeric_limits<T>::has_infinity ?
                           std::numeric_limits<T>::infinity() :
                           std::numeric_limits<T>::max();
};

template <typename T>
struct ReduceFuncInit<MinWithIndex<T>, T> {
    static constexpr T v = std::numeric_limits<T>::has_infinity ?
                           std::numeric_limits<T>::infinity() :
                           std::numeric_limits<T>::max();
};

#ifdef HIEDNN_USE_FP16
template <>
struct ReduceFuncInit<scalar_functor::Min<half>, half> {
    static constexpr float v = std::numeric_limits<float>::infinity();
};

template <>
struct ReduceFuncInit<MinWithIndex<half>, half> {
    static constexpr float v = std::numeric_limits<float>::infinity();
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct ReduceFuncInit<scalar_functor::Min<bfloat16>, bfloat16> {
    static constexpr float v = std::numeric_limits<float>::infinity();
};

template <>
struct ReduceFuncInit<MinWithIndex<bfloat16>, bfloat16> {
    static constexpr float v = std::numeric_limits<float>::infinity();
};
#endif

//-------------------------------------------------------------
// make data-indix pairs if @WITH_IDX of kernel template is true
//-------------------------------------------------------------
template <bool WITH_IDX, typename ReduceT>
struct WithIndex;

template <typename ReduceT>
struct WithIndex<true, ReduceT> {
    template <typename DT, typename IdxT>
    __device__ __forceinline__
    ReduceT operator()(const DT &x, const IdxT &idx) {
        return ReduceT(x, idx);
    }
};

template <typename ReduceT>
struct WithIndex <false, ReduceT> {
    template <typename DT, typename IdxT>
    __device__ __forceinline__
    ReduceT operator()(const DT &x, const IdxT &idx) {
        return static_cast<ReduceT>(x);
    }
};

//-------------------------------------------------------------
// data loading functions
//-------------------------------------------------------------
template <int UNROLL, typename OffsetT, typename ST>
__device__ __forceinline__
void LoadData(ST (&ldgReg)[UNROLL],
              const ST *ldgPtr,
              const OffsetT &ldgStride) {
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        Ldg<NC>(&ldgReg[i], ldgPtr + ldgStride * i);
    }
}

template <int IDX_STRIDE, int UNROLL, typename OffsetT, typename ST>
__device__ __forceinline__
void LoadData(ST (&ldgReg)[UNROLL],
              const ST *ldgPtr,
              const OffsetT &idxStart,
              const OffsetT &ldgStride,
              const OffsetT &length) {
    OffsetT ldgCount = length > idxStart ?
                       UIntDivRU<OffsetT>(length - idxStart, IDX_STRIDE) : 0;
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        if (i < ldgCount) {
            Ldg<NC>(&ldgReg[i], ldgPtr + ldgStride * i);
        }
    }
}

//-------------------------------------------------------------
// Load data from GMEM and reduce serially inside thread.
//-------------------------------------------------------------
template <bool MULTI_BLOCK,
          int UNROLL,
          int IDX_STRIDE,
          typename PreFunc,
          typename ReduceFunc,
          typename ReduceInitFunc,
          typename MathT,
          typename ReduceT,
          typename OffsetT,
          typename IdxT,
          bool WITH_IDX>
struct ThreadReducer;

template <int UNROLL,
          int IDX_STRIDE,
          typename PreFunc,
          typename ReduceFunc,
          typename ReduceInitFunc,
          typename MathT,
          typename ReduceT,
          typename OffsetT,
          typename IdxT,
          bool WITH_IDX>
struct ThreadReducer<true, UNROLL, IDX_STRIDE, PreFunc, ReduceFunc,
                     ReduceInitFunc, MathT, ReduceT, OffsetT, IdxT, WITH_IDX> {
    template <typename ST>
    static __device__ __forceinline__
    ReduceT Reduce(const ST *ldgPtr,
                   const OffsetT &idxStart,
                   const OffsetT &ldgStride,
                   const OffsetT &length,
                   const bool &fullTile) {
        ST ldgReg[UNROLL];
        MathT reduceReg[UNROLL];

        if (fullTile) {
            LoadData(ldgReg, ldgPtr, ldgStride);
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                reduceReg[i] = PreFunc()(static_cast<MathT>(ldgReg[i]));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                reduceReg[i] = ReduceInitFunc::v;
            }

            LoadData<IDX_STRIDE>(ldgReg, ldgPtr, idxStart, ldgStride, length);

            OffsetT ldgCount = length > idxStart ?
                UIntDivRU<OffsetT>(length - idxStart, IDX_STRIDE) : 0;
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                if (i < ldgCount) {
                    reduceReg[i] = PreFunc()(static_cast<MathT>(ldgReg[i]));
                }
            }
        }

        WithIndex<WITH_IDX, ReduceT> WithIndexFunc;

        ReduceT ret = WithIndexFunc(reduceReg[0], idxStart);
        #pragma unroll
        for (int i = 1; i < UNROLL; ++i) {
            ret = ReduceFunc()(
                  WithIndexFunc(reduceReg[i], idxStart + i * IDX_STRIDE), ret);
        }
        return ret;
    }
};

template <int UNROLL,
          int IDX_STRIDE,
          typename PreFunc,
          typename ReduceFunc,
          typename ReduceInitFunc,
          typename MathT,
          typename ReduceT,
          typename OffsetT,
          typename IdxT,
          bool WITH_IDX>
struct ThreadReducer<false, UNROLL, IDX_STRIDE, PreFunc, ReduceFunc,
                     ReduceInitFunc, MathT, ReduceT, OffsetT, IdxT, WITH_IDX> {
    template <typename ST>
    static __device__ __forceinline__
    ReduceT Reduce(const ST *ldgPtr,
                   const OffsetT &idxStart,
                   const OffsetT &ldgStride,
                   const OffsetT &length,
                   const bool &fulltile) {
        ST ldgReg[UNROLL];
        MathT reduceReg[UNROLL];

        WithIndex<WITH_IDX, ReduceT> WithIndexFunc;
        ReduceT ret = WithIndexFunc(MathT(ReduceInitFunc::v), 0);
        OffsetT idx = idxStart;

        // full tile loop
        for (OffsetT iter = length / (IDX_STRIDE * UNROLL); iter > 0; --iter) {
            LoadData(ldgReg, ldgPtr, ldgStride);
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                reduceReg[i] = PreFunc()(static_cast<MathT>(ldgReg[i]));
            }

            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                ret = ReduceFunc()(
                      WithIndexFunc(reduceReg[i], idx + i * IDX_STRIDE), ret);
            }

            ldgPtr += UNROLL * ldgStride;
            idx += UNROLL * IDX_STRIDE;
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            reduceReg[i] = MathT(ReduceInitFunc::v);
        }

        LoadData<IDX_STRIDE>(ldgReg, ldgPtr, idx, ldgStride, length);

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (idx + i * IDX_STRIDE < length) {
                reduceReg[i] = PreFunc()(static_cast<MathT>(ldgReg[i]));
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            ret = ReduceFunc()(
                  WithIndexFunc(reduceReg[i], idx + i * IDX_STRIDE), ret);
        }
        return ret;
    }
};

template <bool RET_DATA, bool RET_IDX, typename PostFunc>
struct WriteBackTensor;

template <typename PostFunc>
struct WriteBackTensor<true, false, PostFunc> {
    template <typename ReduceT, typename DT, typename IdxT, typename OffsetT>
    static __device__ __forceinline__
    void Store(const ReduceT &val,
               DT *y,
               IdxT *indices,
               const DT &alpha,
               const OffsetT &offset) {
        y[offset] = PostFunc()(val) * alpha;
    }
};

template <bool RET_DATA, typename PostFunc>
struct WriteBackTensor<RET_DATA, true, PostFunc> {
    template <typename ReduceT, typename DT, typename IdxT, typename OffsetT>
    static __device__ __forceinline__
    void Store(const ReduceT &val,
               DT *y,
               IdxT *indices,
               const DT &alpha,
               const OffsetT &offset) {
        if (RET_DATA) {
            y[offset] = PostFunc()(val.data) * alpha;
        }
        indices[offset] = val.idx;
    }
};

template <bool USE_WS, bool RET_DATA, bool RET_IDX, typename PostFunc>
struct WriteBack;

template <bool RET_DATA, bool RET_IDX, typename PostFunc>
struct WriteBack<true, RET_DATA, RET_IDX, PostFunc> {
    template <typename ReduceT, typename DT, typename IdxT, typename OffsetT>
    static __device__ __forceinline__
    void Store(const ReduceT &val,
               ReduceT *ws,
               DT *y,
               IdxT *indices,
               const DT &alpha,
               const OffsetT &offset) {
        Stg(val, ws + offset);
    }
};

template <bool RET_DATA, bool RET_IDX, typename PostFunc>
struct WriteBack<false, RET_DATA, RET_IDX, PostFunc> {
    template <typename ReduceT, typename DT, typename IdxT, typename OffsetT>
    static __device__ __forceinline__
    void Store(const ReduceT &val,
               ReduceT *ws,
               DT *y,
               IdxT *indices,
               const DT &alpha,
               const OffsetT &offset) {
        WriteBackTensor<RET_DATA, RET_IDX, PostFunc>::
            Store(val, y, indices, alpha, offset);
    }
};

}  // namespace reduce

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_REDUCE_REDUCE_UTILS_CUH_


