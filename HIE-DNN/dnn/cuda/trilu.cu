/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    trilu.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <tensor_desc.hpp>
#include <utils.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, typename T, bool UPPER, bool NEG_K>
__global__ void TriluKernel(const T *x,
                            T *y,
                            U32DivMod nDivMod,
                            U32DivMod mDivMod,
                            uint32_t size,
                            int32_t k) {
    uint32_t blockOffset0 = blockIdx.x * BLOCK * UNROLL;
    // thread's first (without unrolling) y offset
    uint32_t offset0 = blockOffset0 + threadIdx.x;

    bool doCopy[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // assert: 0 <= mIdx < INT32_MAX
        auto mDM = mDivMod.DivMod(offset0 + i * BLOCK);
        int32_t mIdx = static_cast<int32_t>(mDM.mod);

        // assert: 0 <= nIdx < INT32_MAX
        auto nDM = nDivMod.DivMod(mDM.div);
        int32_t nIdx = static_cast<int32_t>(nDM.mod);

        // make sure no overflow when adding / subtracting k
        if (NEG_K) {
            // assert: -INT32_MAX <= -n <= k < 0
            doCopy[i] = UPPER ? nIdx + k <= mIdx : nIdx + k >= mIdx;
        } else {
            // assert: 0 <= k <= m <= INT32_MAX
            doCopy[i] = UPPER ? nIdx <= mIdx - k : nIdx >= mIdx - k;
        }
    }

    T regs[UNROLL];
    uint32_t ioCount = size > offset0 ?
                       UIntDivRU<uint32_t>(size - offset0, BLOCK) : 0;
    if (ioCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            regs[i] = doCopy[i] ? x[offset0 + i * BLOCK] : 0;
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            y[offset0 + i * BLOCK] = regs[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            if (i < ioCount) {
                regs[i] = doCopy[i] ? x[offset0 + i * BLOCK] : 0;
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            if (i < ioCount) {
                y[offset0 + i * BLOCK] = regs[i];
            }
        }
    }
}

template <typename T, bool UPPER>
hiednnStatus_t
LaunchTrilu(const T *x,
            T *y,
            int64_t n,
            int64_t m,  // shape of y is [*, n, m]
            uint32_t size,
            int64_t k,
            cudaStream_t stream) {
    constexpr uint32_t BLOCK = 128;
    constexpr uint32_t UNROLLED_BYTE = 16;
    constexpr uint32_t UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                                UNROLLED_BYTE / sizeof(T) : 8;

    /* If the shape of y is [*, n, m], it is guaranteed that
     *       n <= INT32_MAX, and m <= INT32_MAX.
     * So any index on the last 2 dims is *strictly less than* INT32_MAX.
     * Inside the kernel, we check ``index0 ? index1 - k'' for non-negative k
     * and ``index0 + k ? index1'' for negative k, where index0 and index1
     * are indices on the last 2 dims, with
     *       index0 < n, and index1 < m.
     *
     * 1) If -n < k < 0, since index0 \in [0, n), index0 + k \in (-n, n) which
     * is bounded by int32_t.
     *
     * 2) If 0 <= k < m, since index1 \in [0, m), index1 - k \in (-m, m) which
     * is also bounded by int32_t.
     *
     * 3) Otherwise, y must be either zero matrix or a copy of x.
     */
    // assert: 0 < n <= INT32_MAX, 0 < m <= INT32_MAX
    if (-n < k && k < m) {
        uint32_t nBlock = UIntDivRU(size, BLOCK * UNROLL);

        U32DivMod nDivMod(n);
        U32DivMod mDivMod(m);

        // assert: -INT32_MAX <= -n < k < m <= INT32_MAX
        if (k < 0) {
            TriluKernel<BLOCK, UNROLL, T, UPPER, true>
                <<<nBlock, BLOCK, 0, stream>>>(
                    x, y, nDivMod, mDivMod, size, static_cast<int32_t>(k));
        } else {
            TriluKernel<BLOCK, UNROLL, T, UPPER, false>
                <<<nBlock, BLOCK, 0, stream>>>(
                    x, y, nDivMod, mDivMod, size, static_cast<int32_t>(k));
        }
        CHECK_CUDA_RETURN(cudaGetLastError());
    } else {
        if ((k <= -n && !UPPER) || (k >= m && UPPER)) {
            // zero output
            CHECK_CUDA_RETURN(cudaMemsetAsync(
                y, 0, size * sizeof(T), stream));
        } else {  // (k <= -n && UPPER) || (k >= m && !UPPER)
            // same as memcpy
            CHECK_CUDA_RETURN(cudaMemcpyAsync(
                y, x, size * sizeof(T), cudaMemcpyDefault, stream));
        }
    }
    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct TriluImpl {
    hiednnStatus_t
    operator()(const HiednnTensorDesc &desc,
               const void *x,
               void *y,
               int64_t k,
               hiednnTriluOp_t op,
               cudaStream_t stream) {
        // fast div range check
        if (desc.size > UINT32_MAX ||
            desc.dims[1] > INT32_MAX ||
            desc.dims[2] > INT32_MAX) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        int64_t n = desc.dims[1];
        int64_t m = desc.dims[2];

        switch (op) {
            case HIEDNN_TRILU_UPPER:
                return LaunchTrilu<T, true>(
                    static_cast<const T *>(x), static_cast<T *>(y),
                    n, m, static_cast<uint32_t>(desc.size), k, stream);
            case HIEDNN_TRILU_LOWER:
                return LaunchTrilu<T, false>(
                    static_cast<const T *>(x), static_cast<T *>(y),
                    n, m, static_cast<uint32_t>(desc.size), k, stream);
            default:
                return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaTrilu(HiednnCudaHandle *cudaHandle,
                HiednnTensorDesc *xDesc,
                const void *x,
                HiednnTensorDesc *yDesc,
                void *y,
                int64_t k,
                hiednnTriluOp_t triluOp) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->nDims != 3 || *yDesc != *xDesc) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->size == 0UL) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchItemSize<hiednn::cuda::TriluImpl>(xDesc->dataType,
        *xDesc, x, y, k, triluOp, cudaHandle->stream);
}
