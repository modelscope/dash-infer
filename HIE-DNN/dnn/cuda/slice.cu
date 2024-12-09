/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    slice.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <packed_memory_access.hpp>
#include <tensor_desc.hpp>
#include <utils.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, typename T, int AXES, typename IDX_T>
__global__ void SliceNormalKernel(
        const T *x,
        T *y,
        uint32_t ySize,
        Array<IDX_T, AXES> starts,
        Array<IDX_T, AXES> steps,
        Array<IDX_T, AXES> xStrides,
        Array<U32DivMod, AXES> yStrideDivMods) {
    // thread's first (without unrolling) y offset
    uint32_t yOffset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    // inner offset is bounded by uint32_t, because ySize <= UINT32_MAX
    uint32_t innerOffsets[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        innerOffsets[i] = yOffset0 + i * BLOCK;
    }

    IDX_T xOffsets[UNROLL]{0};
    // use AXES as outer loop to possibly reduce const mem loading
    #pragma unroll
    for (int dim = 0; dim < AXES; dim++) {
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            auto divmod = yStrideDivMods[dim].DivMod(innerOffsets[i]);
            /* yIndex as an intermediate variable is eliminated:
             *      yIndex[dim] = divmod.div;
             *      xIndex[dim] = starts[dim] + yIndex[dim] * steps[dim];
             *
             * Unsigned divmod.div needs casting down to signed type.
             * This is safe because yIndex <= xIndex <= xDim, and xDim is
             * ensured to be in-bound.
             */
            IDX_T xIdx = starts[dim] + divmod.div * steps[dim];
            xOffsets[i] += xIdx * xStrides[dim];
            innerOffsets[i] = divmod.mod;
        }
    }

    T regs[UNROLL];

    uint32_t yCount = ySize > yOffset0 ?
                      UIntDivRU<uint32_t>(ySize - yOffset0, BLOCK) : 0;
    if (yCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            regs[i] = x[xOffsets[i] + innerOffsets[i]];
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            y[yOffset0 + i * BLOCK] = regs[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            if (i < yCount) {
                regs[i] = x[xOffsets[i] + innerOffsets[i]];
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; i++) {
            if (i < yCount) {
                y[yOffset0 + i * BLOCK] = regs[i];
            }
        }
    }
}

template <typename T, int AXES>
hiednnStatus_t
LaunchSliceNormal(const T *x, T *y, size_t xSize, uint32_t ySize,
                  const int64_t (&starts)[TENSOR_DIM_MAX],
                  const int64_t (&steps)[TENSOR_DIM_MAX],
                  const int64_t (&xDims)[AXES],
                  const int64_t (&xStrides)[AXES],
                  const int64_t (&yStrides)[AXES],
                  cudaStream_t stream) {
    constexpr uint32_t BLOCK = 128;
    constexpr uint32_t UNROLLED_BYTE = 16;
    constexpr uint32_t UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                                UNROLLED_BYTE / sizeof(T) : 8;
    uint32_t nBlock = UIntDivRU(ySize, BLOCK * UNROLL);

    /* xDims <= INT32_MAX means xIndex & starts <= INT32_MAX,
     * and yIndex <= xIndex <= INT32_MAX.
     *
     * xSize <= UINT32_MAX means xStrides <= UINT32_MAX.
     *
     * Though steps may be out-of-bound, in-bound xDims ensures
     * any out-of-bound step will only multiply with zero,
     * and the result will be safe.
     */
    bool xNotLong = xSize <= UINT32_MAX;
    for (int i = 0; i < AXES; i++) {
        xNotLong = xNotLong && (xDims[i] <= INT32_MAX);
    }

    // negative step is compatible with unsigned integer multiply and add
    if (xNotLong) {
        using IDX_T = uint32_t;
        SliceNormalKernel<BLOCK, UNROLL, T, AXES, IDX_T>
            <<<nBlock, BLOCK, 0, stream>>>(
                x, y, ySize, Array<IDX_T, AXES>(starts, AXES),
                Array<IDX_T, AXES>(steps, AXES),
                Array<IDX_T, AXES>(xStrides, AXES),
                Array<U32DivMod, AXES>(yStrides, AXES));
    } else {
        using IDX_T = uint64_t;
        SliceNormalKernel<BLOCK, UNROLL, T, AXES, IDX_T>
            <<<nBlock, BLOCK, 0, stream>>>(
                x, y, ySize, Array<IDX_T, AXES>(starts, AXES),
                Array<IDX_T, AXES>(steps, AXES),
                Array<IDX_T, AXES>(xStrides, AXES),
                Array<U32DivMod, AXES>(yStrides, AXES));
    }
    return HIEDNN_STATUS_SUCCESS;
}

template <typename T, int AXES>
hiednnStatus_t
SliceCompleteAndPack(const HiednnTensorDesc &xDesc,
                     const T *x,
                     const HiednnTensorDesc &yDesc,
                     T *y,
                     int64_t (&starts)[TENSOR_DIM_MAX],
                     int64_t (&steps)[TENSOR_DIM_MAX],
                     cudaStream_t stream) {
    static_assert(sizeof(T) <= 8, "invalid typename T");

    // complete axes which are still missing
    for (int i = 0; i < AXES; i++) {
        // outSteps[i] == 0 is a flag indicating a missing axis
        if (steps[i] == 0L) {
            starts[i] = 0L;
            steps[i] = 1L;
        }
    }

    const int &nDims = yDesc.nDims;

    // decide pack size
    size_t innerSize = 1UL;
    for (int i = AXES; i < nDims; i++) {
        innerSize *= yDesc.dims[i];
    }
    size_t packSize = std::min(GetPackSize(x), GetPackSize(y));
    while (innerSize % packSize != 0UL) {
        packSize /= 2UL;
    }

    // pack at most 8 Bytes
    size_t packBytes = packSize * sizeof(T) < 8UL ?
                       packSize * sizeof(T) : 8UL;
    packSize = packBytes / sizeof(T);

    /* By packing, inner dims collapse to one dimension.
     * Packed tensors are of (AXES + 1) dims,
     * dims[AXES] == stride[AXES - 1] == innerSize / packSize,
     * stride[AXES] == 1, and we just omit dims[AXES] & stride[AXES].
     *
     * Valid innerSize / packSize <= INT32_MAX, because yStrides <= INT32_MAX.
     * When AXES >= nDims, innerSize == packSize == 1.
     */
    int64_t xDimsPacked[AXES];
    int64_t xStridesPacked[AXES];
    int64_t yStridesPacked[AXES];

    for (int i = 0; i < AXES; ++i) {
        xDimsPacked[i] = xDesc.dims[i];
        xStridesPacked[i] = xDesc.strides[i] / packSize;
        yStridesPacked[i] = yDesc.strides[i] / packSize;
    }

    if (nDims <= AXES) {
        xDimsPacked[nDims - 1] = xDesc.dims[nDims - 1] / packSize;
        xStridesPacked[nDims - 1] = 1L;
        yStridesPacked[nDims - 1] = 1L;
    }

    // deal with tailing missing dims, if any
    for (int i = nDims; i < AXES; ++i) {
        xDimsPacked[i] = 1L;
        xStridesPacked[i] = 1L;
        yStridesPacked[i] = 1L;
    }

    size_t xSizePacked = xDesc.size / packSize;
    size_t ySizePacked = yDesc.size / packSize;
    // range check: fast DivMod only support uint32_t for now
    if (ySizePacked > UINT32_MAX
        || U32DivMod::OutOfBound(yStridesPacked, AXES)) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    // re-type packed inputs
    switch (packBytes) {
        case 1UL:
            return LaunchSliceNormal<uint8_t, AXES>(
                reinterpret_cast<const uint8_t *>(x),
                reinterpret_cast<uint8_t *>(y),
                xSizePacked, static_cast<uint32_t>(ySizePacked), starts, steps,
                xDimsPacked, xStridesPacked, yStridesPacked, stream);
        case 2UL:
            return LaunchSliceNormal<uint16_t, AXES>(
                reinterpret_cast<const uint16_t *>(x),
                reinterpret_cast<uint16_t *>(y),
                xSizePacked, static_cast<uint32_t>(ySizePacked), starts, steps,
                xDimsPacked, xStridesPacked, yStridesPacked, stream);
        case 4UL:
            return LaunchSliceNormal<uint32_t, AXES>(
                reinterpret_cast<const uint32_t *>(x),
                reinterpret_cast<uint32_t *>(y),
                xSizePacked, static_cast<uint32_t>(ySizePacked), starts, steps,
                xDimsPacked, xStridesPacked, yStridesPacked, stream);
        case 8UL:
            return LaunchSliceNormal<uint64_t, AXES>(
                reinterpret_cast<const uint64_t *>(x),
                reinterpret_cast<uint64_t *>(y),
                xSizePacked, static_cast<uint32_t>(ySizePacked), starts, steps,
                xDimsPacked, xStridesPacked, yStridesPacked, stream);
        default:
            return HIEDNN_STATUS_INTERNAL_ERROR;
    }
}

template <typename T>
struct SliceImpl {
    hiednnStatus_t operator()(const HiednnTensorDesc &xDesc,
                              const void *x,
                              const HiednnTensorDesc &yDesc,
                              void *y,
                              int64_t (&starts)[TENSOR_DIM_MAX],
                              int64_t (&steps)[TENSOR_DIM_MAX],
                              int nAxes,
                              cudaStream_t stream) {
        // no axis is sliced, i.e. just memcpy
        if (nAxes == 0) {
            // param check ensures xDesc == yDesc
            if (x == y) {
                return HIEDNN_STATUS_SUCCESS;
            } else {
                return CHECK_CUDA(cudaMemcpyAsync(
                    y, x, xDesc.size * sizeof(T), cudaMemcpyDefault, stream));
            }
        }

        const T *xPtr = static_cast<const T *>(x);
        T *yPtr = static_cast<T *>(y);
        // guaranteed nAxes > 0
        if (nAxes <= 1) {
            return SliceCompleteAndPack<T, 1>(xDesc, xPtr, yDesc, yPtr,
                                              starts, steps, stream);
        } else if (nAxes <= 2) {
            return SliceCompleteAndPack<T, 2>(xDesc, xPtr, yDesc, yPtr,
                                              starts, steps, stream);
        } else if (nAxes <= 4) {
            return SliceCompleteAndPack<T, 4>(xDesc, xPtr, yDesc, yPtr,
                                              starts, steps, stream);
        } else if (nAxes <= 8) {
            return SliceCompleteAndPack<T, 8>(xDesc, xPtr, yDesc, yPtr,
                                              starts, steps, stream);
        } else {
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }
    }
};

/**
 * @brief Check slice output shape.
 *
 * @retval true if the output shape is computed correctly
 * @retval false if the output shape is incorrect
 */
bool
SliceCheckShape(const HiednnTensorDesc &xDesc,
                const HiednnTensorDesc &yDesc,
                const int64_t (&starts)[TENSOR_DIM_MAX],
                const int64_t (&ends)[TENSOR_DIM_MAX],
                const int64_t (&steps)[TENSOR_DIM_MAX]) {
    for (int i = 0; i < xDesc.nDims; i++) {
        if (steps[i] != 0) {
            // sliced dims are computed using ceil((end - start) / step)
            int64_t dim = SIntDivRU(ends[i] - starts[i], steps[i]);
            /* NOTE: when dim computed above is negative, the output dim is zero.
             * This can be regarded as:
             *      for (int dim = start; dim < [or >] end; dim += step) {...}
             * So we need to clamp dim to be a non-negative value here.
             */
            dim = std::max(0L, dim);
            if (dim != yDesc.dims[i]) {
                return false;
            }
        } else {
            // inner dims must match
            if (xDesc.dims[i] != yDesc.dims[i]) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Tensor check, null check, range check, axis duplication check.
 *
 * @retval true if all parameters are valid
 * @retval false if any parameter is invalid
 */
bool
SliceCheckParam(const HiednnTensorDesc *xDesc,
                const HiednnTensorDesc *yDesc,
                const int64_t *starts,
                const int64_t *ends,
                const int64_t *steps,
                const int *axes,
                int nParams) {
    // tensor property check
    if (xDesc->nDims != yDesc->nDims || xDesc->dataType != yDesc->dataType) {
        return false;
    }

    const auto &rank = xDesc->nDims;
    if (!starts || !ends || nParams < 0 || nParams > rank) {
        return false;
    }

    // range & value check
    if (steps) {
        for (int i = 0; i < nParams; i++) {
            // step cannot be zero
            if (steps[i] == 0L) {
                return false;
            }
        }
    }

    if (axes) {
        int axisCount[TENSOR_DIM_MAX]{0};
        for (int i = 0; i < nParams; i++) {
            // accepted range is [-r, r-1]
            if (axes[i] < -rank || axes[i] >= rank) {
                return false;
            }

            // adjust negative axes
            int axis = axes[i] < 0 ? axes[i] + rank : axes[i];
            // forbid axis duplication
            if (++axisCount[axis] > 1) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Negative adjustment, clamp, sort inputs by axes.
 *
 * @return int the number of axes to slice through; guaranteed to be
 * greater than or equal to 0
 *
 * @pre @c outSteps is zero-initialized.
 */
int
SlicePreprocessParam(int64_t (&outStarts)[TENSOR_DIM_MAX],
                     int64_t (&outEnds)[TENSOR_DIM_MAX],
                     int64_t (&outSteps)[TENSOR_DIM_MAX],
                     const HiednnTensorDesc &xDesc,
                     const int64_t *starts,
                     const int64_t *ends,
                     const int64_t *steps,
                     const int *axes,
                     int nParams) {
    /* Note that steps can never be zero, so we use step == 0 as a flag
     * indicating a missing axis.
     * By pre-condition, outSteps has been zero-init, i.e.
     *      (void) memset(outSteps, 0, rank * sizeof(int64_t));
     */

    const auto &rank = xDesc.nDims;
    int maxAxis = -1;
    for (int i = 0; i < nParams; i++) {
        // default steps are [1, 1, ..., 1]
        int64_t step = steps ? steps[i] : 1L;
        // default axes are [0, 1, ..., nDim]
        int axis = axes ? (axes[i] < 0 ? axes[i] + rank : axes[i]) : i;

        // adjust starts & ends
        const auto &dim = xDesc.dims[axis];
        int64_t start = starts[i] < 0L ? starts[i] + dim : starts[i];
        int64_t end = ends[i] < 0L ? ends[i] + dim : ends[i];

        // [0:dim:1] equals to no slicing along this axis
        if (step == 1L && start <= 0L && end >= dim) {
            continue;
        }

        if (step > 0L) {
            /* starts[i] is clamped into the range [0, dims[axes[i]]]
             * for positive stepping
             */
            start = std::min(std::max(start, 0L), dim);
            /* ends[i] is clamped into the range [0, dims[axes[i]]]
             * for positive stepping
             */
            end = std::min(std::max(end, 0L), dim);
        } else {
            /* starts[i] is clamped into the range [0, dims[axes[i]]-1]
             * for negative stepping
             */
            start = std::min(std::max(start, 0L), dim - 1L);
            /* ends[i] is clamped into the range [-1, dims[axes[i]]-1]
             * for negative stepping
             */
            end = std::min(std::max(end, -1L), dim - 1L);
        }

        outStarts[axis] = start;
        outEnds[axis] = end;
        outSteps[axis] = step;
        maxAxis = std::max(maxAxis, axis);
    }

    // when nParam == 0, maxAxis is -1 and thus nAxes is 0
    return maxAxis + 1;
}

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaSlice(HiednnCudaHandle *cudaHandle,
                HiednnTensorDesc *xDesc,
                const void *x,
                const int64_t *starts,
                const int64_t *ends,
                const int64_t *steps,
                const int *axes,
                int nParams,
                HiednnTensorDesc *yDesc,
                void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, starts, ends, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::cuda::SliceCheckParam(
            xDesc, yDesc, starts, ends, steps, axes, nParams)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    int64_t mStarts[TENSOR_DIM_MAX];
    int64_t mEnds[TENSOR_DIM_MAX];
    int64_t mSteps[TENSOR_DIM_MAX]{0};

    // guaranteed nAxes >= 0
    int nAxes = hiednn::cuda::SlicePreprocessParam(
        mStarts, mEnds, mSteps, *xDesc, starts, ends, steps, axes, nParams);

    if (!hiednn::cuda::SliceCheckShape(
            *xDesc, *yDesc, mStarts, mEnds, mSteps)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    /* We are sure the shape is computed correctly,
     * so just return success if the size is zero.
     */
    if (xDesc->size == 0UL || yDesc->size == 0UL) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchItemSize<hiednn::cuda::SliceImpl>(
        xDesc->dataType, *xDesc, x, *yDesc, y,
        mStarts, mSteps, nAxes, cudaHandle->stream);
}
