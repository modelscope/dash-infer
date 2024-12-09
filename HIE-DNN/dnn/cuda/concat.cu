/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    concat.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <datatype_dispatch.hpp>
#include <tensor_desc.hpp>
#include <utils.hpp>
#include <integer_divmod.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

constexpr int MAX_INPUT_COUNT = 128;

template <int BLOCK, int INPUT_COUNT, typename XConcatDimT, typename T>
__global__ void ConcatFastMapInitKernel(
        Array<const T *, INPUT_COUNT> xPtr,
        Array<uint32_t, INPUT_COUNT> xConcatDim,
        uint32_t yConcatDim,
        const T **xPtrRet,
        XConcatDimT *xConcatDimRet) {
    uint32_t xConcatDimIdx = blockIdx.x * BLOCK + threadIdx.x;
    if (xConcatDimIdx >= yConcatDim) {
        return;
    }

    uint32_t bar = 0;
    uint32_t offset = 0;
    uint32_t concatDimRet = 0;
    const T *ptrRet = nullptr;
    #pragma unroll
    for (int i = 0; i < INPUT_COUNT; ++i) {
        if (xConcatDimIdx >= bar) {
            offset = xConcatDimIdx - bar;
            concatDimRet = xConcatDim[i];
            ptrRet = xPtr[i];
        }
        bar += xConcatDim[i];
    }

    xPtrRet[xConcatDimIdx] = ptrRet + offset;
    xConcatDimRet[xConcatDimIdx] = concatDimRet;
}

template <int BLOCK,
          int UNROLL,
          typename XConcatDimT,
          typename T>
__global__ void ConcatFastMapKernel(
        const T * const *x,
        const XConcatDimT *xConcatDim,
        U32DivMod yConcatDimDivMod,
        uint32_t ySize,
        T *y) {
    uint32_t yOffset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    uint32_t batchIdx[UNROLL];
    uint32_t yConcatDimIdx[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        auto dm = yConcatDimDivMod.DivMod(yOffset0 + i * BLOCK);
        batchIdx[i] = dm.div;
        yConcatDimIdx[i] = dm.mod;
    }

    const T *xPtr[UNROLL];
    XConcatDimT xStride[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        Ldg<NC>(&xPtr[i], x + yConcatDimIdx[i]);
        Ldg<NC>(&xStride[i], xConcatDim + yConcatDimIdx[i]);
    }

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xPtr[i] += batchIdx[i] * xStride[i];
    }

    T data[UNROLL];
    uint32_t yCount = ySize > yOffset0 ?
                      UIntDivRU<uint32_t>(ySize - yOffset0, BLOCK) : 0;
    T *yPtr = y + yOffset0;

    if (yCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg<NC, LTCMAX>(&data[i], xPtr[i]);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            yPtr[i * BLOCK] = data[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                Ldg<NC, LTCMAX>(&data[i], xPtr[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                yPtr[i * BLOCK] = data[i];
            }
        }
    }
}

template <int INPUT_COUNT, typename XConcatDimT, typename T>
hiednnStatus_t ConcatFastMapInit(
        const T * const *x,
        const T **xPtrRet,
        XConcatDimT *xConcatDimRet,
        const int64_t *xConcatDims,
        const int64_t &yConcatDim,
        const int &inputCount,
        cudaStream_t stream) {
    const int BLOCK = 128;

    Array<const T *, INPUT_COUNT> xPtr(
        x, inputCount, (const T *)nullptr);
    Array<uint32_t, INPUT_COUNT> xConcatDim(
        xConcatDims, inputCount, int64_t(0));
    uint32_t grid = UIntDivRU<uint32_t>(yConcatDim, BLOCK);

    ConcatFastMapInitKernel<BLOCK, INPUT_COUNT, XConcatDimT, T>
        <<<grid, BLOCK, 0, stream>>>(
        xPtr, xConcatDim, yConcatDim, xPtrRet, xConcatDimRet);

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
hiednnStatus_t ConcatFastMap(
        const HiednnCudaHandle &handle,
        const T * const *x,
        T *y,
        const int64_t &batch,
        const int64_t *xConcatDims,
        const int64_t &yConcatDim,
        const int &inputCount) {
    using XConcatDimT = uint16_t;
    const size_t WS_MEMORY_ALIGN = 128;
    const int BLOCK = 128;
    const int UNROLL_BYTE = 16;
    const int MAX_UNROLL = 8;

    int64_t ySize = batch * yConcatDim;
    // yConcatDim is limited by XConcatDimT
    if (ySize > UINT32_MAX || yConcatDim > UINT16_MAX) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    // allocate workspace for xPtrWs and xConcatDimWs
    size_t xPtrWsSize = yConcatDim * sizeof(const T *);
    xPtrWsSize = UIntDivRU(xPtrWsSize, WS_MEMORY_ALIGN) * WS_MEMORY_ALIGN;
    size_t xConcatDimWsSize = yConcatDim * sizeof(XConcatDimT);
    char *ws;
    DeviceWsGuard wsGuard(handle);
    if (wsGuard.GetWorkspace(&ws, xPtrWsSize + xConcatDimWsSize) !=
        HIEDNN_STATUS_SUCCESS) {
        return HIEDNN_STATUS_INTERNAL_ERROR;
    }

    const T **xPtrWs = reinterpret_cast<const T **>(ws);
    XConcatDimT *xConcatDimWs =
        reinterpret_cast<XConcatDimT *>(ws + xPtrWsSize);

    // init xPtrWs and xConcatDimWs
    if (inputCount <= 16) {
        CHECK_HIEDNN_RETURN((ConcatFastMapInit<16, XConcatDimT, T>(
            x, xPtrWs, xConcatDimWs, xConcatDims, yConcatDim,
            inputCount, handle.stream)));
    } else if (inputCount <= 32) {
        CHECK_HIEDNN_RETURN((ConcatFastMapInit<32, XConcatDimT, T>(
            x, xPtrWs, xConcatDimWs, xConcatDims, yConcatDim,
            inputCount, handle.stream)));
    } else if (inputCount <= 64) {
        CHECK_HIEDNN_RETURN((ConcatFastMapInit<64, XConcatDimT, T>(
            x, xPtrWs, xConcatDimWs, xConcatDims, yConcatDim,
            inputCount, handle.stream)));
    } else if (inputCount <= MAX_INPUT_COUNT) {
        CHECK_HIEDNN_RETURN((ConcatFastMapInit<MAX_INPUT_COUNT, XConcatDimT, T>(
            x, xPtrWs, xConcatDimWs, xConcatDims, yConcatDim,
            inputCount, handle.stream)));
    } else {
        return HIEDNN_STATUS_INTERNAL_ERROR;
    }

    // concat
    const int UNROLL = UNROLL_BYTE / sizeof(T) < MAX_UNROLL ?
                       UNROLL_BYTE / sizeof(T) : MAX_UNROLL;
    uint32_t grid = UIntDivRU<uint32_t>(ySize, BLOCK * UNROLL);
    ConcatFastMapKernel<BLOCK, UNROLL, XConcatDimT, T>
        <<<grid, BLOCK, 0, handle.stream>>>(
        xPtrWs, xConcatDimWs, U32DivMod(yConcatDim), ySize, y);

    return HIEDNN_STATUS_SUCCESS;
}

/**
 * HybridSearcher search the input tensor informations (pointer, dimision...)
 * associated with the input parameter @idx. HybridSearcher do binary search
 * until SEARCH_LENGTH <= SERIAL_SEARCH_LENGTH, and turn to serial search (scan
 * from the first to the last) for SEARCH_LENGTH <= SERIAL_SEARCH_LENGTH to
 * avoid warp divergence.
 */
template <int INPUT_COUNT, int SERIAL_SEARCH_LENGTH, typename T>
struct HybridSearcher {
    static_assert(((INPUT_COUNT / SERIAL_SEARCH_LENGTH) &
                   (INPUT_COUNT / SERIAL_SEARCH_LENGTH - 1)) == 0,
                  "invalid INPUT_COUNT or SERIAL_SEARCH_LENGTH");
    using DimBarT = uint32_t;

    Array<const T *, INPUT_COUNT> xPtr;
    Array<DimBarT, INPUT_COUNT + 1> xConcatDimBar;

    template <typename DimT>
    HybridSearcher(const T * const *xPtrs,
                   const DimT *xConcatDims,
                   int inputCount) {
        xPtr = Array<const T *, INPUT_COUNT>(
            xPtrs, inputCount, (const T *)nullptr);

        DimBarT bar = 0;
        for (int i = 0; i < inputCount; ++i) {
            xConcatDimBar[i] = bar;
            bar += xConcatDims[i];
        }
        for (int i = inputCount; i < INPUT_COUNT + 1; ++i) {
            xConcatDimBar[i] = bar;
        }
    }

    template <int SEARCH_LENGTH,
              int SEARCH_START,
              typename DimRetT,
              typename IdxT>
    __device__ __forceinline__
    void SearchImpl(const T **ptrRet,
                    DimRetT *concatDimRet,
                    const IdxT &idx) {
        if (SEARCH_LENGTH > SERIAL_SEARCH_LENGTH) {
            // binary search
            if (idx >= xConcatDimBar[SEARCH_START + SEARCH_LENGTH / 2]) {
                return SearchImpl<SEARCH_LENGTH / 2,
                                  SEARCH_START + SEARCH_LENGTH / 2>(
                    ptrRet, concatDimRet, idx);
            } else {
                return SearchImpl<SEARCH_LENGTH / 2, SEARCH_START>(
                    ptrRet, concatDimRet, idx);
            }
        } else {
            // serial search
            DimBarT offset = SEARCH_START == 0 ?
                             idx : idx - xConcatDimBar[SEARCH_START];
            *ptrRet = xPtr[SEARCH_START];
            *concatDimRet = SEARCH_START == 0 ?
                            xConcatDimBar[1] :
                            xConcatDimBar[SEARCH_START + 1] -
                            xConcatDimBar[SEARCH_START];
            #pragma unroll
            for (int i = 1; i < SEARCH_LENGTH; ++i) {
                if (idx >= xConcatDimBar[SEARCH_START + i]) {
                    offset = idx - xConcatDimBar[SEARCH_START + i];
                    *ptrRet = xPtr[SEARCH_START + i];
                    *concatDimRet = xConcatDimBar[SEARCH_START + i + 1] -
                                    xConcatDimBar[SEARCH_START + i];
                }
            }
            *ptrRet += offset;
        }
    }

    template <typename DimRetT, typename IdxT>
    __device__ __forceinline__
    void Search(const T **ptrRet, DimRetT *concatDimRet, const IdxT &idx) {
        SearchImpl<INPUT_COUNT, 0, DimRetT, IdxT>(ptrRet, concatDimRet, idx);
    }
};

template <int INPUT_COUNT, typename T>
using SerialSearcher = HybridSearcher<INPUT_COUNT, INPUT_COUNT, T>;

template <int BLOCK_X,
          int BLOCK_Y,
          int UNROLL,
          typename SearcherT,
          typename T>
__global__ void ConcatKernel(
        T *y,
        SearcherT searcher,
        U32DivMod yConcatDimDivMod,
        uint32_t batch,
        uint32_t yConcatDim) {
    __shared__ const T *xPtrSmem[BLOCK_X];
    __shared__ uint32_t xStrideSmem[BLOCK_X];

    uint32_t tidX = threadIdx.x % BLOCK_X;
    uint32_t tidY = threadIdx.x / BLOCK_X;

    auto dm = yConcatDimDivMod.DivMod(blockIdx.x * BLOCK_X + tidX);
    uint32_t idxX = dm.mod;
    uint32_t idxY = dm.div * BLOCK_Y * UNROLL + tidY;

    // search input tensors
    if (tidY == 0) {
        const T *xPtr;
        uint32_t xConcatDim;
        searcher.Search(&xPtr, &xConcatDim, idxX);

        xPtrSmem[tidX] = xPtr + idxY * xConcatDim;
        xStrideSmem[tidX] = xConcatDim;
    }
    __syncthreads();

    if (idxY >= batch) {
        return;
    }

    // init LDG pointer
    uint32_t xStride = xStrideSmem[tidX];
    const T *xPtrs[UNROLL];
    xPtrs[0] = xPtrSmem[tidX] + xStride * tidY;
    #pragma unroll
    for (int i = 1; i < UNROLL; ++i) {
        xPtrs[i] = xPtrs[0] + xStride * BLOCK_Y * i;
    }

    // load and store
    uint32_t yCount = UIntDivRU<uint32_t>(batch - idxY, BLOCK_Y);
    T *yPtr = y + idxY * yConcatDim + idxX;
    T dataReg[UNROLL];

    if (yCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg<NC, LTCMAX>(&dataReg[i], xPtrs[i]);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            yPtr[i * yConcatDim * BLOCK_Y] = dataReg[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                Ldg<NC, LTCMAX>(&dataReg[i], xPtrs[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                yPtr[i * yConcatDim * BLOCK_Y] = dataReg[i];
            }
        }
    }
}

template <int INPUT_COUNT,
          typename SearcherT,
          typename T>
hiednnStatus_t LaunchConcatKernel(
        const HiednnCudaHandle &handle,
        const T * const *x,
        T *y,
        const int64_t &batch,
        const int64_t *xConcatDims,
        const int64_t &yConcatDim,
        const int &inputCount) {
    const int BLOCK_X = 64;
    const int BLOCK_Y = 8;
    const int UNROLL = INPUT_COUNT <= 32 ?
                       (16 / sizeof(T) < 8 ? 16 / sizeof(T) : 8) : 8;

    SearcherT searcher(x, xConcatDims, inputCount);
    U32DivMod yConcatDimDivMod(yConcatDim);
    uint32_t blockXSum = UIntDivRU<uint32_t>(batch, BLOCK_Y * UNROLL) *
                         yConcatDim;
    uint32_t grid = UIntDivRU<uint32_t>(blockXSum, BLOCK_X);

    ConcatKernel<BLOCK_X, BLOCK_Y, UNROLL, SearcherT, T>
        <<<grid, BLOCK_X * BLOCK_Y, 0, handle.stream>>>(
        y, searcher, yConcatDimDivMod, batch, yConcatDim);

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
hiednnStatus_t Concat(
        const HiednnCudaHandle &handle,
        const T * const *x,
        T *y,
        const int64_t &batch,
        const int64_t *xConcatDims,
        const int64_t &yConcatDim,
        const int &inputCount) {
    // limited by integer fast division
    if (batch * yConcatDim > UINT32_MAX || yConcatDim > INT32_MAX) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    hiednnStatus_t ret;

    if (inputCount <= 4) {
        ret = LaunchConcatKernel<4, SerialSearcher<4, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= 8) {
        ret = LaunchConcatKernel<8, SerialSearcher<8, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= 16) {
        ret = LaunchConcatKernel<16, SerialSearcher<16, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= 24) {
        ret = LaunchConcatKernel<24, SerialSearcher<24, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= 32) {
        ret = LaunchConcatKernel<32, SerialSearcher<32, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= 48) {
        ret = LaunchConcatKernel<48, SerialSearcher<48, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= 64) {
        ret = LaunchConcatKernel<64, HybridSearcher<64, 32, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else if (inputCount <= MAX_INPUT_COUNT) {
        ret = LaunchConcatKernel<MAX_INPUT_COUNT,
                                 HybridSearcher<MAX_INPUT_COUNT, 32, T>>(
            handle, x, y, batch, xConcatDims, yConcatDim, inputCount);
    } else {
        ret = HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return ret;
}

template <int BLOCK, int UNROLL, typename SearcherT, typename T>
__global__ void ConcatSmallBatchKernel(
        T *y,
        SearcherT searcher,
        U32DivMod yConcatDimDivMod,
        uint32_t ySize) {
    uint32_t yOffset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    // batch index and yConcatDim index
    uint32_t batchIdx[UNROLL];
    uint32_t yConcatDimIdx[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        auto dm = yConcatDimDivMod.DivMod(yOffset0 + i * BLOCK);
        batchIdx[i] = dm.div;
        yConcatDimIdx[i] = dm.mod;
    }

    // search input tensors
    uint32_t xConcatDims[UNROLL];
    const T *xPtrs[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        searcher.Search(&xPtrs[i], &xConcatDims[i], yConcatDimIdx[i]);
    }
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xPtrs[i] += batchIdx[i] * xConcatDims[i];
    }

    // load and store
    uint32_t yCount = ySize > yOffset0 ?
                      UIntDivRU<uint32_t>(ySize - yOffset0, BLOCK) : 0;
    T *yPtr = y + yOffset0;
    T data[UNROLL];

    if (yCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg<CACHE_DEFAULT, LTCMAX>(&data[i], xPtrs[i]);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            yPtr[i * BLOCK] = data[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                Ldg<CACHE_DEFAULT, LTCMAX>(&data[i], xPtrs[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < yCount) {
                yPtr[i * BLOCK] = data[i];
            }
        }
    }
}

template <typename T>
hiednnStatus_t ConcatSmallBatch(
        const HiednnCudaHandle &handle,
        const T * const *x,
        T *y,
        const int64_t &batch,
        const int64_t *xConcatDims,
        const int64_t &yConcatDim,
        const int &inputCount) {
    const int BLOCK = 128;
    const int UNROLL = 16 / sizeof(T) < 8 ? 16 / sizeof(T) : 8;
    const int SERIAL_SEARCH = 4;

    int64_t ySize = batch * yConcatDim;

    // limited by integer fast division
    if (ySize > UINT32_MAX || yConcatDim > INT32_MAX) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    uint32_t grid = UIntDivRU<uint32_t>(ySize, BLOCK * UNROLL);
    U32DivMod yConcatDimDivMod(yConcatDim);

    if (inputCount <= 4) {
        using SearcherT = HybridSearcher<4, SERIAL_SEARCH, T>;
        SearcherT searcher(x, xConcatDims, inputCount);
        ConcatSmallBatchKernel<BLOCK, UNROLL, SearcherT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            y, searcher, yConcatDimDivMod, ySize);
    } else if (inputCount <= 8) {
        using SearcherT = HybridSearcher<8, SERIAL_SEARCH, T>;
        SearcherT searcher(x, xConcatDims, inputCount);
        ConcatSmallBatchKernel<BLOCK, UNROLL, SearcherT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            y, searcher, yConcatDimDivMod, ySize);
    } else if (inputCount <= 16) {
        using SearcherT = HybridSearcher<16, SERIAL_SEARCH, T>;
        SearcherT searcher(x, xConcatDims, inputCount);
        ConcatSmallBatchKernel<BLOCK, UNROLL, SearcherT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            y, searcher, yConcatDimDivMod, ySize);
    } else if (inputCount <= 32) {
        using SearcherT = HybridSearcher<32, SERIAL_SEARCH, T>;
        SearcherT searcher(x, xConcatDims, inputCount);
        ConcatSmallBatchKernel<BLOCK, UNROLL, SearcherT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            y, searcher, yConcatDimDivMod, ySize);
    } else if (inputCount <= 64) {
        using SearcherT = HybridSearcher<64, SERIAL_SEARCH, T>;
        SearcherT searcher(x, xConcatDims, inputCount);
        ConcatSmallBatchKernel<BLOCK, UNROLL, SearcherT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            y, searcher, yConcatDimDivMod, ySize);
    } else if (inputCount <= MAX_INPUT_COUNT) {
        using SearcherT = HybridSearcher<MAX_INPUT_COUNT, SERIAL_SEARCH, T>;
        SearcherT searcher(x, xConcatDims, inputCount);
        ConcatSmallBatchKernel<BLOCK, UNROLL, SearcherT, T>
            <<<grid, BLOCK, 0, handle.stream>>>(
            y, searcher, yConcatDimDivMod, ySize);
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct ConcatImpl {
    template <typename ImplT>
    struct Impl {
        hiednnStatus_t operator()(
                const HiednnCudaHandle &handle,
                const void * const *x,
                void *y,
                const int64_t &batch,
                const int64_t *xConcatDims,
                const int64_t &yConcatDim,
                const int &inputCount) {
            const ImplT * const *xPtr =
                reinterpret_cast<const ImplT * const *>(x);
            ImplT *yPtr = static_cast<ImplT *>(y);

            if (batch < 128) {
                return ConcatSmallBatch<ImplT>(
                    handle, xPtr, yPtr, batch, xConcatDims, yConcatDim,
                    inputCount);
            } else if (yConcatDim <= 512) {
                return ConcatFastMap<ImplT>(
                    handle, xPtr, yPtr, batch, xConcatDims, yConcatDim,
                    inputCount);
            } else {
                return Concat<ImplT>(
                    handle, xPtr, yPtr, batch, xConcatDims, yConcatDim,
                    inputCount);
            }
        }
    };

    size_t MaxAlignedByte(
            const void * const *x,
            const int64_t *xConcatDims,
            const int &inputCount) {
        const int MAX_ALIGNED_BYTE = 8;

        size_t ret = MAX_ALIGNED_BYTE;

        for (int i = 0; i < inputCount; ++i) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(x[i]);
            size_t concatByte = sizeof(T) * xConcatDims[i];
            size_t alignedByte = 1;

            for (size_t i = 2; i <= MAX_ALIGNED_BYTE; i *= 2) {
                if (addr % i == 0 && concatByte % i == 0) {
                    alignedByte = i;
                }
            }
            ret = alignedByte < ret ? alignedByte : ret;
        }
        return ret;
    }

    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            const void * const *x,
            void *y,
            const int64_t &batch,
            const int64_t *xConcatDims,
            const int64_t &yConcatDim,
            const int &inputCount) {
        if (inputCount < 2) {
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }

        size_t alignedByte = MaxAlignedByte(x, xConcatDims, inputCount);
        size_t packSize = alignedByte / sizeof(T);
        int64_t yConcatDimPacked = yConcatDim / packSize;
        int64_t xConcatDimPacked[128];
        for (int i = 0; i < inputCount; ++i) {
            xConcatDimPacked[i] = xConcatDims[i] / packSize;
        }

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;

        switch (alignedByte) {
            case 1:
                ret = Impl<uint8_t>()(
                    handle, x, y, batch, xConcatDimPacked, yConcatDimPacked,
                    inputCount);
                break;
            case 2:
                ret = Impl<uint16_t>()(
                    handle, x, y, batch, xConcatDimPacked, yConcatDimPacked,
                    inputCount);
                break;
            case 4:
                ret = Impl<uint32_t>()(
                    handle, x, y, batch, xConcatDimPacked, yConcatDimPacked,
                    inputCount);
                break;
            case 8:
                ret = Impl<uint64_t>()(
                    handle, x, y, batch, xConcatDimPacked, yConcatDimPacked,
                    inputCount);
                break;
            default:
                ret = HIEDNN_STATUS_INTERNAL_ERROR;
                break;
        }
        return ret;
    }
};

void ConcatParameterPreprocess(
        int64_t *batch,
        int64_t *xConcatDims,
        int64_t *yConcatDim,
        HiednnTensorDesc * const *xDescs,
        int inputCount,
        int axis,
        HiednnTensorDesc *yDesc) {
    int nDims = yDesc->nDims;
    int concatAxis = axis < 0 ? axis + yDesc->nDims : axis;

    // batch
    *batch = 1;
    for (int i = 0; i < concatAxis; ++i) {
        *batch *= yDesc->dims[i];
    }

    // concat dims of input tensor
    for (int i = 0; i < inputCount; ++i) {
        xConcatDims[i] = 1;
        for (int j = concatAxis; j < nDims; ++j) {
            xConcatDims[i] *= xDescs[i]->dims[j];
        }
    }

    // concat dims of output tensor
    *yConcatDim = 1;
    for (int i = concatAxis; i < nDims; ++i) {
        *yConcatDim *= yDesc->dims[i];
    }
}

hiednnStatus_t ConcatParameterCheck(
        HiednnTensorDesc * const *xDescs,
        const void * const *x,
        int inputCount,
        int axis,
        HiednnTensorDesc *yDesc) {
    if (inputCount < 1 || yDesc->size == 0) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    int concatAxis = axis < 0 ? axis + yDesc->nDims : axis;
    int nDims = yDesc->nDims;
    const int64_t *yDims = yDesc->dims;
    hiednnDataType_t yType = yDesc->dataType;

    if (axis < -nDims || axis >= nDims) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    int64_t concatDim = 0;
    for (int i = 0; i < inputCount; ++i) {
        if (x[i] == nullptr || !CheckNormalFormat(*xDescs[i]) ||
            xDescs[i]->nDims != nDims || xDescs[i]->dataType != yType) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        for (int j = 0; j < nDims; ++j) {
            if (j != concatAxis) {
                if (xDescs[i]->dims[j] != yDims[j]) {
                    return HIEDNN_STATUS_INVALID_PARAMETER;
                }
            } else {
                concatDim += xDescs[i]->dims[j];
            }
        }
    }

    if (concatDim != yDims[concatAxis]) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t RemoveEmptyInputs(
        HiednnTensorDesc **xDescsRet,
        const void **xRet,
        int *inputCountRet,
        HiednnTensorDesc * const *xDescs,
        const void * const *x,
        int inputCount) {
    hiednnStatus_t statRet = HIEDNN_STATUS_SUCCESS;
    int count = 0;
    for (int i = 0; i < inputCount; ++i) {
        if (xDescs[i]->size != 0) {
            xDescsRet[count] = xDescs[i];
            xRet[count] = x[i];
            ++count;
            if (x[i] == nullptr) {
                statRet = HIEDNN_STATUS_INVALID_PARAMETER;
            }
        }
    }
    *inputCountRet = count;
    return statRet;
}

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaConcat(HiednnCudaHandle *cudaHandle,
                 HiednnTensorDesc * const *xDescs,
                 const void * const *x,
                 int inputCount,
                 int axis,
                 HiednnTensorDesc *yDesc,
                 void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDescs, x, yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    CHECK_HIEDNN_RETURN(hiednn::cuda::ConcatParameterCheck(
        xDescs, x, inputCount, axis, yDesc));

    // remove empty tensors (tensors with dim 0) from input.
    HiednnTensorDesc *inDescs[hiednn::cuda::MAX_INPUT_COUNT];
    const void *inTensors[hiednn::cuda::MAX_INPUT_COUNT];
    int tensorCount;
    CHECK_HIEDNN_RETURN(hiednn::cuda::RemoveEmptyInputs(
        inDescs, inTensors, &tensorCount, xDescs, x, inputCount));

    // all input tensors are empty.
    if (tensorCount == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    if (tensorCount == 1) {
        size_t size = inDescs[0]->size *
                      hiednn::ItemSizeInByte(inDescs[0]->dataType);
        CHECK_CUDA_RETURN(cudaMemcpyAsync(
            y, inTensors[0], size, cudaMemcpyDefault, cudaHandle->stream));
        return HIEDNN_STATUS_SUCCESS;
    } else {
        int64_t batch;
        int64_t yConcatDim;
        int64_t xConcatDims[hiednn::cuda::MAX_INPUT_COUNT];
        if (tensorCount > hiednn::cuda::MAX_INPUT_COUNT) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        hiednn::cuda::ConcatParameterPreprocess(
            &batch, xConcatDims, &yConcatDim, inDescs, tensorCount,
            axis, yDesc);

        return hiednn::DispatchItemSize<hiednn::cuda::ConcatImpl>(
            yDesc->dataType, *cudaHandle, inTensors, y,
            batch, xConcatDims, yConcatDim, tensorCount);
    }
}


