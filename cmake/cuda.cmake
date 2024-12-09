include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "No CUDA support")
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g ")
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS}  -Xcompiler -ldl --threads 2 --expt-relaxed-constexpr --extended-lambda")

# support V100/A100
set(CMAKE_CUDA_ARCHITECTURES
    "80;86"
    CACHE STRING "CUDA SM")

set(CUDA_VERSION
    "11.4"
    CACHE STRING "CUDA VERSION")
find_package(CUDAToolkit ${CUDA_VERSION} EXACT REQUIRED)

# cutlass
include(cutlass)

# NOTE: flash-attention MUST be included before cudart
# flash-attention
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
include(flash-attention)
list(APPEND CUDA_3RD_PARTY_LIBS ${FLASHATTN_LIBRARY})
endif()

if(ENABLE_NV_STATIC_LIB)
  message("Using static lib of CUDAToolkit")
  set(AS_CUDA_CUDART CUDA::cudart_static)
  set(AS_CUDA_CUBLAS CUDA::cublas_static)
else()
  set(AS_CUDA_CUDART CUDA::cudart)
  set(AS_CUDA_CUBLAS CUDA::cublas)
endif()

# HIE-DNN
include(hie-dnn)
list(APPEND CUDA_3RD_PARTY_LIBS ${HIEDNN_LIBRARY})

# span-attention
include(span-attention)
list(APPEND CUDA_3RD_PARTY_LIBS ${SPANATTN_LIBRARY})

# notes for cuda static link, order matters, first cublas, cublas_lt, cuda runtime.
list(APPEND CUDA_3RD_PARTY_LIBS  ${AS_CUDA_CUBLAS})

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "10.1")
  if(ENABLE_NV_STATIC_LIB)
    set(HIE_CUDA_CUBLASLT CUDA::cublasLt_static)
  else()
    set(HIE_CUDA_CUBLASLT CUDA::cublasLt)
  endif()
  list(APPEND CUDA_3RD_PARTY_LIBS ${HIE_CUDA_CUBLASLT})
endif()

list(APPEND CUDA_3RD_PARTY_LIBS ${AS_CUDA_CUDART} )

if(ENABLE_SPARSE)
  if(ENABLE_NV_STATIC_LIB)
    set(AS_CUDA_CUSPARSE CUDA::cusparse_static)
  else()
    set(AS_CUDA_CUSPARSE CUDA::cusparse)
  endif()
  list(APPEND CUDA_3RD_PARTY_LIBS ${AS_CUDA_CUSPARSE})
endif()

# nccl and cudnn name already used by other depends, wordaround for static link
find_package(NCCL REQUIRED)

if(ENABLE_NV_STATIC_LIB)
  set(NCCL_LIBNAME "nccl_static")
else()
  set(NCCL_LIBNAME "nccl")
endif()
if (USE_SYSTEM_NV_LIB)
  list(APPEND CUDA_3RD_PARTY_LIBS CUDA::cuda_driver ${NCCL_LIBNAME})
else()
  list(APPEND CUDA_3RD_PARTY_LIBS CUDA::cuda_driver CUDA::${NCCL_LIBNAME})
endif()

include(nvtx)
list(APPEND CUDA_3RD_PARTY_LIBS nvtx3-cpp)

if (ENABLE_CUSPARSELT)
  message("Find cuSparseLt lib")
  find_package(CUSPARSELT)
  list(APPEND CUDA_3RD_PARTY_LIBS CUDA::${CUSPARSELT_LIBRARY_NAME})
endif()

# cuda nvml support
set(AS_CUDA_NVML CUDA::nvml)
list(APPEND CUDA_3RD_PARTY_LIBS  ${AS_CUDA_NVML})

message(
  STATUS
    "NVIDIA GPU dependencies: ${CUDAToolkit_INCLUDE_DIR}, ${CUDA_3RD_PARTY_LIBS}"
)
