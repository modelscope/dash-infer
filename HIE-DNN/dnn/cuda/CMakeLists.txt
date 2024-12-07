enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)

# clear CMake CUDA options inherited from parent project
# HIE-DNN uses its own compiling options
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_FLAGS_DEBUG "")
set(CMAKE_CUDA_ARCHITECTURES "")

find_package(Cudart)

include(CudaSetArch)
set_cuda_arch(${CUDA_DEVICE_ARCH})

# warning for register spill
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-warn-lmem-usage")

# parallel compiling option
if(CUDA_PARALLEL_COMPILE AND ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.2")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 0")
endif()

# fast math option
if(CUDA_FAST_MATH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
endif()

# debug mode
if(ENABLE_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# hiednn cuda source file
set(HIEDNN_CUDA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HIEDNN_CUDA_SRC
    ${HIEDNN_CUDA_SRC_DIR}/hiednn_cuda.cu
    ${HIEDNN_CUDA_SRC_DIR}/cast.cu
    ${HIEDNN_CUDA_SRC_DIR}/set_tensor_value.cu
    ${HIEDNN_CUDA_SRC_DIR}/unary_elementwise.cu
    ${HIEDNN_CUDA_SRC_DIR}/binary_elementwise.cu
    ${HIEDNN_CUDA_SRC_DIR}/expand.cu
    ${HIEDNN_CUDA_SRC_DIR}/scatter_nd.cu
    ${HIEDNN_CUDA_SRC_DIR}/prefix_scan/prefix_scan.cu
    ${HIEDNN_CUDA_SRC_DIR}/interpolation/linear_interpolation.cu
    ${HIEDNN_CUDA_SRC_DIR}/interpolation/nearest_interpolation.cu
    ${HIEDNN_CUDA_SRC_DIR}/interpolation/cubic_interpolation.cu
    ${HIEDNN_CUDA_SRC_DIR}/reduce/reduce.cu
    ${HIEDNN_CUDA_SRC_DIR}/reduce/reduce_index.cu
    ${HIEDNN_CUDA_SRC_DIR}/slice.cu
    ${HIEDNN_CUDA_SRC_DIR}/pad/pad.cu
    ${HIEDNN_CUDA_SRC_DIR}/gather_elements.cu
    ${HIEDNN_CUDA_SRC_DIR}/trilu.cu
    ${HIEDNN_CUDA_SRC_DIR}/where.cu
    ${HIEDNN_CUDA_SRC_DIR}/scatter_elements.cu
    ${HIEDNN_CUDA_SRC_DIR}/non_zero.cu
    ${HIEDNN_CUDA_SRC_DIR}/concat.cu)

add_library(hiednn_cuda OBJECT ${HIEDNN_CUDA_SRC})
set_property(TARGET hiednn_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(
    hiednn_cuda
    Cudart::cudart)
