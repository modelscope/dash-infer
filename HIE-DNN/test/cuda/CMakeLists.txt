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

# debug mode
if(ENABLE_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# cuda utest source file
set(CUDA_UTEST_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDA_UTEST_SRC
    ${CUDA_UTEST_SRC_DIR}/main.cu
    ${CUDA_UTEST_SRC_DIR}/type_conversion_utest.cu
    ${CUDA_UTEST_SRC_DIR}/cast_utest.cu
    ${CUDA_UTEST_SRC_DIR}/set_tensor_utest.cu
    ${CUDA_UTEST_SRC_DIR}/unary_elementwise_utest.cu
    ${CUDA_UTEST_SRC_DIR}/binary_elementwise_utest.cu
    ${CUDA_UTEST_SRC_DIR}/expand_utest.cu
    ${CUDA_UTEST_SRC_DIR}/scatter_nd_utest.cu
    ${CUDA_UTEST_SRC_DIR}/linear_interpolation_utest.cu
    ${CUDA_UTEST_SRC_DIR}/nearest_interpolation_utest.cu
    ${CUDA_UTEST_SRC_DIR}/cubic_interpolation_utest.cu
    ${CUDA_UTEST_SRC_DIR}/reduce_utest.cu
    ${CUDA_UTEST_SRC_DIR}/prefix_sum_utest.cu
    ${CUDA_UTEST_SRC_DIR}/slice_utest.cu
    ${CUDA_UTEST_SRC_DIR}/pad_utest.cu
    ${CUDA_UTEST_SRC_DIR}/gather_elements_utest.cu
    ${CUDA_UTEST_SRC_DIR}/trilu_utest.cu
    ${CUDA_UTEST_SRC_DIR}/where_utest.cu
    ${CUDA_UTEST_SRC_DIR}/scatter_elements_utest.cu
    ${CUDA_UTEST_SRC_DIR}/non_zero_utest.cu
    ${CUDA_UTEST_SRC_DIR}/concat_utest.cu)

add_executable(cuda_utest ${CUDA_UTEST_SRC})
target_link_libraries(cuda_utest hiednn gtest Cudart::cudart)

# make test
add_test(NAME cuda_utest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cuda_utest)
