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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/02_cuda)

function(hiednn_cuda_example BIN SRC)
    add_executable(cuda_${BIN} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC})
    set_target_properties(cuda_${BIN} PROPERTIES OUTPUT_NAME ${BIN})
    target_link_libraries(cuda_${BIN} hiednn Cudart::cudart)
endfunction()

hiednn_cuda_example(01_unary_elementwise        01_unary_elementwise.cu)
hiednn_cuda_example(02_binary_elementwise       02_binary_elementwise.cu)
hiednn_cuda_example(03_cast                     03_cast.cu)
hiednn_cuda_example(04_expand                   04_expand.cu)
hiednn_cuda_example(05_set_tensor_value         05_set_tensor_value.cu)
hiednn_cuda_example(06_scatter_nd               06_scatter_nd.cu)
hiednn_cuda_example(07_prefix_sum               07_prefix_sum.cu)
hiednn_cuda_example(08_reduce                   08_reduce.cu)
hiednn_cuda_example(09_slice                    09_slice.cu)
hiednn_cuda_example(10_linear_interpolation     10_linear_interpolation.cu)
hiednn_cuda_example(11_nearest_interpolation    11_nearest_interpolation.cu)
hiednn_cuda_example(12_pad                      12_pad.cu)
hiednn_cuda_example(13_gather_elements          13_gather_elements.cu)
hiednn_cuda_example(14_trilu                    14_trilu.cu)
hiednn_cuda_example(15_where                    15_where.cu)
hiednn_cuda_example(16_scatter_elements         16_scatter_elements.cu)
hiednn_cuda_example(17_non_zero                 17_non_zero.cu)
hiednn_cuda_example(18_concat                   18_concat.cu)
