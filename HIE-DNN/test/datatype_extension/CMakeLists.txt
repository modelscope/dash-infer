
set(DATATYPE_EXTENSION_UTEST_SRC datatype_extension_test.cpp)
set(DATATYPE_EXTENSION_UTEST_LIB gtest)

if(USE_FP16)
    set(FP16_UTEST_DIR half)

    set(DATATYPE_EXTENSION_UTEST_SRC
        ${DATATYPE_EXTENSION_UTEST_SRC}
        ${FP16_UTEST_DIR}/test_half.cpp)

    if(USE_CUDA)
        set(DATATYPE_EXTENSION_UTEST_SRC
            ${DATATYPE_EXTENSION_UTEST_SRC}
            ${FP16_UTEST_DIR}/test_half.cu)
    endif()
endif()

if(USE_BF16)
    set(BF16_UTEST_DIR bfloat16)

    set(DATATYPE_EXTENSION_UTEST_SRC
        ${DATATYPE_EXTENSION_UTEST_SRC}
        ${BF16_UTEST_DIR}/test_bfloat16.cpp)

    if(USE_CUDA)
        set(DATATYPE_EXTENSION_UTEST_SRC
            ${DATATYPE_EXTENSION_UTEST_SRC}
            ${BF16_UTEST_DIR}/test_bfloat16.cu)
    endif()
endif()

if(USE_CUDA)
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
    set(DATATYPE_EXTENSION_UTEST_LIB
        ${DATATYPE_EXTENSION_UTEST_LIB} Cudart::cudart)
endif()

add_executable(datatype_extension_utest ${DATATYPE_EXTENSION_UTEST_SRC})
target_link_libraries(datatype_extension_utest ${DATATYPE_EXTENSION_UTEST_LIB})

# make test
add_test(NAME datatype_extension_utest
         COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/datatype_extension_utest)
