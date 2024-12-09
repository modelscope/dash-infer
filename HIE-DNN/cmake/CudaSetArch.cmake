
function(detect_cuda_arch CUDA_ARCH_LIST)
    if(CMAKE_CUDA_COMPILER)
        set(DEVICE_DETECT_CU "${PROJECT_BINARY_DIR}/detect_cuda_device.cu")
        file(WRITE ${DEVICE_DETECT_CU} ""
             "#include <stdio.h> \n"
             "int main() { \n"
             "   int device_count = 0; \n"
             "   if (cudaGetDeviceCount(&device_count) != cudaSuccess) { \n"
             "       return -1; \n"
             "   } \n"
             "   if (device_count == 0) { \n"
             "       return -2; \n"
             "   } \n"
             "   for (int device_id = 0; device_id < device_count; ++device_id) { \n"
             "       cudaDeviceProp prop; \n"
             "       if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) { \n"
             "           printf(\"%d%d \", prop.major, prop.minor); \n"
             "       } else { \n"
             "           return -1; \n"
             "       } \n"
             "   } \n"
             "   return 0; \n"
             "} \n")

        try_run(MAIN_RET COMPILE_RET ${PROJECT_BINARY_DIR} ${DEVICE_DETECT_CU}
                RUN_OUTPUT_VARIABLE DETECT_ARCH_LIST)

        if(COMPILE_RET STREQUAL "FALSE")
            message(FATAL_ERROR "Auto CUDA Arch: compile cuda device detection code failed")
        endif()

        if(MAIN_RET EQUAL -1)
            message(FATAL_ERROR "Auto CUDA Arch: cuda runtime error")
        elseif(MAIN_RET EQUAL -2)
            message(FATAL_ERROR "Auto CUDA Arch: no CUDA capable device was found")
        endif()

        string(REGEX MATCHALL "[0-9]+" DETECT_ARCH_LIST "${DETECT_ARCH_LIST}")
        separate_arguments(DETECT_ARCH_LIST)
        list(REMOVE_DUPLICATES DETECT_ARCH_LIST)
        set(${CUDA_ARCH_LIST} ${DETECT_ARCH_LIST} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Auto CUDA Arch: CUDA not enabled in cmake")
    endif()
endfunction()

# ARCH: AUTO, ALL, SERVER, or compute capability list (62;70;75 ...)
function(set_cuda_arch)
    # CUDA version must >= 8.0
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "8.0")
        message(FATAL_ERROR "CUDA version less than 8.0")
    endif()

    # -----------------------------------------------
    # set CUDA_CC list based on CUDA_VEERSION
    # -----------------------------------------------
    # Maxwell & Pascal
    set(CUDA_SERVER_CC "50" "52" "60" "61")
    set(CUDA_JETSON_CC "53" "62")

    # Volta
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.0")
        list(APPEND CUDA_SERVER_CC "70")
        list(APPEND CUDA_JETSON_CC "72")
    endif()

    # Turing
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.0")
        list(APPEND CUDA_SERVER_CC "75")
    endif()

    # Ampere
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.0")
        list(APPEND CUDA_SERVER_CC "80")
        # cc50 is deprecated in CUDA11
        list(REMOVE_ITEM CUDA_SERVER_CC "50")
    endif()
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.1")
        list(APPEND CUDA_SERVER_CC "86")
    endif()
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.4")
        list(APPEND CUDA_JETSON_CC "87")
    endif()
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.8")
        list(APPEND CUDA_SERVER_CC "89" "90")
    endif()

    # -----------------------------------------------
    # parse arguments
    # -----------------------------------------------
    if("${ARGN}" STREQUAL "ALL")
        set(CUDA_BUILD_CC ${CUDA_SERVER_CC} ${CUDA_JETSON_CC})
    elseif("${ARGN}" STREQUAL "SERVER")
        set(CUDA_BUILD_CC ${CUDA_SERVER_CC})
    elseif("${ARGN}" STREQUAL "AUTO")
        detect_cuda_arch(CUDA_BUILD_CC)
        message(STATUS "Autodetected CUDA architecture(s): ${CUDA_BUILD_CC}")
    else()
        set(CUDA_BUILD_CC "${ARGN}")
    endif()

    string(REGEX REPLACE "[ \t]+" ";" CUDA_BUILD_CC "${CUDA_BUILD_CC}")
    list(REMOVE_DUPLICATES CUDA_BUILD_CC)
    list(SORT CUDA_BUILD_CC)

    # CC for binary
    foreach(BIN_CC ${CUDA_BUILD_CC})
        set(NVCC_ARCH_FLAGS
            "${NVCC_ARCH_FLAGS} -gencode arch=compute_${BIN_CC},code=sm_${BIN_CC}")
        set(PTX_CC ${BIN_CC})
    endforeach()

    # CC for PTX
    set(NVCC_ARCH_FLAGS
        "${NVCC_ARCH_FLAGS} -gencode arch=compute_${PTX_CC},code=compute_${PTX_CC}")

    message(STATUS "CUDA gencode flags: " ${NVCC_ARCH_FLAGS})
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} ${NVCC_ARCH_FLAGS} PARENT_SCOPE)
endfunction()


