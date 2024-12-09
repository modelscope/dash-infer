cmake_minimum_required(VERSION 3.21)

# TODO: version
project(spanattn LANGUAGES C CXX CUDA)

set(SPANATTN_LIB_SHARED ${CMAKE_PROJECT_NAME}_shared)
set(SPANATTN_LIB_STATIC ${CMAKE_PROJECT_NAME}_static)

option(SPANATTN_STATIC_CUDART "Link to static cudart library" OFF)
option(SPANATTN_EXTERNAL_CUTLASS "Build with external CUTLASS" OFF)
option(SPANATTN_ENABLE_FP16 "Build with float16 support" ON)
option(SPANATTN_ENABLE_BF16 "Build with bfloat16 support" ON)
option(SPANATTN_ENABLE_TEST "Build tests" ON)
set(SPANATTN_CUDA_ARCHS "75;80;90a" CACHE STRING "Target CUDA architectures")

set(CMAKE_VERBOSE_MAKEFILE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
        options are: Debug/Release/RelWithDebInfo/MinSizeRel." FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_COMPILER_IS_GNUCXX)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--coverage>)
        add_link_options(--coverage)
    endif()
endif()

set(SPANATTN_INTERFACE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include/spanattn
)

# cpp
set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

if(SPANATTN_ENABLE_FP16)
    message(STATUS "spanattn enable FP16")
    list(APPEND SPANATTN_DEFINITION "-DENABLE_FP16")
endif()
if(SPANATTN_ENABLE_BF16)
    message(STATUS "spanattn enable BF16")
    list(APPEND SPANATTN_DEFINITION "-DENABLE_BF16")
endif()

# cuda
set(CUDA_VERSION "12.0" CACHE STRING "CUDA VERSION")
find_package(CUDAToolkit ${CUDA_VERSION} REQUIRED)

# config cuda
message(STATUS "SPANATTN_CUDA_ARCHS: ${SPANATTN_CUDA_ARCHS}")
set(CMAKE_CUDA_ARCHITECTURES ${SPANATTN_CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --generate-line-info")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xptxas -warn-spills")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xptxas -warn-lmem-usage")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wextra -Xcompiler -Wno-unused-parameter")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror=return-type")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 8")

message(STATUS "SPANATTN_STATIC_CUDART: ${SPANATTN_STATIC_CUDART}")
if(SPANATTN_STATIC_CUDART)
  set(SPANATTN_CUDART_LIBRARY CUDA::cudart_static)
else()
  set(SPANATTN_CUDART_LIBRARY CUDA::cudart)
endif()

# third party

# CUTLASS
message(STATUS "SPANATTN_EXTERNAL_CUTLASS: ${SPANATTN_EXTERNAL_CUTLASS}")
if(SPANATTN_EXTERNAL_CUTLASS)
    message(STATUS "spanattn use external cutlass")
    find_package(NvidiaCutlass PATHS ${CUTLASS_INSTALL_PATH})
    set(CUTLASS_INCLUDE_DIR ${CUTLASS_INSTALL_PATH}/include)
else()
    set(SPANATTN_CUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cutlass)
    set(CUTLASS_INCLUDE_DIR ${SPANATTN_CUTLASS_PATH}/include)
endif()
message(STATUS "CUTLASS_INCLUDE_DIR: ${CUTLASS_INCLUDE_DIR}")
set(CUTLASS_LIBRARY spanattn_cutlass)
add_library(${CUTLASS_LIBRARY} INTERFACE)
target_include_directories(${CUTLASS_LIBRARY} INTERFACE
    ${CUTLASS_INCLUDE_DIR}
)

# GTest
if(SPANATTN_ENABLE_TEST)
    set(BUILD_GMOCK OFF CACHE BOOL "Enable building gmock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Install gtest" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest)
endif()

# sources
add_subdirectory(src)

# tests
if(SPANATTN_ENABLE_TEST)
    message(STATUS "spanattn enable test")
    add_subdirectory(test)
endif()

# install
add_library(spanattn::spanattn ALIAS ${SPANATTN_LIB_SHARED})
add_library(spanattn::spanattn_static ALIAS ${SPANATTN_LIB_STATIC})
install(DIRECTORY ${SPANATTN_INTERFACE_DIR} TYPE INCLUDE)
install(TARGETS ${SPANATTN_LIB_SHARED} ${SPANATTN_LIB_STATIC}
    EXPORT spanattn
)
