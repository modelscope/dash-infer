cmake_minimum_required(VERSION 3.15.0)

project(HIE-DNN LANGUAGES CXX)

option(USE_CPP "Enable CPP" ON)
option(USE_CUDA "Enable CUDA" ON)
option(USE_FP16 "Enable FP16" ON)
option(USE_BF16 "Enable BF16" ON)

option(ENABLE_DEBUG "Enable Debug" OFF)

option(UTEST "Enable Unit Test" ON)
option(EXAMPLE "Compile Example" ON)

#------------ CUDA Backend Options -------------

# CUDA ISA option, valid options are:
# AUTO:     detect the GPU ISA version of current machine and only generate
#           code for this ISA version
# ALL:      generate code for all ISA versions supported by the current CUDA
#           toolkit
# SERVER:   generate code for all ISA versions supported by the current CUDA
#           toolkit, excpet Jetson device ISA
# ISA list: specify the target ISAs by ISA list, such as '60,61,70,75', will
#           generate code for sm_60, sm_61, sm_70, sm_75
set(CUDA_DEVICE_ARCH AUTO CACHE STRING "CUDA device gencode options")

# nvcc parallel compiling option
# `--threads 0` option of nvcc were enabled if CUDA_PARALLEL_COMPILE is ON,
# it will enable parallel compiling for multiple architectures of a kernel.
# This option only work for CUDA version 11.2 or above, it will be ignored
# for other CUDA versions
option(CUDA_PARALLEL_COMPILE "Compile kernels parallel" ON)

# CUDA fast math option
# --use_fast_math option of nvcc were enabled if CUDA_FAST_MATH is ON, this
# option enable some fast approximate math functions, such as floating point
# division, trigonometric functions.
#
# ATTENTION: for some operators which are math function performance sensitive,
# such as interpolation, fast approximate math functions are also used even if
# CUDA_FAST_MATH is OFF
option(CUDA_FAST_MATH "Enable CUDA Fast Math" OFF)

#--------- End of CUDA backend options ---------

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/dnn/include)

set(CMAKE_CXX_STANDARD 11)

# disable CMP0104 warning for cmake 3.18
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    cmake_policy(SET CMP0104 OLD)
endif()

# enable warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

if(ENABLE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

if(USE_FP16)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/half_2.2.0/include)
    add_definitions(-DHIEDNN_USE_FP16)
endif()

if(USE_BF16)
    add_definitions(-DHIEDNN_USE_BF16)
endif()

# executable binary path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# hiednn basic source file
set(HIEDNN_SRC_DIR ${PROJECT_SOURCE_DIR}/dnn)
set(HIEDNN_SRC
    ${HIEDNN_SRC_DIR}/tensor_desc.cpp)

set(HIEDNN_BACKEND)

# hiednn cpp source path
if(USE_CPP)
    add_definitions(-DHIEDNN_USE_CPP)
    add_subdirectory(dnn/cpp)
    set(HIEDNN_BACKEND ${HIEDNN_BACKEND} hiednn_cpp)
endif()

# hiednn cuda source path
if(USE_CUDA)
    add_definitions(-DHIEDNN_USE_CUDA)
    add_subdirectory(dnn/cuda)
    set(HIEDNN_BACKEND ${HIEDNN_BACKEND} hiednn_cuda)
endif()

add_library(hiednn SHARED ${HIEDNN_SRC})
add_library(hiednn_static STATIC ${HIEDNN_SRC})
set_property(TARGET hiednn_static PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(hiednn ${HIEDNN_BACKEND})
target_link_libraries(hiednn_static ${HIEDNN_BACKEND})

target_include_directories(hiednn PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(hiednn_static PUBLIC ${PROJECT_SOURCE_DIR}/include)

# unit test
if(UTEST)
    enable_testing()
    add_subdirectory(test)
endif()

# example
if(EXAMPLE)
    add_subdirectory(example)
endif()

# doxygen
include("cmake/Doxygen.cmake")

include(GNUInstallDirs)
# install libs
install(TARGETS hiednn hiednn_static
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
# install header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# cpplint for static code check
include(CppLint)
list(APPEND CPPLINT_SCAN_DIR ${CMAKE_SOURCE_DIR}/dnn)
list(APPEND CPPLINT_SCAN_DIR ${CMAKE_SOURCE_DIR}/test)
list(APPEND CPPLINT_SCAN_DIR ${CMAKE_SOURCE_DIR}/example)
cpplint_exclude(${CMAKE_SOURCE_DIR}/dnn/include/datatype_extension/half)
cpplint_exclude(${CMAKE_SOURCE_DIR}/dnn/include/datatype_extension/bfloat16)
add_cpplint_target(cpplint "${CPPLINT_SCAN_DIR}")

