cmake_minimum_required(VERSION 3.20)
set(project_version_in_env $ENV{AS_RELEASE_VERSION})
# remove -rc1 like string in version name.

if ((NOT DEFINED project_version_in_env))
    set(project_version_in_env "2.0.0")
endif()

string(REGEX REPLACE "-rc[0-9]+" "" STRIPED_VERSION_STRING ${project_version_in_env})
set(project_version_in_env ${STRIPED_VERSION_STRING})

message("Build DashInfer with version: ${project_version_in_env}")

project(DashInfer LANGUAGES C CXX VERSION ${project_version_in_env})

include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-${PROJECT_VERSION} CACHE STRING "Force modify install dir" FORCE)
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
if (BUILD_PYTHON)
    # building manylinux pkg need this setting to find local libflash-attn.so
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
        options are: Debug/Release/RelWithDebInfo/MinSizeRel." FORCE)
endif()

set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_VERBOSE_MAKEFILE ON)
message(STATUS "CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")

set(ALLSPARK_CBLAS "MKL"
    CACHE STRING "Blas library choice, MKL/BLIS/NONE")

set(RUNTIME_THREAD "OMP" CACHE STRING "cpu threading runtime") # VALUE : OMP/TBB
#option
option(ENABLE_CUDA "build with cuda support" ON)

set(CONFIG_ACCELERATOR_TYPE "CUDA" CACHE STRING  "compute accelerator type(like CUDA), NONE means cpu")
set(CONFIG_HOST_CPU_TYPE "X86" CACHE STRING "host cpu type, like X86, ARMV9, etc.")

## CUDA Related option.
##
option(ENABLE_NV_STATIC_LIB "build with static lib of nvidia" OFF)
option(USE_SYSTEM_NV_LIB "use system nccl lib instead download binary." OFF)
option(BUILD_HIEDNN "build HIE-DNN from source" OFF)
option(ENABLE_CUDA_PINNED_WEIGHT_LOAD "enable cuda pinned memory for load weight" OFF)
option(ENABLE_SPAN_ATTENTION "enable build with span attention" ON)
option(ENABLE_MULTINUMA "enable multinuma, if on cpu multinuma service will be compiled" OFF)
# set var let cmake module can get this var.
set(USE_SYSTEM_NV_LIB ${USE_SYSTEM_NV_LIB})
set(ENABLE_NV_STATIC_LIB ${ENABLE_NV_STATIC_LIB})

## x86 related option.
option(ENABLE_AVX2 "enable avx2" ON)
option(ENABLE_AVX512 "enable avx512" ON)

## ARM related option.
option(ENABLE_ARMCL "enable use of Arm Compute Library" OFF)
option(ENABLE_ARM_V84_V9 "enable v8.4-a or higher version of Arm instructions, such as sdot" OFF)

option(ENABLE_FP16 "build with fp16 mode" ON)
option(ENABLE_BF16 "build with bf16 mode" ON)
option(ENABLE_FP8 "build with fp8 mode" ON)
option(ENABLE_SPARSE "build with sparse mode" ON)
option(ENABLE_CUSPARSELT "build with CUSPARSELT lib" OFF)
option(BUILD_UTEST "build with unit test" ON)
option(BUILD_EXAMPLE "build with examples" ON)
option(BUILD_PYTHON "build with python api" ON)
option(BUILD_PACKAGE "build cpp package" OFF)
option(MEM_CHECK "check memory" OFF)
option(LOCK_CHECK "check deadlock" OFF)
option(ALWAYS_READ_LOAD_MODEL "load and parse model via every read" OFF)
option(ENABLE_GLIBCXX11_ABI "build with cxx11 abi" OFF)

# Enable JSON Mode by using LMFE lib
option(ENABLE_JSON_MODE "enable json mode support" ON)

#setting compiler flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
message(STATUS "CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS} ALLSPARK_CBLAS:${ALLSPARK_CBLAS}")

if(ENABLE_AVX2)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_AVX2")
endif()

if(ENABLE_AVX512)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_AVX512")
endif()

if(ENABLE_ARM_V84_V9)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_ARM_V84_V9")
    if (ENABLE_BF16)
        list(APPEND ALLSPARK_DEFINITION "-DENABLE_BF16")
    endif()
endif()

if(ENABLE_GLIBCXX11_ABI)
    if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0))
      message(FATAL_ERROR "gcc version < 5, ENABLE_GLIBCXX11_ABI should not be ON")
    endif()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    message("ENABLE_GLIBCXX11_ABI ${ENABLE_GLIBCXX11_ABI}")
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DALLSPARK_DEBUG_MODE)
    if(CMAKE_COMPILER_IS_GNUCXX)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:--coverage>)
        add_link_options(--coverage)
    endif()
#    add_definitions(-DENABLE_SPAN_DEBUG)  # will print span attention log.
#    add_definitions(-DCONFIG_MEM_DEBUG)  # will print memory allocation log.
#    add_definitions(-DCONFIG_OP_DEBUG)  # will print op forward, init, also sync after op forward.
#    add_definitions(-DCONFIG_LOCK_DEBUG)  # will print mutex lock unlock
else()
    add_definitions(-DNVTX_DISABLE)
endif()

include(cmake/CheckGit.cmake)
CheckGitSetup()
add_definitions(-DALLSPARK_VERSION_MAJOR="${PROJECT_VERSION_MAJOR}")
add_definitions(-DALLSPARK_VERSION_MINOR="${PROJECT_VERSION_MINOR}")
add_definitions(-DALLSPARK_VERSION_PATCH="${PROJECT_VERSION_PATCH}")

# 3rd dependency
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Protobuf REQUIRED)

# thirdparty
set(INSTALL_LOCATION
      ${CMAKE_CURRENT_BINARY_DIR}/third_party/install/${CMAKE_BUILD_TYPE})

include(threading)
include(cpp-ipc)
include(smhasher)

if (ENABLE_ARMCL)
    include(armcl)
endif()

include(AutoCCache)
include(dnnl)
include(cblas)

if (ENABLE_JSON_MODE)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_JSON_MODE")
    include(lmfe)
endif()

if(CONFIG_HOST_CPU_TYPE STREQUAL "X86")
    include(intelgemm)
endif()


if(MEM_CHECK)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
endif()

if(LOCK_CHECK)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fPIE>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
    # add_link_options(-fsanitize=thread)
endif()

if (ENABLE_CUDA)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_CUDA")
        if (ENABLE_CUSPARSELT)
        list(APPEND ALLSPARK_DEFINITION "-DENABLE_CUSPARSELT")
        endif()
    include(cuda)
    if (ENABLE_SPARSE)
        list(APPEND ALLSPARK_DEFINITION "-DENABLE_SPARSE")
    endif()
    if (ENABLE_FP8)
            list(APPEND ALLSPARK_DEFINITION "-DENABLE_FP8")
    endif()
endif()

if (ENABLE_MULTINUMA)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_MULTINUMA")
endif()

if (ENABLE_FP16)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_FP16")
endif()
if (ENABLE_BF16)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_BF16")
endif()

if (ALWAYS_READ_LOAD_MODEL)
    list(APPEND ALLSPARK_DEFINITION "-DALWAYS_READ_LOAD_MODEL")
endif()

add_subdirectory(csrc)

if (BUILD_UTEST)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

if (BUILD_PYTHON)
    add_subdirectory(python)
endif()

if (BUILD_PACKAGE)
    # config system arch
    if(CONFIG_HOST_CPU_TYPE STREQUAL "ARM")
        set(CPACK_SYSTEM_ARCHITECTURE "aarch64")
    else()
        set(CPACK_SYSTEM_ARCHITECTURE "x86_64")
    endif()

    if (ENABLE_CUDA)
        if(ENABLE_NV_STATIC_LIB)
            set(CPACK_PACKAGE_DEVICE_NAME "cuda-${CUDA_VERSION}-static")
        else()
            set(CPACK_PACKAGE_DEVICE_NAME "cuda-${CUDA_VERSION}-shared")
        endif()
    else()
        if (ENABLE_MULTINUMA)
            set(CPACK_PACKAGE_DEVICE_NAME "cpu-multinuma")
        else()
            set(CPACK_PACKAGE_DEVICE_NAME "cpu")
        endif()
    endif()

    set(CPACK_PACKAGE_NAME "DashInfer")
    set(CPACK_PACKAGE_VENDOR "Alibaba Tongyi")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "DashInfer AllSpark is a LLM inference engine.")
    set(CPACK_PACKAGE_VERSION ${project_version_in_env})
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_GENERATOR "TGZ")
    set(CPACK_THREADS 16)

    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.${CPACK_PACKAGE_DEVICE_NAME}.${CPACK_SYSTEM_ARCHITECTURE}")

    INCLUDE(CPack)
endif()

#install
include(cmake/install.cmake)
