cmake_minimum_required(VERSION 3.20)
set(project_version_in_env $ENV{AS_RELEASE_VERSION})
# remove -rc1 like string in version name.

if ((NOT DEFINED project_version_in_env))
    set(project_version_in_env "1.0.0")
endif()

string(REGEX REPLACE "-rc[0-9]+" "" STRIPED_VERSION_STRING ${project_version_in_env})
set(project_version_in_env ${STRIPED_VERSION_STRING})

message("Build AllSpark with version:${project_version_in_env}")

project(DashInfer LANGUAGES C CXX VERSION ${project_version_in_env})

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
        options are: Debug/Release/RelWithDebInfo/MinSizeRel." FORCE)
endif()

# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_VERBOSE_MAKEFILE ON)
message(STATUS "CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")

set(ALLSPARK_CBLAS "MKL"
    CACHE STRING "Blas library choice, MKL/BLIS/NONE")

set(RUNTIME_THREAD "OMP" CACHE STRING "cpu threading runtime")

set(CONFIG_HOST_CPU_TYPE "X86" CACHE STRING "host cpu type, like X86, ARMV9, etc.")

## x86 related option.
option(ENABLE_AVX2 "enable avx2" ON)

## ARM related option.
option(ENABLE_ARMCL "enable use of Arm Compute Library" OFF)
option(ENABLE_ARM_V84_V9 "enable v8.4-a or higher version of Arm instructions, such as sdot" OFF)

option(ENABLE_FP16 "build with fp16 mode" ON)
option(ENABLE_BF16 "build with bf16 mode" ON)
option(ENABLE_SPARSE "build with sparse mode" ON)
option(BUILD_PYTHON "build with python api" OFF)
option(BUILD_PACKAGE "build rpm or deb package" OFF)
option(MEM_CHECK "check memory" OFF)
option(LOCK_CHECK "check deadlock" OFF)
option(ENABLE_GLIBCXX11_ABI "build with cxx11 abi" ON)
#setting compiler flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
message(STATUS "CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS} ALLSPARK_CBLAS:${ALLSPARK_CBLAS}")

if(ENABLE_AVX2)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_AVX2")
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
    message("cxx11 abi is off")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DALLSPARK_DEBUG_MODE)
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

# thirdparth
set(INSTALL_LOCATION
      ${CMAKE_CURRENT_BINARY_DIR}/third_party/install/${CMAKE_BUILD_TYPE})

include(threading)
include(cpp-ipc)

if (ENABLE_ARMCL)
    include(armcl)
endif()

include(AutoCCache)
include(dnnl)
include(cblas)

if(MEM_CHECK)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
endif()

if(LOCK_CHECK)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fPIE>)
endif()

if (ENABLE_FP16)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_FP16")
endif()
if (ENABLE_BF16)
    list(APPEND ALLSPARK_DEFINITION "-DENABLE_BF16")
endif()

set(AS_MODEL_PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/proto/allspark.proto)

add_subdirectory(csrc)

if (BUILD_PYTHON)
    add_subdirectory(python)
endif()

set(FILE_PATH "/etc/os-release")
file(STRINGS ${FILE_PATH} ID_LINE REGEX "^ID=.*$")

if(ID_LINE)
    string(REGEX REPLACE "^ID=(.*)$" "\\1" OS_NAME ${ID_LINE})
    # ~S~G~S~^~\
    message(STATUS "The OS ID is: ${OS_NAME}")
else()
    message(FATAL "No ID line found in file.")
endif()

string (REGEX MATCH "ubuntu" IS_UBUNTU ${OS_NAME})


if (BUILD_PACKAGE)
  # config system arch
  if(CONFIG_HOST_CPU_TYPE STREQUAL "ARM")
    set(CPACK_SYSTEM_ARCHITECTURE "aarch64")
  else()
    set(CPACK_SYSTEM_ARCHITECTURE "x86_64")
  endif()

  set(CPACK_PACKAGE_DEVICE_NAME "cpu")
  set(CPACK_PACKAGE_VENDOR "Alibaba")
  set(CPACK_PACKAGE_NAME "DashInfer")
  set(CPACK_PACKAGE_VERSION ${project_version_in_env})
  set(CPACK_PACKAGE_VENDOR "Alibaba Tongyi")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "DashInfer AllSpark is a LLM inference engine.")
  set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  set(CPACK_PACKAGE_DEVICE_NAME "cpu")

  if(IS_UBUNTU)
    message("build deb package.")
    SET(CPACK_GENERATOR "DEB")
    set(CPACK_SYSTEM_NAME "ubuntu")
    SET(CPACK_THREADS 16)
    set(CPACK_SOURCE_GENERATOR "TGZ")
    set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /build
        /\\\\.DS_Store
        )
    SET(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_COMPONENTS_ALL libraries headers)
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alibaba Tongyi") #required
    INCLUDE(CPack)
  else()
    # only support centos like rpm system.
    # rpm related settings.
    set(CPACK_GENERATOR "RPM")
    set(CPACK_SYSTEM_NAME "centos")
    set(CPACK_RPM_PACKAGE_LICENSE "Apache2")
    set(CPACK_RPM_PACKAGE_GROUP "DashInfer")
    set(CPACK_RPM_COMPRESSION_TYPE "gzip")
    set(CPACK_RPM_PACKAGE_SUMMARY "DashInfer")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_RPM_PACKAGE_RELOCATABLE ON)
    INCLUDE(CPack)
  endif()

  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.${CPACK_PACKAGE_DEVICE_NAME}.${CPACK_SYSTEM_NAME}.${CPACK_SYSTEM_ARCHITECTURE}")
endif()

#install
include(cmake/install.cmake)
