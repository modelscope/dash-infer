if (USE_SYSTEM_NV_LIB)
  message("Bypass download cudnn, use system provided cudnn.")
  return()
endif()

include(FindPackageHandleStandardArgs)
include(FetchContent)
set(OSS_LOCATION
    "http://ait-public.oss-cn-hangzhou-zmf.aliyuncs.com/hci_team/HIE/hie_third_party"
)
set(CUDNN_VERSION
    "8.2.2"
    CACHE STRING "CUDNN VERSION")
set(CUDNN_URL
    ${OSS_LOCATION}/x86_64/linux/cudnn-${CUDNN_VERSION}-cuda-${CUDA_VERSION}.tar.gz
)
set(CUDNN_PROJECT "extern_cudnn")
FetchContent_Declare(${CUDNN_PROJECT} URL ${CUDNN_URL})
message(STATUS "Fetch CUDNN from ${CUDNN_URL}")
FetchContent_MakeAvailable(${CUDNN_PROJECT})

set(CUDNN_ROOT_DIR
    "${${CUDNN_PROJECT}_SOURCE_DIR}"
    CACHE PATH "NVIDIA cuDNN")
message(STATUS "CUDNN_ROOT_DIR : ${CUDNN_ROOT_DIR}")

find_path(
  CUDNN_INCLUDE_DIR cudnn.h
  HINTS ${CUDNN_ROOT_DIR}
  PATH_SUFFIXES cuda/include include
                cudnn-${CUDNN_VERSION}-cuda-${CUDA_VERSION}/include)

if(ENABLE_NV_STATIC_LIB)
  set(CUDNN_LIBNAME "cudnn_static")
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  message("set suffix ${CMAKE_FIND_LIBRARY_SUFFIXES}")
else()
  set(CUDNN_LIBNAME "cudnn")
endif()


message("===================== cudnn name ${CUDNN_LIBNAME}")
find_library(
  AS_CUDNN_LIBRARY ${CUDNN_LIBNAME} HINTS ${CUDNN_ROOT_DIR}
  PATH_SUFFIXES lib lib64 cudnn-${CUDNN_VERSION}-cuda-${CUDA_VERSION}/lib64)

if(ENABLE_NV_STATIC_LIB)
  add_library(as_cudnn STATIC IMPORTED GLOBAL)
else()
  add_library(as_cudnn SHARED IMPORTED GLOBAL)
endif()
set_property(TARGET as_cudnn PROPERTY IMPORTED_LOCATION ${AS_CUDNN_LIBRARY})
set_property(TARGET as_cudnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                   ${CUDNN_INCLUDE_DIR})

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR
                                  AS_CUDNN_LIBRARY)

if(CUDNN_FOUND)
  message(STATUS "Found CUDNN: success , library path : ${AS_CUDNN_LIBRARY}")
endif()
