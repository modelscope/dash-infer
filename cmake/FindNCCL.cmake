message("======= FindNCCL")

if (USE_SYSTEM_NV_LIB)
  message("Bypass download nccl, use system provided nccl.")
  return()
endif()
include(FindPackageHandleStandardArgs)
include(FetchContent)
set(NCCL_VERSION
    "2.11.4"
    CACHE STRING "NCCL VERSION")
set(NCCL_URL https://github.com/NVIDIA/nccl/archive/refs/tags/v${NCCL_VERSION}-1.tar.gz)
set(NCCL_PROJECT "extern_nccl")
FetchContent_Declare(${NCCL_PROJECT} URL ${NCCL_URL})
message(STATUS "Fetch NCCL from ${NCCL_URL}")
FetchContent_MakeAvailable(${NCCL_PROJECT})

set(NCCL_ROOT_DIR
    "${${NCCL_PROJECT}_SOURCE_DIR}"
    CACHE PATH "NVIDIA NCCL")
message(STATUS "NCCL_ROOT_DIR : ${NCCL_ROOT_DIR}")

find_path(
  NCCL_INCLUDE_DIR nccl.h
  HINTS ${NCCL_ROOT_DIR}
  PATH_SUFFIXES cuda/include include
                nccl-${NCCL_VERSION}-cuda-${CUDA_VERSION}/include)

if(ENABLE_NV_STATIC_LIB)
  set(NCCL_LIBNAME "nccl_static")
else()
  set(NCCL_LIBNAME "nccl")
endif()

message("find nccl with ${NCCL_LIBNAME}")
find_library(
  AS_NCCL_LIBRARY ${NCCL_LIBNAME}
  HINTS ${NCCL_ROOT_DIR}
  PATH_SUFFIXES lib lib64 nccl-${NCCL_VERSION}-cuda-${CUDA_VERSION}/lib64)

if(ENABLE_NV_STATIC_LIB)
  message("add nccl static lib")
  add_library(CUDA::${NCCL_LIBNAME} STATIC IMPORTED GLOBAL)
else()

  message("add nccl shared lib")
  add_library(CUDA::${NCCL_LIBNAME} SHARED IMPORTED GLOBAL)
endif()
set_property(TARGET CUDA::${NCCL_LIBNAME} PROPERTY IMPORTED_LOCATION ${AS_NCCL_LIBRARY})
set_property(TARGET CUDA::${NCCL_LIBNAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                  ${NCCL_INCLUDE_DIR})

# install nccl

if(NOT ENABLE_NV_STATIC_LIB)
get_filename_component(NCCL_LIB_DIR ${AS_NCCL_LIBRARY} DIRECTORY)
install(DIRECTORY ${NCCL_LIB_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        USE_SOURCE_PERMISSIONS FILES_MATCHING
        PATTERN "*nccl.so*"
)
endif()


find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR
                                  AS_NCCL_LIBRARY)

if(NCCL_FOUND)
  message(STATUS "Found NCCL: success , library path : ${AS_NCCL_LIBRARY}")
endif()
