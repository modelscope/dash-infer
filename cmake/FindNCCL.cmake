message("======= FindNCCL")

if (USE_SYSTEM_NV_LIB)
  message("Bypass download nccl, use system provided nccl.")
  return()
endif()
include(FindPackageHandleStandardArgs)

find_path(
  NCCL_INCLUDE_DIR nccl.h
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
file(GLOB NCCL_LIBS ${NCCL_LIB_DIR}/*nccl.so*)
install(FILES ${NCCL_LIBS}
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()


find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR
                                  AS_NCCL_LIBRARY)

if(NCCL_FOUND)
  message(STATUS "Found NCCL: success, library path : ${AS_NCCL_LIBRARY}")
endif()
