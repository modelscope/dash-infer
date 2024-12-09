if(CONFIG_HOST_CPU_TYPE STREQUAL "X86")
  set(INTEL_GEMM_PROJECT "extern_intel_gemm")
  include(FetchContent)
  set(INTEL_GEMM_URL ${CMAKE_SOURCE_DIR}/third_party/intel_gemm.tar.gz)
  message(STATUS "enter intel gemm build...")
  FetchContent_Declare(${INTEL_GEMM_PROJECT}
        URL ${INTEL_GEMM_URL}
  )
  FetchContent_MakeAvailable(${INTEL_GEMM_PROJECT})
  set(INTEL_GEMM_ROOT_DIR
    ${${INTEL_GEMM_PROJECT}_SOURCE_DIR}
    CACHE PATH "intel gemm library")
  set(INTEL_GEMM_ROOT ${INTEL_GEMM_ROOT_DIR})
  message(STATUS "INTEL_GEMM_ROOT_DIR: ${INTEL_GEMM_ROOT_DIR} INTEL_GEMM_ROOT:${INTEL_GEMM_ROOT}")
  set(INTEL_GEMM_STATIC_ONLY_LIB "ig_static")
  foreach (SO_NAME ${INTEL_GEMM_STATIC_ONLY_LIB})
    add_library(${SO_NAME} STATIC IMPORTED GLOBAL)
    set_target_properties(
                ${SO_NAME} PROPERTIES
                IMPORTED_LOCATION ${INTEL_GEMM_ROOT}/lib/lib${SO_NAME}.a
                INTERFACE_INCLUDE_DIRECTORIES
                ${INTEL_GEMM_ROOT}/include/
        )
  endforeach()
endif()
