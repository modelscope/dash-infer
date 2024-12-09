if(CONFIG_HOST_CPU_TYPE STREQUAL "X86")
  set(INTEL_OMP_PROJECT "extern_intel_omp")
  include(FetchContent)
  set(INTEL_OMP_URL ${CMAKE_SOURCE_DIR}/third_party/intel_omp.tar.gz)
  message(STATUS "enter intel omp build...")
  FetchContent_Declare(${INTEL_OMP_PROJECT}
        URL ${INTEL_OMP_URL}
  )
  FetchContent_MakeAvailable(${INTEL_OMP_PROJECT})
  set(INTEL_OMP_ROOT_DIR
    ${${INTEL_OMP_PROJECT}_SOURCE_DIR}
    CACHE PATH "intel omp library")
  set(INTEL_OMP_ROOT ${INTEL_OMP_ROOT_DIR})
  message(STATUS "INTEL_OMP_ROOT_DIR: ${INTEL_OMP_ROOT_DIR} INTEL_OMP_ROOT:${INTEL_OMP_ROOT}")
  set(INTEL_OMP_STATIC_ONLY_LIB "iomp5")
  foreach (SO_NAME ${INTEL_OMP_STATIC_ONLY_LIB})
    add_library(${SO_NAME} STATIC IMPORTED GLOBAL)
    set_target_properties(
                ${SO_NAME} PROPERTIES
                IMPORTED_LOCATION ${INTEL_OMP_ROOT}/lib/lib${SO_NAME}.a
                INTERFACE_INCLUDE_DIRECTORIES
                ${INTEL_OMP_ROOT}/include/
        )
  endforeach()
endif()
