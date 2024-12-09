option(AUTO_CCACHE "Use ccache to speed up rebuilds" ON)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND ${AUTO_CCACHE})
  message(STATUS "Using ${CCACHE_PROGRAM} as compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  # requires at least CMake 3.9 to be any use
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
