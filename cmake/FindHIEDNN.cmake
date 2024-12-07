message("========== HIE-DNN ==========")
set(USE_FP16
    ${ENABLE_FP16}
    CACHE BOOL "")
set(USE_BF16
    ${ENABLE_BF16}
    CACHE BOOL "")
set(CUDA_DEVICE_ARCH
    ${CMAKE_CUDA_ARCHITECTURES}
    CACHE STRING "")

if(ENABLE_CUDA)
  set(USE_CUDA
      ON
      CACHE BOOL "")
else()
  set(USE_CUDA
      OFF
      CACHE BOOL "")
endif()

# disable hie-dnn utest & examples
set(UTEST
    OFF
    CACHE BOOL "")
set(EXAMPLE
    OFF
    CACHE BOOL "")

message(STATUS "\tBuild HIE-DNN with: CUDA_DEVICE_ARCH=${CUDA_DEVICE_ARCH}")
message(STATUS "\tBuild HIE-DNN with: USE_FP16=${USE_FP16}")
message(STATUS "\tBuild HIE-DNN with: USE_BF16=${USE_BF16}")
message(STATUS "\tBuild HIE-DNN with: USE_CUDA=${USE_CUDA}")
set(HIEDNN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/HIE-DNN)
add_subdirectory(${HIEDNN_SOURCE_DIR} EXCLUDE_FROM_ALL)
set_target_properties(hiednn PROPERTIES FOLDER "External/HIE-DNN")
set_target_properties(hiednn_static PROPERTIES FOLDER "External/HIE-DNN")
unset(CUDA_DEVICE_ARCH CACHE)
unset(USE_FP16 CACHE)
unset(USE_BF16 CACHE)
unset(USE_CUDA CACHE)
unset(UTEST CACHE)
unset(EXAMPLE CACHE)
message(STATUS "Build HIE-DNN in: ${HIEDNN_SOURCE_DIR}")

message("=============================")
