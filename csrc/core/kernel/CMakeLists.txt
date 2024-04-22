set(KERNEL_INC_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${THREAD_INCLUDE}
    ${PROJECT_SOURCE_DIR}/csrc
    ${PROJECT_SOURCE_DIR}/csrc/common
    ${PROJECT_SOURCE_DIR}/csrc/device
    ${CMAKE_BINARY_DIR}/csrc
)

if(NOT ALLSPARK_CBLAS MATCHES "NONE")
  list(APPEND KERNEL_INC_DIR ${CBLAS_INCLUDE_DIR})
  list(APPEND KERNEL_3RD_LIBS ${CBLAS_LIBRARY})
endif()

# check if cross compiling
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
  list(APPEND ALLSPARK_PUBLIC_DEFINITIONS -DALLSPARK_USE_NEON_)
  list(APPEND ALLSPARK_DEFINITION "-DENABLE_ARM")
  if(ENABLE_ARM_V84_V9)
    if(${CMAKE_CXX_COMPILER} MATCHES "clang") # arm compiler
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2a+dotprod+fp16+bf16+i8mm+sve+sve2")
    else()  # gcc
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2-a+dotprod+fp16+bf16+i8mm+sve+sve2")
    endif()
  else()
    if(${CMAKE_CXX_COMPILER} MATCHES "clang") # arm compiler
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2a+dotprod+fp16+i8mm")
    else()  # gcc
        list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2-a+dotprod+fp16+i8mm")
    endif()
  endif()
endif() # AARCH64

file(GLOB_RECURSE src_cpu_common cpu/*.cpp)

file(
  GLOB_RECURSE
  src_avx2
  cpu/layernorm.cpp
  cpu/mha.cpp
  cpu/rotary.cpp
)

file(
  GLOB_RECURSE
  src_arm
  cpu/gemm_lowp/arm/*
)

if(ENABLE_AVX2)
    set(AVX2_FLAGS "-mavx2 -mfma")
    message("AVX2 flags: ${AVX2_FLAGS}, files: ${src_avx2}")
    set_source_files_properties(${src_avx2} PROPERTIES COMPILE_FLAGS "${AVX2_FLAGS}")
endif(ENABLE_AVX2)

if(NOT ENABLE_ARM_V84_V9)
    foreach(file ${src_arm})
        list(REMOVE_ITEM src_cpu_common "${file}")
    endforeach()
endif(NOT ENABLE_ARM_V84_V9)

list(APPEND KERNEL_SRC ${src_cpu_common})

message(STATUS "allspark kernel deps: ${AS_MODEL_HDR} ${AS_MODEL_CPP}")

list(APPEND KERNEL_SRC ${AS_MODEL_HDR} ${AS_MODEL_CPP})

message("kernel cpp list: ${KERNEL_SRC}")

list(APPEND KERNEL_3RD_LIBS ${THREAD_LIB} ${CBLAS_LIBRARY})

list(APPEND KERNEL_3RD_LIBS CONAN_PKG::openmpi)

set(ALLSPARK_PUBLIC_DEFINITIONS ${ALLSPARK_PUBLIC_DEFINITIONS})

message(STATUS "ALLSPARK_PUBLIC_DEFINITIONS:${ALLSPARK_PUBLIC_DEFINITIONS}")

add_library(allspark_kernel STATIC ${KERNEL_SRC})

set_target_properties(allspark_kernel PROPERTIES CXX_STANDARD ${CXX_STD})
add_dependencies(allspark_kernel as_model_generate)

target_compile_definitions(
    allspark_kernel
    PRIVATE ${ALLSPARK_PUBLIC_DEFINITIONS}
    PUBLIC ${ALLSPARK_PUBLIC_DEFINITIONS})

target_include_directories(allspark_kernel PUBLIC ${KERNEL_INC_DIR} ${CMAKE_BINARY_DIR}/csrc)

target_link_libraries(allspark_kernel PRIVATE ${KERNEL_3RD_LIBS} CONAN_PKG::glog CONAN_PKG::protobuf)
target_compile_definitions(allspark_kernel PUBLIC ${ALLSPARK_DEFINITION})

target_compile_options(allspark_kernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${ALLSPARK_CXX_FLAGS}>)
if(ALLSPARK_CBLAS MATCHES "BLIS")
    add_dependencies(allspark_kernel project_blis)
endif()
