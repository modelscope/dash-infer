configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_common.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/test_common.h)
add_library(test_as_common test_common.cpp)
target_link_libraries(test_as_common allspark_framework CONAN_PKG::gtest CONAN_PKG::glog CONAN_PKG::zlib ${ONEDNN_LIBRARY} ${CUDA_3RD_PARTY_LIBS} ${THREAD_LIB})
set_target_properties(test_as_common PROPERTIES CXX_STANDARD ${CXX_STD})

if (MEM_CHECK)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

target_include_directories(
  test_as_common
  PUBLIC ${CMAKE_CURRENT_BINARY_DIR} 
         ${PROJECT_BINARY_DIR}/csrc/
         ${PROJECT_SOURCE_DIR}/csrc/
         ${PROJECT_SOURCE_DIR}/csrc/common/
         ${PROJECT_SOURCE_DIR}/csrc/device/
         ${PROJECT_SOURCE_DIR}/csrc/device/cpu/  
         ${PROJECT_SOURCE_DIR}/csrc/device/cuda/
         ${PROJECT_SOURCE_DIR}/csrc/interface/
         ${PROJECT_SOURCE_DIR}/csrc/core/kernel/
         ${PROJECT_SOURCE_DIR}/csrc/runtime/
  )



if(ENABLE_ARM_V84_V9)
  if(${CMAKE_CXX_COMPILER} MATCHES "clang") # arm compiler
      list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2a+dotprod+fp16+bf16+i8mm+sve+sve2")
  else()  # gcc
      list(APPEND ALLSPARK_CXX_FLAGS "-march=armv8.2-a+dotprod+fp16+bf16+i8mm+sve+sve2")
  endif()
endif()

# interface test
file(GLOB_RECURSE INTERFACE_TEST_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp)
list(APPEND INTERFACE_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp)

add_executable(cpp_interface_test ${INTERFACE_TEST_SRCS})
target_compile_definitions(cpp_interface_test PUBLIC ${ALLSPARK_DEFINITION})
target_link_libraries(cpp_interface_test test_as_common CONAN_PKG::protobuf CONAN_PKG::zlib)

# operator test
file(GLOB_RECURSE OPERATOR_TEST_CPU_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/operator/cpu/*.cpp)
list(APPEND OPERATOR_TEST_SRCS ${OPERATOR_TEST_CPU_SRCS}
     ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp)

if(ENABLE_CUDA)
  file(GLOB_RECURSE OPERATOR_TEST_CUDA_SRCS
       ${CMAKE_CURRENT_SOURCE_DIR}/operator/cuda/*.cpp)
  list(APPEND OPERATOR_TEST_SRCS ${OPERATOR_TEST_CUDA_SRCS})
endif()
add_executable(cpp_operator_test ${OPERATOR_TEST_SRCS})
target_link_libraries(cpp_operator_test test_as_common CONAN_PKG::protobuf CONAN_PKG::zlib)
target_compile_definitions(cpp_operator_test PUBLIC ${ALLSPARK_DEFINITION})
target_compile_options(cpp_operator_test PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${ALLSPARK_CXX_FLAGS}>)

# kernel test
if(ENABLE_CUDA)
  file(GLOB_RECURSE KERNEL_TEST_CUDA_SRCS
       ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cuda/*.cpp)
  list(APPEND KERNEL_TEST_SRCS ${KERNEL_TEST_CUDA_SRCS})
add_executable(cpp_kernel_test ${KERNEL_TEST_SRCS})
target_include_directories(cpp_kernel_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cuda/)
target_link_libraries(cpp_kernel_test test_as_common CONAN_PKG::protobuf CONAN_PKG::zlib allspark_kernel)
endif()


# model test
file(GLOB_RECURSE MODEL_TEST_CPU_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/model/cpu/*.cpp)
list(APPEND MODEL_TEST_SRCS ${MODEL_TEST_CPU_SRCS}
     ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp)

if (ENABLE_CUDA)
  file(GLOB_RECURSE MODEL_TEST_CUDA_SRCS
       ${CMAKE_CURRENT_SOURCE_DIR}/model/cuda/*.cpp)
  list(APPEND MODEL_TEST_SRCS ${MODEL_TEST_CUDA_SRCS})
endif()


add_executable(cpp_model_test ${MODEL_TEST_SRCS})


target_link_libraries(cpp_model_test test_as_common CONAN_PKG::protobuf CONAN_PKG::zlib)
target_compile_definitions(cpp_model_test PUBLIC ${ALLSPARK_DEFINITION})


if (ENABLE_MULTINUMA)
  # sample test
  add_executable(model_stress_test_client ${CMAKE_CURRENT_SOURCE_DIR}/model/stresstest/model_stress_test.cpp)
  target_link_libraries(model_stress_test_client allspark_client pthread)
  target_include_directories(model_stress_test_client PUBLIC
           ${CMAKE_CURRENT_BINARY_DIR}
           ${PROJECT_SOURCE_DIR}/csrc/interface/
           ${PROJECT_SOURCE_DIR}/csrc/service/
  )
  target_compile_definitions(model_stress_test_client PRIVATE ENABLE_AS_SERVICE)
endif()

# under cuda: command line run with ASAN_OPTIONS=protect_shadow_gap=false
add_executable(model_stress_test ${CMAKE_CURRENT_SOURCE_DIR}/model/stresstest/model_stress_test.cpp)
target_link_libraries(model_stress_test allspark_framework pthread)
target_include_directories(model_stress_test PUBLIC
         ${CMAKE_CURRENT_BINARY_DIR}
         ${PROJECT_SOURCE_DIR}/csrc/interface/
         ${PROJECT_SOURCE_DIR}/csrc/service/
)


# make test
add_test(NAME cpp_interface_test
         COMMAND ${CMAKE_BINARY_DIR}/bin/cpp_interface_test)
add_test(NAME cpp_operator_test
         COMMAND ${CMAKE_BINARY_DIR}/bin/cpp_operator_test)
add_test(NAME cpp_model_test COMMAND ${CMAKE_BINARY_DIR}/bin/cpp_model_test)
