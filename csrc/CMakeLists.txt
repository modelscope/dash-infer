set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
# following warnings should not be ignored, but there are too many
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")  # ~600 warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")   # ~650 warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")  # ~300 warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")  # ~200 warnings

message(STATUS "Subdirectory csrc CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")

set(PROTO_RAW_FILES ${CMAKE_CURRENT_SOURCE_DIR}/proto/allspark.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS  ${PROTO_RAW_FILES})
message(STATUS "Protobuf file: ${PROTO_SRCS}, ${PROTO_HDRS}")
# 在执行 add_subdirectory 之前先执行 protobuf_generate_cpp
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} ${PROTO_RAW_FILES}
    DEPENDS ${PROTO_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(protobuf_target DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core/kernel)

set(ALLSPARK_INC_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/utility
    ${CMAKE_CURRENT_SOURCE_DIR}/interface
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core/kernel
    ${THREAD_INCLUDE}
    ${CPP_IPC_INCLUDE}
    ${SMHASHER_INCLUDE}
    ${CMAKE_BINARY_DIR}/csrc
)

    list(APPEND ALLSPARK_INC_DIR ${PROJECT_SOURCE_DIR}/third_party/from_source/cutlass/include)

set(ALLSPARK_3RD_LIBS
    ${THREAD_LIB}
)
if(ENABLE_ARMCL MATCHES "ON")
    message(STATUS "ARMCL_LIBRARY:${ARMCL_LIBRARY} ONEDNN_LIBRARY: ${ONEDNN_LIBRARY}")
    list(APPEND ALLSPARK_3RD_LIBS ${ARMCL_LIBRARY})
endif()

if (ENABLE_CUDA)
    list(APPEND ALLSPARK_3RD_LIBS
        ${CUDA_3RD_PARTY_LIBS}
    )
endif()

if (OSX_BUILD)
    list(APPEND ALLSPARK_3RD_LIBS ${TBB_LIB_STATIC})
endif()

if (ENABLE_MULTINUMA)
    list(APPEND ALLSPARK_3RD_LIBS CONAN_PKG::openmpi)
endif()
list(APPEND ALLSPARK_3RD_LIBS CONAN_PKG::zlib)

list(APPEND ALLSPARK_3RD_LIBS ${CPP_IPC_LIBRARY})
list(APPEND ALLSPARK_3RD_LIBS ${SMHASHER_LIBRARY})

if (ENABLE_JSON_MODE)
    list(APPEND ALLSPARK_INC_DIR ${LMFE_INCLUDE})
    list(APPEND ALLSPARK_3RD_LIBS ${LMFE_LIBRARY})
endif()

if(CONFIG_HOST_CPU_TYPE STREQUAL "X86")
    list(APPEND ALLSPARK_3RD_LIBS ig_static)
endif()


file(GLOB_RECURSE ALLSPARK_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/state/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/model/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/operator/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/*.cpp
)


if (CONFIG_ACCELERATOR_TYPE STREQUAL "CUDA")
    file(GLOB_RECURSE ALLSPARK_ACC_DEVICE_SPECIAL
        ${CMAKE_CURRENT_SOURCE_DIR}/device/cuda/*.cpp
        )
endif()

# device common
file(GLOB ALLSPARK_DEVICE_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp
    )

# host common
file(GLOB ALLSPARK_HOST_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/device/cpu/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device/cpu/cpp/*.cpp
    )

# host special for arm
if (CONFIG_HOST_CPU_TYPE STREQUAL "ARM")
     message("build with ARM op support.")
     file(GLOB_RECURSE ALLSPARK_HOST_SPECIAL
        ${CMAKE_CURRENT_SOURCE_DIR}/device/cpu/arm/*.cpp
        )
endif()

# host special for x86
if (CONFIG_HOST_CPU_TYPE STREQUAL "X86")
     file(GLOB_RECURSE ALLSPARK_HOST_SPECIAL
        ${CMAKE_CURRENT_SOURCE_DIR}/device/cpu/x86/*.cpp
        )
endif()

list(APPEND ALLSPARK_SRCS ${ALLSPARK_DEVICE_COMMON} ${ALLSPARK_ACC_DEVICE_SPECIAL} ${ALLSPARK_HOST_COMMON} ${ALLSPARK_HOST_SPECIAL})

list(APPEND ALLSPARK_SRCS ${PROTO_SRCS})

add_library(allspark_objects OBJECT ${ALLSPARK_SRCS})
set_target_properties(allspark_objects PROPERTIES CXX_STANDARD ${CXX_STD})  # c17 for cutlass.
set_target_properties(allspark_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(allspark_objects
    PRIVATE
        ${ONEDNN_LIBRARY}
        git_version
        allspark_kernel
        CONAN_PKG::glog
        CONAN_PKG::protobuf
        CONAN_PKG::zlib
        ${ALLSPARK_3RD_LIBS})
if (ENABLE_SPAN_ATTENTION)
target_compile_definitions(allspark_objects PRIVATE -DENABLE_SPAN_ATTENTION=1)
else()
target_compile_definitions(allspark_objects PRIVATE -DENABLE_SPAN_ATTENTION=0)
endif()
target_include_directories(allspark_objects PRIVATE ${ALLSPARK_INC_DIR})

# rdynamic for backtrace.
if (NOT CONFIG_HOST_CPU_TYPE STREQUAL "ARM")
    target_compile_options(allspark_objects PRIVATE "-mcmodel=large")
endif()

if (MEM_CHECK)
    target_compile_options(allspark_objects PUBLIC "-fsanitize=address")
    target_link_options(allspark_objects PUBLIC "-fsanitize=address")
endif()

if(ENABLE_ARMCL)
    add_dependencies(allspark_objects project_armcl)
endif()

if (ENABLE_CUDA_PINNED_WEIGHT_LOAD)
    target_compile_definitions(allspark_objects PRIVATE -DENABLE_CUDA_PINNED_WEIGHT)
endif()

# add_dependencies(allspark_objects project_dnnl)

if (ENABLE_CUDA)
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
add_dependencies(allspark_objects project_flashattn)
endif()
endif()

# build static library for _allspark which will be loaded in python process
add_library(allspark_framework_static STATIC $<TARGET_OBJECTS:allspark_objects>)
# The linking to allspark_objects only aims to bring propagation INTERFACE_XXXX properties
# It DOES NOT actually link to allspark_objects
target_link_libraries(allspark_framework_static
    PRIVATE
        allspark_objects)

set_target_properties(allspark_framework_static PROPERTIES INSTALL_RPATH "$ORIGIN")

# build shared library for allspark_daemon, cpp unit tests
add_library(allspark_framework SHARED $<TARGET_OBJECTS:allspark_objects>)
# The linking to allspark_objects only aims to bring propagation INTERFACE_XXXX properties
# It DOES NOT actually link to allspark_objects
target_link_libraries(allspark_framework
    PRIVATE
        allspark_objects)

if(UNIX AND NOT APPLE)
  if(BUILD_UTEST)
    # if test, protobuf-linking set global
    set(ALLSPARK_LINK_MAP ${PROJECT_SOURCE_DIR}/link_with_proto.map)
  else()  # BUILD_UTEST
    set(ALLSPARK_LINK_MAP ${PROJECT_SOURCE_DIR}/link.map)
  endif() # BUILD_UTEST
  set_target_properties(
    allspark_framework PROPERTIES
      LINK_DEPENDS ${ALLSPARK_LINK_MAP}
      LINK_FLAGS   "-Wl,--version-script=${ALLSPARK_LINK_MAP}")
endif()

if (ENABLE_MULTINUMA)
    add_subdirectory(service)
endif()
