find_package(Protobuf REQUIRED)
protobuf_generate_cpp(AS_MODEL_CPP AS_MODEL_HDR  ${AS_MODEL_PROTO_FILE})
# protobuf generate requires a target to trigger.
add_custom_target(as_model_generate DEPENDS ${AS_MODEL_CPP} ${AS_MODEL_HDR})
message(STATUS "Protobuf C++ file: ${AS_MODEL_CPP}, ${AS_MODEL_HDR}")

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
    ${CMAKE_BINARY_DIR}/csrc
)

set(ALLSPARK_3RD_LIBS
    ${THREAD_LIB}
)
if(ENABLE_ARMCL MATCHES "ON")
    message(STATUS "ARMCL_LIBRARY:${ARMCL_LIBRARY} ONEDNN_LIBRARY: ${ONEDNN_LIBRARY}")
    list(APPEND ALLSPARK_3RD_LIBS ${ARMCL_LIBRARY})
endif()

list(APPEND ALLSPARK_3RD_LIBS CONAN_PKG::openmpi ${CPP_IPC_LIBRARY})

file(GLOB_RECURSE ALLSPARK_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utility/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/model/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/operator/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/tensor/*.cpp
)

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

list(APPEND ALLSPARK_SRCS ${AS_MODEL_CPP})

add_library(allspark_framework SHARED ${ALLSPARK_SRCS})
set_target_properties(allspark_framework PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(allspark_framework PRIVATE ${ALLSPARK_INC_DIR})

add_dependencies(allspark_framework as_model_generate)

# rdynamic for backtrace.
if (CONFIG_HOST_CPU_TYPE STREQUAL "ARM")
target_compile_options(allspark_framework PRIVATE "-rdynamic")
else()
target_compile_options(allspark_framework PRIVATE "-rdynamic;-mcmodel=large")
endif()

if (MEM_CHECK)
    target_link_options(allspark_framework PUBLIC "-fsanitize=address")
endif()

target_link_libraries(allspark_framework PRIVATE ${ONEDNN_LIBRARY} git_version stdc++fs allspark_kernel CONAN_PKG::glog CONAN_PKG::protobuf  ${ALLSPARK_3RD_LIBS} )

if(ENABLE_ARMCL)
    add_dependencies(allspark_framework project_armcl)
endif()

if(UNIX AND NOT APPLE)
  set(ALLSPARK_LINK_MAP ${PROJECT_SOURCE_DIR}/link.map)
  set_target_properties(
    allspark_framework PROPERTIES
      LINK_DEPENDS ${ALLSPARK_LINK_MAP}
      LINK_FLAGS   "-Wl,--version-script=${ALLSPARK_LINK_MAP}")
endif()

add_dependencies(allspark_framework project_dnnl)
set_target_properties(allspark_framework PROPERTIES INSTALL_RPATH "$ORIGIN")
add_subdirectory(service)
