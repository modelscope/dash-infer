set(PROTO_SERVICE_RAW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../proto/")
set(PROTO_SERVICE_RAW_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../proto/allspark_service.proto")
message(STATUS "PROTO_SERVICE_RAW_FILES:${PROTO_SERVICE_RAW_FILES} CMAKE_CURRENT_BINARY_DIR:${CMAKE_CURRENT_BINARY_DIR} PROTOBUF_PROTOC_EXECUTABLE:${PROTOBUF_PROTOC_EXECUTABLE}")
protobuf_generate_cpp(PROTO_SVC_SRCS PROTO_SVC_HDRS  ${PROTO_SERVICE_RAW_FILES})

set(PROTO_SVC_GRPC_SRC "${CMAKE_CURRENT_BINARY_DIR}/allspark_service.grpc.pb.cc")
set(PROTO_SVC_GRPC_HDR "${CMAKE_CURRENT_BINARY_DIR}/allspark_service.grpc.pb.h")
set(PROTO_SVC_GEN_GRPC "${CMAKE_CURRENT_BINARY_DIR}/../../bin/grpc_cpp_plugin")

add_custom_command(
    OUTPUT "${PROTO_SVC_SRCS}" "${PROTO_SVC_HDRS}" "${PROTO_SVC_GRPC_SRC}" "${PROTO_SVC_GRPC_HDR}"
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
    -I "${PROTO_SERVICE_RAW_PATH}"
    --plugin=protoc-gen-grpc="${PROTO_SVC_GEN_GRPC}"
    "${PROTO_SERVICE_RAW_FILES}"
    DEPENDS ${PROTO_SERVICE_RAW_FILES}
)

add_custom_target(protobuf_svc_target DEPENDS ${PROTO_SVC_SRCS} ${PROTO_SVC_HDRS} ${PROTO_SVC_GRPC_SRC} ${PROTO_SVC_GRPC_HDR})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs")
add_executable(allspark_daemon ${PROTO_SVC_SRCS} ${PROTO_SVC_GRPC_SRC} allspark_service.cpp)

set_target_properties(allspark_daemon PROPERTIES CXX_STANDARD 17)  # c17 for grpc
target_link_libraries(allspark_daemon allspark_framework CONAN_PKG::grpc CONAN_PKG::protobuf CONAN_PKG::glog)
target_include_directories(
  allspark_daemon
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../interface)

add_library(allspark_client SHARED ${PROTO_SVC_SRCS} ${PROTO_SVC_GRPC_SRC} allspark_client.cpp allspark_client_impl.cpp allspark_service_parallel.cpp)

set_target_properties(allspark_client PROPERTIES CXX_STANDARD 17)  # c17 for grpc
target_link_libraries(allspark_client CONAN_PKG::grpc CONAN_PKG::protobuf CONAN_PKG::glog ${THREAD_LIB})
target_include_directories(
  allspark_client
  PUBLIC
  ${THREAD_INCLUDE}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../interface)
