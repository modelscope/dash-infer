cmake_minimum_required(VERSION 3.12)
project(DashInferExample)

# install some 3rdparty requirement
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# make sure your dashinfer allspark have same abi. otherwise, there will be a
# std::string crash.
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

###########################################
# Example 1: Single NUMA or GPU qwen v1 example.
###########################################
add_executable(
  example_qwen 0_basic/example_qwen.cpp tokenizer/tokenizer.cpp
                           tokenizer/base64.cpp)

target_link_libraries(example_qwen PRIVATE allspark_framework
                                                       )

target_include_directories(example_qwen PRIVATE tokenizer utils)

###########################################
# Example 2: Multi-NUMA qwen v1 example.
###########################################
#if (MEM_CHECK)
#  message("ddd")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
#endif()

#add_executable(
#  example_qwen_multi_numa 0_basic/example_qwen.cpp tokenizer/tokenizer.cpp
#                          tokenizer/base64.cpp)
#target_compile_definitions(example_qwen_multi_numa PRIVATE -DEXAMPLE_MULTI_NUMA)
#target_link_libraries(example_qwen_multi_numa PRIVATE allspark_client
#                                                      )
#target_include_directories(example_qwen_multi_numa PRIVATE tokenizer utils)
#target_include_directories(example_qwen_multi_numa
#                           PRIVATE /usr/include/allspark/)

###########################################
# Example 3: simple api Server
###########################################
include(FetchContent)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

set(EXTERNAL_PROJECTS_PREFIX ${CMAKE_BINARY_DIR}/external-projects)
set(EXTERNAL_PROJECTS_INSTALL_PREFIX ${EXTERNAL_PROJECTS_PREFIX}/installed)
ExternalProject_Add(
  externalRestbed
  PREFIX "${EXTERNAL_PROJECTS_PREFIX}"
  EXCLUDE_FROM_ALL TRUE
  URL "https://github.com/Corvusoft/restbed/archive/refs/tags/4.8.zip"
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECTS_INSTALL_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DBUILD_SSL=NO
    -DBUILD_TESTS=NO
    -Dasio_INCLUDE=${EXTERNAL_PROJECTS_PREFIX}/src/externalRestbed/dependency/asio/asio/include
    -DCMAKE_CXX_FLAGS="-Wno-narrowing"
  PATCH_COMMAND
    echo "Disabling catch" && sed -i s/find_package\(\ catch/\#/ CMakeLists.txt
    && echo "Disabling tests" && sed -i s/add_subdirectory/\#/ CMakeLists.txt
    && echo "Downloading ASIO" && cd dependency && wget -q -O asio.tar.gz
    https://github.com/Corvusoft/asio-dependency/archive/b3d2ab7255fabe46a49b24a584c9fd797c8248e5.tar.gz
    && rm -rf asio && tar -xf asio.tar.gz && mv
    asio-dependency-b3d2ab7255fabe46a49b24a584c9fd797c8248e5 asio && rm
    asio.tar.gz)

add_executable(tgi_server 1_apiserver/apiserver.cpp tokenizer/tokenizer.cpp
                          tokenizer/base64.cpp)
target_link_directories(tgi_server PRIVATE /home/jiejing.zjj/workspace/HIE-AllSpark/build/HIE-AllSpark-3.0.0/lib64/)
target_include_directories(tgi_server PRIVATE /home/jiejing.zjj/workspace/HIE-AllSpark/build/HIE-AllSpark-3.0.0/include)


add_dependencies(tgi_server externalRestbed)

# single numa server
target_include_directories(
  tgi_server
  PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/include /usr/include/allspark/
          ${CMAKE_SOURCE_DIR}/utils ${CMAKE_SOURCE_DIR}/tokenizer)

target_link_directories(
  tgi_server PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib
  ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64
  ${CMAKE_SOURCE_DIR}/1_apiserver/restbed/distribution/library)
target_link_libraries(
  tgi_server PRIVATE restbed allspark_framework
                     nlohmann_json::nlohmann_json)

# multi numa server
#add_executable(
#  tgi_server_multi_numa 1_apiserver/apiserver.cpp tokenizer/tokenizer.cpp
#                        tokenizer/base64.cpp)

#add_dependencies(tgi_server_multi_numa externalRestbed)

#target_include_directories(
#  tgi_server_multi_numa
#  PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/include /usr/include/allspark/
#          ${CMAKE_SOURCE_DIR}/utils ${CMAKE_SOURCE_DIR}/tokenizer)

#target_compile_definitions(tgi_server_multi_numa PRIVATE -DEXAMPLE_MULTI_NUMA)
#target_link_directories(
#  tgi_server_multi_numa PRIVATE ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib
#  ${EXTERNAL_PROJECTS_INSTALL_PREFIX}/lib64)
#target_link_libraries(
#  tgi_server_multi_numa PRIVATE restbed allspark_client
#                                nlohmann_json::nlohmann_json)
