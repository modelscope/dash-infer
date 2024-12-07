if (ENABLE_CUDA)
    include_directories(
        ${PROJECT_SOURCE_DIR}/csrc/interface
        ${PROJECT_SOURCE_DIR}/csrc/utility
    )
    add_definitions(-DENABLE_CUDA)
else()
    include_directories(
        ${PROJECT_SOURCE_DIR}/csrc/interface
        ${PROJECT_SOURCE_DIR}/csrc/utility
        ${PROJECT_SOURCE_DIR}/csrc/service
    )
endif()
if (ENABLE_JSON_MODE)
  add_definitions(-DENABLE_JSON_MODE)
endif()
if (CONFIG_ACCELERATOR_TYPE STREQUAL "NONE")
    add_definitions(-DUSE_MPI_SERVICE)
endif()
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
find_package(pybind11 REQUIRED)

# include(pybind11Tools)
pybind11_add_module(_allspark MODULE "allspark_binding.cpp")
target_include_directories(_allspark
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/csrc)
target_link_options(_allspark PRIVATE -Wl,--whole-archive $<TARGET_FILE:allspark_framework_static> -Wl,--no-whole-archive -rdynamic)
target_link_libraries(_allspark PRIVATE
   allspark_framework_static
   CONAN_PKG::protobuf
   CONAN_PKG::zlib)
set_target_properties(_allspark PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(_allspark PROPERTIES CXX_STANDARD 17)
if(UNIX AND NOT APPLE)
  set(ALLSPARK_LINK_MAP ${PROJECT_SOURCE_DIR}/link_python.map)
  set_target_properties(
    _allspark PROPERTIES
      LINK_DEPENDS ${ALLSPARK_LINK_MAP}
      LINK_FLAGS   "-Wl,--version-script=${ALLSPARK_LINK_MAP}")
endif()

###

if (ENABLE_MULTINUMA)
    pybind11_add_module(_allspark_client MODULE "allspark_client_binding.cpp")
    target_include_directories(_allspark_client
      PRIVATE
        ${CUDAToolkit_INCLUDE_DIR}
        ${CMAKE_BINARY_DIR}/csrc)
    target_link_options(_allspark_client PRIVATE "-rdynamic")
    target_link_libraries(_allspark_client PRIVATE
       -Wl,--whole-archive
       allspark_client
       -Wl,--no-whole-archive
       CONAN_PKG::protobuf)
    # target_link_libraries(_allspark_client PRIVATE allspark_client)
    set_target_properties(_allspark_client PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(_allspark_client PROPERTIES CXX_STANDARD 17)
    if(UNIX AND NOT APPLE)
      set(ALLSPARK_LINK_MAP ${PROJECT_SOURCE_DIR}/link_python.map)
      set_target_properties(
        _allspark_client PROPERTIES
          LINK_DEPENDS ${ALLSPARK_LINK_MAP}
          LINK_FLAGS   "-Wl,--version-script=${ALLSPARK_LINK_MAP}")
    endif()
endif()
