include_directories(
    ${PROJECT_SOURCE_DIR}/csrc/interface
    ${PROJECT_SOURCE_DIR}/csrc/utility
    ${PROJECT_SOURCE_DIR}/csrc/service
)
add_definitions(-DUSE_MPI_SERVICE)

find_package(Python REQUIRED COMPONENTS Interpreter Development )
find_package(pybind11 REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

pybind11_add_module(_allspark MODULE "allspark_binding.cpp")

target_link_libraries(_allspark PRIVATE pybind11::headers allspark_framework allspark_client)
set_target_properties(_allspark PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_STANDARD 17
                                VISIBILITY_INLINES_HIDDEN ON)

# generate python inferface.
find_package(Protobuf REQUIRED)
protobuf_generate_python(AS_MODEL_PYTHON  ${AS_MODEL_PROTO_FILE})
# protobuf generate requires a target to trigger.
add_custom_target(as_model_generate_python DEPENDS ${AS_MODEL_PYTHON})
add_dependencies(_allspark as_model_generate_python)

# install python allspark pb files.
install(FILES ${AS_MODEL_PYTHON} DESTINATION ${PYTHON_LIB_DIRS}/model)
