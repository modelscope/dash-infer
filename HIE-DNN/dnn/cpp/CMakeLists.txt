# hiednn cpp backend source file
set(HIEDNN_CPP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HIEDNN_CPP_SRC
    ${HIEDNN_CPP_SRC_DIR}/hiednn_cpp.cpp
    ${HIEDNN_CPP_SRC_DIR}/cast.cpp
    ${HIEDNN_CPP_SRC_DIR}/set_tensor_value.cpp
    ${HIEDNN_CPP_SRC_DIR}/unary_elementwise.cpp
    ${HIEDNN_CPP_SRC_DIR}/expand.cpp)

add_library(hiednn_cpp OBJECT ${HIEDNN_CPP_SRC})
set_property(TARGET hiednn_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

