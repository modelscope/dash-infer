# cpp backend utest source file
set(CPP_UTEST_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CPP_UTEST_SRC
    ${CPP_UTEST_SRC_DIR}/main.cpp
    ${CPP_UTEST_SRC_DIR}/cast_utest.cpp
    ${CPP_UTEST_SRC_DIR}/set_tensor_utest.cpp
    ${CPP_UTEST_SRC_DIR}/unary_elementwise_utest.cpp
    ${CPP_UTEST_SRC_DIR}/expand_utest.cpp)

add_executable(cpp_utest ${CPP_UTEST_SRC})
target_link_libraries(cpp_utest hiednn gtest)

# make test
add_test(NAME cpp_utest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpp_utest)

