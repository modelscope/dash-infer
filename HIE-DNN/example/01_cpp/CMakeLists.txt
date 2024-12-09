set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/01_cpp)

function(hiednn_cuda_example BIN SRC)
    add_executable(cpp_${BIN} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC})
    set_target_properties(cpp_${BIN} PROPERTIES OUTPUT_NAME ${BIN})
    target_link_libraries(cpp_${BIN} hiednn)
endfunction()

hiednn_cuda_example(01_unary_elementwise    01_unary_elementwise.cpp)

