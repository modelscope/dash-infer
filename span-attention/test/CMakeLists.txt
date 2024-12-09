set(SPANATTN_TEST ${CMAKE_PROJECT_NAME}_test)

# HIE
add_library(hie INTERFACE)
target_include_directories(hie INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/hie
)

set(SPANATTN_TEST_INC_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}
)
file(GLOB SPANATTN_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

set(SPANATTN_TEST_COMMON ${SPANATTN_TEST}_common)
add_library(${SPANATTN_TEST_COMMON} OBJECT ${SPANATTN_TEST_SRC})
target_include_directories(${SPANATTN_TEST_COMMON}
    PUBLIC ${SPANATTN_INTERFACE_DIR}
    PRIVATE ${SPANATTN_TEST_INC_DIR}
)
target_compile_definitions(${SPANATTN_TEST_COMMON}
    PUBLIC ${SPANATTN_DEFINITION}
)
target_link_libraries(${SPANATTN_TEST_COMMON}
    PUBLIC gtest hie ${SPANATTN_CUDART_LIBRARY}
)

set(SPANATTN_REF_KERNEL ${CMAKE_PROJECT_NAME}_ref_kernel)
add_subdirectory(ref_kernel)

add_subdirectory(test_lib)
