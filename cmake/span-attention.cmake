message("========== span-attention ==========")

set(SPANATTN_CUDA_ARCHS
    ${CMAKE_CUDA_ARCHITECTURES}
    CACHE STRING "spanattn CUDA archs")
set(SPANATTN_STATIC_CUDART
    ${ENABLE_NV_STATIC_LIB}
    CACHE BOOL "spanattn use static CUDA runtime" FORCE)
set(SPANATTN_EXTERNAL_CUTLASS
    ON
    CACHE BOOL "spanattn use external cutlass" FORCE)
set(SPANATTN_ENABLE_TEST
    OFF
    CACHE BOOL "spanattn build tests" FORCE)

message(STATUS "SPANATTN_CUDA_ARCHS: ${SPANATTN_CUDA_ARCHS}")
message(STATUS "SPANATTN_STATIC_CUDART: ${SPANATTN_STATIC_CUDART}")

set(SPANATTN_INSTALL ${INSTALL_LOCATION}/span-attention/install)
set(SPANATTN_LIBRARY_PATH ${SPANATTN_INSTALL}/lib/libspanattn.a)
message(STATUS "SPANATTN_INSTALL: ${SPANATTN_INSTALL}")
message(STATUS "SPANATTN_LIBRARY_PATH: ${SPANATTN_LIBRARY_PATH}")

set(SPANATTN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/span-attention)

include(ExternalProject)

ExternalProject_Add(
    project_spanattn
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/span-attention
    SOURCE_DIR ${SPANATTN_SOURCE_DIR}
    DEPENDS project_cutlass
    CMAKE_GENERATOR "Ninja"
    BUILD_COMMAND ${CMAKE_COMMAND} --build . -j32 -v
    BUILD_BYPRODUCTS ${SPANATTN_LIBRARY_PATH}
    CMAKE_CACHE_ARGS
        -DSPANATTN_CUDA_ARCHS:STRING=${SPANATTN_CUDA_ARCHS}
    CMAKE_ARGS
        -DSPANATTN_STATIC_CUDART=${SPANATTN_STATIC_CUDART}
        -DSPANATTN_EXTERNAL_CUTLASS=${SPANATTN_EXTERNAL_CUTLASS}
        -DSPANATTN_ENABLE_TEST=${SPANATTN_ENABLE_TEST}
        -DCMAKE_INSTALL_PREFIX=${SPANATTN_INSTALL}
        -DCUTLASS_INSTALL_PATH=${CUTLASS_INSTALL}
)

unset(SPANATTN_CUDA_ARCHS)
unset(SPANATTN_STATIC_CUDART)
unset(SPANATTN_EXTERNAL_CUTLASS)
unset(SPANATTN_ENABLE_TEST)

file(MAKE_DIRECTORY ${SPANATTN_INSTALL}/include)
add_library(spanattn::spanattn_static STATIC IMPORTED)
add_dependencies(spanattn::spanattn_static project_spanattn)
set_target_properties(spanattn::spanattn_static
    PROPERTIES
        IMPORTED_LOCATION ${SPANATTN_LIBRARY_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${SPANATTN_INSTALL}/include
)
set(SPANATTN_LIBRARY spanattn::spanattn_static)
message("====================================")
