include(GNUInstallDirs)

set(MKL_PROJECT "extern_mkl")
include(FetchContent)
set(MKL_URL ${CMAKE_SOURCE_DIR}/third_party/mkl_2022.0.2.tar.gz)

#list(APPEND CMAKE_MODULE_PATH ${${MKL_PROJECT}_SOURCE_DIR}/lib/cmake/mkl)

message("MKL root: ${MKL_ROOT_DIR}, module path:${CMAKE_MODULE_PATH}")


if (${RUNTIME_THREAD} STREQUAL "TBB")
    FetchContent_Declare(${MKL_PROJECT}
        URL ${MKL_URL}
        )
    set(MKL_THREADING tbb_thread)
elseif(${RUNTIME_THREAD} STREQUAL "OMP")
    FetchContent_Declare(${MKL_PROJECT}
        URL ${MKL_URL}
        )
    set(MKL_THREADING gnu_thread)
endif()

message(STATUS "Fetch MKL from ${MKL_URL}")
FetchContent_MakeAvailable(${MKL_PROJECT})

set(MKL_ROOT_DIR
    ${${MKL_PROJECT}_SOURCE_DIR}
    CACHE PATH "MKL library")

set(MKL_ROOT
    ${${MKL_PROJECT}_SOURCE_DIR}
    CACHE PATH "MKL library")

set(MKL_ROOT ${MKL_ROOT_DIR})

set(MKL_INTERFACE lp64)
set(MKL_LINK static)
set(MKL_H ${MKL_ROOT}/include)

find_package(MKL REQUIRED)
