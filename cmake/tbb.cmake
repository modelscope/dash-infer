include(GNUInstallDirs)
include(FetchContent)

set(TBB_PROJECT tbb)
message("tbb:${CMAKE_SOURCE_DIR}/third_party/tbb_2021.5.1.tar.bz2")

FetchContent_Declare(${TBB_PROJECT}
    URL ${CMAKE_SOURCE_DIR}/third_party/tbb_2021.5.1.tar.bz2
    )

FetchContent_MakeAvailable(${TBB_PROJECT})

set(TBB_ROOT ${${TBB_PROJECT}_SOURCE_DIR})
set(_cmake_proj_dir "${TBB_ROOT}/lib/cmake/tbb")
set(TBB_FIND_RELEASE_ONLY ON)
find_package(TBB REQUIRED CONFIG COMPONENTS tbb HINTS ${_cmake_proj_dir})
get_property(TBB_INCLUDE TARGET TBB::tbb PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
