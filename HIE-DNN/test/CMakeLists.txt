if(NOT TARGET gtest)
    # gtest subdirectory
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest-release-1.11.0
                     ${PROJECT_BINARY_DIR}/test/google_test
                     EXCLUDE_FROM_ALL)
endif()

# hiednn include dir
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/dnn/include)
include_directories(${PROJECT_SOURCE_DIR}/test/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/utest)

# datatype extension utest
add_subdirectory(datatype_extension)

# CPP utest
if(USE_CPP)
    add_subdirectory(cpp)
endif()

# CUDA utest
if(USE_CUDA)
    add_subdirectory(cuda)
endif()

