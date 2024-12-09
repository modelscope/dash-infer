# hiednn include dir
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/example)

# CPP example
if(USE_CPP)
    add_subdirectory(01_cpp)
endif()

# CUDA example
if(USE_CUDA)
    add_subdirectory(02_cuda)
endif()

