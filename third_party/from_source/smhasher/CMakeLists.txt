cmake_minimum_required(VERSION 2.4)

project(smhasher)

message("============ smhasher cmakelists ================")

# set(CMAKE_BUILD_TYPE Release)

add_library(
  smhasher
  MurmurHash3.cpp
  City.cpp
)
