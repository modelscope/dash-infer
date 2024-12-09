message("========== curl ==========")
option(BUILD_CURL "build curl" ON)
if(BUILD_CURL)
  message(STATUS "Build curl from submodule")
  set(BUILD_TESTING
      OFF
      CACHE BOOL "Build curl tests")

  set(BUILD_CURL_EXE OFF CACHE BOOL "Build curl exe")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static curl")
  set(CURL_USE_OPENSSL ON CACHE BOOL "curl use openssl")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/from_source/curl
                   EXCLUDE_FROM_ALL)
  unset(BUILD_TESTING CACHE)
  unset(BUILD_SHARED_LIBS CACHE)
  unset(BUILD_CURL_EXE CACHE)
  unset(CURL_USE_OPENSSL CACHE)
else()
  message(STATUS "Use curl from system")
  find_package(CURL REQUIRED)
endif()
set(CURL_LIBRARY CURL::libcurl)
message("=============================")
