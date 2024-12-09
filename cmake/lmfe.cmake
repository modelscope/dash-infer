message("========== LMFE ==========")
message("Use lmfe-cpp from submodule")

set(LMFE_INSTALL ${INSTALL_LOCATION}/lmfe-cpp/install)
set(LMFE_GIT_REPO https://github.com/noamgat/lmfe-cpp.git)
set(LMFE_GIT_TAG 0b242cad8853aedd74e50e6e6be9a14803aed7cf)
set(LMFE_GIT_PATCH ${PROJECT_SOURCE_DIR}/third_party/patch/lmfe-cpp.patch)

include(FetchContent)
FetchContent_Declare(
  project_lmfe
  GIT_REPOSITORY ${LMFE_GIT_REPO}
  GIT_TAG ${LMFE_GIT_TAG}
  PATCH_COMMAND git apply --reverse --check ${LMFE_GIT_PATCH} || git apply ${LMFE_GIT_PATCH}
  BINARY_DIR ${LMFE_INSTALL}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
)
FetchContent_MakeAvailable(project_lmfe)

FetchContent_GetProperties(project_lmfe SOURCE_DIR LMFE_SOURCE_DIR)
set(LMFE_LIBRARY external_lmfe)

add_library(external_lmfe STATIC IMPORTED GLOBAL)

set_target_properties(external_lmfe PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/liblmfe_library.a)
set(LMFE_INCLUDE ${LMFE_SOURCE_DIR}/include)
add_dependencies(external_lmfe project_lmfe)
message("==========================")
