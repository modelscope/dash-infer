message("========== flash-attention ==========")
set(FLASHATTN_CUDA_VERSION
    ${CUDA_VERSION}
    CACHE STRING "flash-attn cuda version")
set(FLASHATTN_GPU_ARCHS
    ${CMAKE_CUDA_ARCHITECTURES}
    CACHE STRING "flash-attn gpu archs")
list(REMOVE_ITEM FLASHATTN_GPU_ARCHS "70")
list(REMOVE_ITEM FLASHATTN_GPU_ARCHS "75")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "FLASHATTN_GPU_ARCHS: ${FLASHATTN_GPU_ARCHS}")

set(FLASHATTN_USE_EXTERNAL_CUTLASS
    ON
    CACHE BOOL "flash-attn use external cutlass target")

set(FLASHATTN_USE_CUDA_STATIC
    ON
    CACHE BOOL "flash-attn use static CUDA")

set(FLASHATTN_USE_STATIC_LIB
    OFF
    CACHE BOOL "use flash-attn static lib")

# only static link when needed, to reduce size.
if(ENABLE_NV_STATIC_LIB)
  set(FLASHATTN_USE_CUDA_STATIC ON)
else()
  set(FLASHATTN_USE_CUDA_STATIC OFF)
endif()

if (FLASHATTN_USE_STATIC_LIB)
  set(FLASHATTN_LIBRARY_NAME libflash-attn.a)
else()
  set(FLASHATTN_LIBRARY_NAME libflash-attn.so)
endif()


include(ExternalProject)

  message(STATUS "build flash-attention from source")

    message(STATUS "Use flash-attention from external project")
    set(FLASH_ATTENTION_GIT_REPO https://github.com/Dao-AILab/flash-attention.git)
    set(FLASH_ATTENTION_GIT_TAG 7551202cb2dd245432bc878447e19015c0af3c22)
    set(FLASH_ATTENTION_GIT_PATCH ${PROJECT_SOURCE_DIR}/third_party/patch/flash-attn.patch)

  set(FLASHATTN_INSTALL ${INSTALL_LOCATION}/flash-attention/install)
  set(FLASHATTN_LIBRARY_PATH ${FLASHATTN_INSTALL}/lib/)

  ExternalProject_Add(
    project_flashattn
    GIT_REPOSITORY ${FLASH_ATTENTION_GIT_REPO}
    GIT_TAG ${FLASH_ATTENTION_GIT_TAG}
    GIT_SUBMODULES ""
    PATCH_COMMAND git apply --reverse --check ${FLASH_ATTENTION_GIT_PATCH} || git apply ${FLASH_ATTENTION_GIT_PATCH}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/flash-attention
    SOURCE_SUBDIR csrc
    DEPENDS project_cutlass
    CMAKE_GENERATOR "Ninja"
    BUILD_COMMAND ${CMAKE_COMMAND} --build . -j32 -v
    BUILD_BYPRODUCTS ${FLASHATTN_LIBRARY_PATH}/${FLASHATTN_LIBRARY_NAME}
    USES_TERMINAL true
    CMAKE_CACHE_ARGS
        -DFLASHATTN_GPU_ARCHS:STRING=${FLASHATTN_GPU_ARCHS}
    CMAKE_ARGS
        -DFLASHATTN_CUDA_VERSION=${FLASHATTN_CUDA_VERSION}
        -DFLASHATTN_USE_EXTERNAL_CUTLASS=${FLASHATTN_USE_EXTERNAL_CUTLASS}
        -DFLASHATTN_USE_CUDA_STATIC=${FLASHATTN_USE_CUDA_STATIC}
        -DCMAKE_INSTALL_PREFIX=${FLASHATTN_INSTALL}
        -DCUTLASS_INSTALL_PATH=${CUTLASS_INSTALL}
  )

  ExternalProject_Get_Property(project_flashattn SOURCE_DIR)
  ExternalProject_Get_Property(project_flashattn SOURCE_SUBDIR)
  set(FLASHATTN_INCLUDE_DIR ${SOURCE_DIR}/${SOURCE_SUBDIR})


message(STATUS "FLASHATTN_LIBRARY_PATH: ${FLASHATTN_LIBRARY_PATH}")
message(STATUS "FLASHATTN_INCLUDE_DIR: ${FLASHATTN_INCLUDE_DIR}")

if (FLASHATTN_USE_STATIC_LIB)
  add_library(flash-attention::flash-attn STATIC IMPORTED)
else()
  add_library(flash-attention::flash-attn SHARED IMPORTED)
  install(FILES ${FLASHATTN_LIBRARY_PATH}/libflash-attn.so
          DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "libflash-attn.so installing path: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

set_target_properties(flash-attention::flash-attn PROPERTIES
                      IMPORTED_LOCATION ${FLASHATTN_LIBRARY_PATH}/${FLASHATTN_LIBRARY_NAME})
include_directories(${FLASHATTN_INCLUDE_DIR})
set(FLASHATTN_LIBRARY flash-attention::flash-attn)

unset(FLASHATTN_CUDA_VERSION)
unset(FLASHATTN_GPU_ARCHS)
unset(FLASHATTN_USE_EXTERNAL_CUTLASS)
unset(FLASHATTN_USE_CUDA_STATIC)
message("=====================================")
