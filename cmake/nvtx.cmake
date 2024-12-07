message("============ NVTX  Start ================")
set(NVTX_DIR
    ${PROJECT_SOURCE_DIR}/third_party/from_source/NVTX/c)


add_subdirectory(${NVTX_DIR} EXCLUDE_FROM_ALL)
message("============ NVTX  End ====== ===========")
