message("============ cpp-ipc  Start ================")
set(CPP_IPC_DIR
    ${PROJECT_SOURCE_DIR}/third_party/from_source/cpp-ipc)
message("CPP_IPC_DIR: ${CPP_IPC_DIR}")
add_subdirectory(${CPP_IPC_DIR} EXCLUDE_FROM_ALL)
set(CPP_IPC_INCLUDE ${CPP_IPC_DIR}/include)
set(CPP_IPC_LIBRARY ipc)
message("============ cpp-ipc  End ==================")
