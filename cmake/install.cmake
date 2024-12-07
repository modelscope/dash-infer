# add install target
SET_TARGET_PROPERTIES(allspark_framework PROPERTIES INSTALL_RPATH "$ORIGIN")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/interface/
    DESTINATION include/allspark/
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*.h"
)
if (NOT BUILD_PYTHON)
    install(TARGETS allspark_framework_static DESTINATION ${CMAKE_INSTALL_DIR})
endif()

if (ENABLE_MULTINUMA)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/orterun
                DESTINATION bin
                RENAME mpirun)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/allspark_daemon
                DESTINATION bin
                RENAME allspark_daemon)
    SET_TARGET_PROPERTIES(allspark_client PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS allspark_client DESTINATION ${CMAKE_INSTALL_DIR})
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/service/
                DESTINATION include/allspark/
                USE_SOURCE_PERMISSIONS FILES_MATCHING
                PATTERN "allspark_client.h")
endif()

if (BUILD_PYTHON)
    if (PYTHON_LIB_DIRS)
        if(NOT ENABLE_NV_STATIC_LIB)
            install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} DESTINATION ${PYTHON_LIB_DIRS} FILES_MATCHING PATTERN "*" PATTERN "libnccl.*" EXCLUDE)
        else()
            install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} DESTINATION ${PYTHON_LIB_DIRS} FILES_MATCHING PATTERN "*")
        endif()
        if (ENABLE_MULTINUMA)
            install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin DESTINATION ${PYTHON_LIB_DIRS} USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "*")
            SET_TARGET_PROPERTIES(_allspark_client PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
            install(TARGETS _allspark_client DESTINATION ${PYTHON_LIB_DIRS})
        endif()
        SET_TARGET_PROPERTIES(_allspark PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
        install(TARGETS _allspark DESTINATION ${PYTHON_LIB_DIRS})
    endif()
else()
  install(TARGETS allspark_framework DESTINATION ${CMAKE_INSTALL_DIR})
endif()
