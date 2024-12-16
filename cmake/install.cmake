# add install target
install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/interface/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/allspark
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*.h"
)
install(TARGETS allspark_framework DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS allspark_framework_static DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (ENABLE_MULTINUMA)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/service/
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/allspark
                USE_SOURCE_PERMISSIONS FILES_MATCHING
                PATTERN "allspark_client.h")
    install(TARGETS allspark_client DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/orterun
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            RENAME mpirun)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/allspark_daemon
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                RENAME allspark_daemon)
endif()

if (BUILD_PYTHON)
if (PYTHON_LIB_DIRS)
    install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} DESTINATION ${PYTHON_LIB_DIRS} FILES_MATCHING PATTERN "libflash-attn.so")

    if (ENABLE_MULTINUMA)
        install(TARGETS _allspark_client DESTINATION ${PYTHON_LIB_DIRS})
    endif()
    install(TARGETS _allspark DESTINATION ${PYTHON_LIB_DIRS})
endif()
endif()
