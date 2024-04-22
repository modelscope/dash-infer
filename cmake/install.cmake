include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-${PROJECT_VERSION})
message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX} CPACK_PACKAGE_DEVICE_NAME:${CPACK_PACKAGE_DEVICE_NAME}")
# add install target
SET_TARGET_PROPERTIES(allspark_framework PROPERTIES INSTALL_RPATH "$ORIGIN")

message(STATUS "install: build python: ${BUILD_PYTHON}")

if (NOT BUILD_PYTHON)
    install(TARGETS allspark_framework DESTINATION ${CMAKE_INSTALL_DIR} COMPONENT libraries)
    install(TARGETS allspark_client    DESTINATION ${CMAKE_INSTALL_DIR} COMPONENT libraries)
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/interface/
    DESTINATION include/allspark/
    COMPONENT headers
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*.h"
)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/orterun
            DESTINATION bin
            RENAME mpirun COMPONENT libraries)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/allspark_daemon
            DESTINATION bin
            RENAME allspark_daemon COMPONENT libraries)

SET_TARGET_PROPERTIES(allspark_client PROPERTIES INSTALL_RPATH "$ORIGIN")

install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/service/
            DESTINATION include/allspark/
            COMPONENT headers
            USE_SOURCE_PERMISSIONS FILES_MATCHING
            PATTERN "allspark_client.h")

string(TOUPPER "$ENV{AS_PYTHON_MANYLINUX}" AS_PYTHON_MANYLINUX_UPPERCASE)


if (BUILD_PYTHON)
    message(STATUS "install build lib into python lib:${PYTHON_LIB_DIRS}")
    if (PYTHON_LIB_DIRS)
        install(DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin DESTINATION ${PYTHON_LIB_DIRS} USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "*")

        install(FILES $<TARGET_FILE:_allspark> DESTINATION ${PYTHON_LIB_DIRS})
        message("manylinux flag: ${AS_PYTHON_MANYLINUX_UPPERCASE}")

        if(AS_PYTHON_MANYLINUX_UPPERCASE AND AS_PYTHON_MANYLINUX_UPPERCASE STREQUAL "ON" )
            message(STATUS "Building Python manylinux, skip install lib into python wheel.")
        else()
            message(STATUS "Building None ManyLinux Python, install framework lib into python package.")
            install(TARGETS allspark_framework DESTINATION ${PYTHON_LIB_DIRS})
            install(TARGETS allspark_client DESTINATION ${PYTHON_LIB_DIRS})
        endif()
    endif()
endif()
