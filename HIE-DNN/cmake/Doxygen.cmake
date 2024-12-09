if(DOXYGEN_CMAKE)
    return()
endif()
set(DOXYGEN_CMAKE true)

find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/doc)
    configure_file(${PROJECT_SOURCE_DIR}/doc/Doxyfile.in
                   ${PROJECT_BINARY_DIR}/doc/Doxyfile @ONLY)
    file(GLOB_RECURSE DOXY_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.h)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
        DEPENDS ${DOXY_DEPENDS}
        VERBATIM)
else()
    message(STATUS "doxygen not found")
endif()

