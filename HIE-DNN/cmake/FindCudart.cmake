find_library(CUDART_LIBRARY_PATH cudart_static HINTS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Cudart DEFAULT_MSG
    CUDART_LIBRARY_PATH)

if(CUDART_FOUND AND NOT TARGET Cudart::cudart)
    add_library(Cudart::cudart UNKNOWN IMPORTED)

    set_target_properties(
        Cudart::cudart PROPERTIES
        IMPORTED_LOCATION ${CUDART_LIBRARY_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        INTERFACE_LINK_LIBRARIES "dl;rt")
endif()

