# Copyright (C) 2013 Daniel Scharrer
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the author(s) be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.


# Copyright (C) 2014 Greg Horn
# In order to comply with the above copyright, I am noting that I took
# Daniel's script and hacked it a bit, mostly changing paths and filters

find_package(PythonInterp)

# Add a target that runs cpplint.py
#
# Parameters:
# - TARGET_NAME the name of the target to add
# - SOURCES_LIST a complete list of source and include files to check
function(add_cpplint_target TARGET_NAME SOURCES_LIST)

    if(NOT PYTHONINTERP_FOUND)
        return()
    endif()

    list(REMOVE_DUPLICATES SOURCES_LIST)
    list(SORT SOURCES_LIST)

    add_custom_target(${TARGET_NAME}
        COMMAND "${CMAKE_COMMAND}" -E chdir
                "${CMAKE_CURRENT_SOURCE_DIR}"
                "${PYTHON_EXECUTABLE}"
                "${CMAKE_SOURCE_DIR}/third_party/cpplint-1.5.5/cpplint.py"
                "--recursive"
                "--extensions=h,hpp,c,cpp,cu,cuh"
                ${CPPLINT_EXCLUDE_CMD}
                ${SOURCES_LIST}
        DEPENDS ${SOURCES_LIST}
        COMMENT "cpplint scanning"
        VERBATIM)

endfunction()

function(cpplint_exclude EXCLUDE_PATH)
    set(CPPLINT_EXCLUDE_CMD ${CPPLINT_EXCLUDE_CMD} --exclude=${EXCLUDE_PATH} PARENT_SCOPE)
endfunction()

