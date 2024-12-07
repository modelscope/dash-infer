if(ENABLE_ARMCL MATCHES "ON")
  message(STATUS "enter armcl build...")
  include(ExternalProject)
  if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpfullversion -dumpversion
            OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
    list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)

    message(STATUS "cmake version=${CMAKE_VERSION}")

    set(GCC_VERSION "${GCC_MAJOR}.${GCC_MINOR}")
    message(STATUS "gcc version=${GCC_VERSION}")

    add_definitions("-Wno-error")
  endif()

  if (ENABLE_BF16 MATCHES "ON")
      set(ARMCL_BUILD_ARCH armv8.6-a-sve)
  elseif (ENABLE_ARM_V84_V9 MATCHES "ON")
      set(ARMCL_BUILD_ARCH armv8.2-a-sve)
  elseif (ENABLE_FLOAT16 MATCHES "ON")
      set(ARMCL_BUILD_ARCH armv8.2)
  else ()
      set(ARMCL_BUILD_ARCH armv8-a)
  endif()
  message(STATUS "build arch for ArmCL: ${ARMCL_BUILD_ARCH} INSTALL_LOCATION: ${INSTALL_LOCATION}")
  list(APPEND HIE_PUBLIC_DEFINITIONS -DHIE_USE_ARMCL_)

  #set(ARMCL_URL ${OSS_LOCATION}/ComputeLibrary-22.02.tar.gz)
  #set(ARMCL_URL http://test-bucket-duplicate.oss-cn-hangzhou.aliyuncs.com/daoxian/HCI/ComputeLibrary-22.08.tar.gz)
  #set(ARMCL_URL_MD5 b1f0bd88535f7fecb97139986cd5a9cb)
  set(ARMCL_INSTALL ${INSTALL_LOCATION}/ComputeLibrary-22.08)
  # set(ARMCL_SRC ${CMAKE_CURRENT_BINARY_DIR}/armcl)
  set(ARMCL_SRC ${PROJECT_SOURCE_DIR}/third_party/armcl)

  set(ARMCL_INCLUDE_DIR ${ARMCL_INSTALL}/include)
  set(ARMCL_LIBRARY_DIR ${ARMCL_INSTALL}/lib)
  include_directories(AFTER ${ARMCL_INCLUDE_DIR})
  # link_directories(${ARMCL_LIBRARY_DIR})

  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(ARMCL_LIBRARY ${ARMCL_LIBRARY_DIR}/libarm_compute.lib)
  else()
    set(ARMCL_LIBRARY ${ARMCL_LIBRARY_DIR}/libarm_compute-static.a)
    # add_compile_options(-std=c++14)  # later: update with CXX_STD / C17 support.
  endif()

  if(${CMAKE_BUILD_TYPE} MATCHES "Release")
      set(ARMCL_DEBUG_TYPE 0)
  else()
      set(ARMCL_DEBUG_TYPE 1)
  endif()

  if(${RUNTIME_THREAD} MATCHES "OMP")
      set(ARMCL_USE_CPPTHREADS 0)
      set(ARMCL_USE_OPENMP 1)
  else()
      set(ARMCL_USE_CPPTHREADS 1)
      set(ARMCL_USE_OPENMP 0)
  endif()

  if(ENABLE_ASAN_MEM_CHECK)
    set(ARMCL_EXTRA_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    set(ARMCL_EXTRA_LINK_FLAGS -lsan)
  endif()

  if(NOT EXISTS ${ARMCL_LIBRARY})
    message(STATUS "Start to build ArmCL, PREFIX=${ARMCL_INSTALL}, CC=${CMAKE_C_COMPILER}")
    ExternalProject_Add(
      project_armcl
      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/armcl
      URL ${CMAKE_SOURCE_DIR}/third_party/armcl.tar.bz2
      BUILD_IN_SOURCE true
      CONFIGURE_COMMAND echo "no configure for ArmCL"
      BUILD_COMMAND pwd && CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} AR=${CMAKE_CXX_COMPILER_AR} RANLIB=${CMAKE_CXX_COMPILER_RANLIB} scons  Werror=0 debug=${ARMCL_DEBUG_TYPE} neon=1 opencl=0 cppthreads=${ARMCL_USE_CPPTHREADS} openmp=${ARMCL_USE_OPENMP} os=linux arch=${ARMCL_BUILD_ARCH} extra_cxx_flags=-fPIC\ ${ARMCL_EXTRA_CXX_FLAGS} extra_link_flags=${ARMCL_EXTRA_LINK_FLAGS} examples=0 benchmark_tests=0 validation_tests=0 -j32 install_dir=${ARMCL_INSTALL} 
      INSTALL_COMMAND pwd && echo "no extra install cmd"
    )
  else()
    message(STATUS "ArmCL installed successfully")
    add_custom_target(project_armcl)
  endif()
endif()
