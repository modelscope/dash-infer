if(${ALLSPARK_CBLAS} MATCHES "NONE")
  #message(FATAL_ERROR "Invalid or Unsupported CBLAS choice")
  return()
endif()
include(ExternalProject)
if(cblas_cmake_included)
  return()
endif()
set(cblas_cmake_included true)

message(STATUS "ALLSPARK_CBLAS:${ALLSPARK_CBLAS}")

if(ALLSPARK_CBLAS MATCHES "MKL")
  list(APPEND ALLSPARK_PUBLIC_DEFINITIONS -DALLSPARK_USE_MKL_)
  set(MKLML_ARCH x64)
  include(mkl)
  set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE})
  set(CBLAS_LIBRARY MKL::MKL)
elseif(ALLSPARK_CBLAS MATCHES "BLIS")
  list(APPEND ALLSPARK_PUBLIC_DEFINITIONS -DALLSPARK_USE_CBLAS_ -DALLSPARK_USE_BLIS_)
#  set(BLIS_URL ${OSS_LOCATION}/blis-0.9.0.tar.gz)
#  set(BLIS_URL_MD5 b39045e450d612f712365b2f4bc16f18)
  set(BLIS_INSTALL ${INSTALL_LOCATION}/blis-0.9.0)
  set(CBLAS_INCLUDE_DIR ${BLIS_INSTALL}/include/blis)
  set(CBLAS_LIBRARY_DIR ${BLIS_INSTALL}/lib)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(CBLAS_LIBRARY ${CBLAS_LIBRARY_DIR}/blis.lib)
  else()
    set(CBLAS_LIBRARY ${CBLAS_LIBRARY_DIR}/libblis.a)
  endif()

  if (${TARGET_PLATFORM} MATCHES "x86")
    message(FATAL_ERROR "cannot use blis on x86 now")
    return()
  else()
    set(BLIS_ARCH "arm64")
  endif()

  if(${CMAKE_BUILD_TYPE} MATCHES "Release")
      set(BLIS_DEBUG_TYPE "opt")
  else()
      set(BLIS_DEBUG_TYPE "DEBUG")
  endif()

  if(${RUNTIME_THREAD} MATCHES "OMP")
		  set(BLIS_PARALLEL "openmp")
  else()
		  set(BLIS_PARALLEL "pthreads")
  endif()

  if(${CMAKE_C_COMPILER} MATCHES "clang") # clang or armclang
	  if (ENABLE_ARM_V84_V9 MATCHES "ON")
		  set(BLIS_MARCH "armv8.2a+sve")
	  else()
		  set(BLIS_MARCH "armv8.2a")
	  endif()
  else()  # assuming gcc or cc
	  if (ENABLE_ARM_V84_V9 MATCHES "ON")
      	 set(BLIS_MARCH "armv8.2-a+sve")
	  else()
      	 set(BLIS_MARCH "armv8.2-a")
	  endif()
  endif()


  if(NOT EXISTS ${CBLAS_LIBRARY})
    message(STATUS "Start to build BLIS, PREFIX=${BLIS_INSTALL}, libname=${CBLAS_LIBRARY}")
    ExternalProject_Add(
      project_blis
      PREFIX PREFIX ${CMAKE_CURRENT_BINARY_DIR}/blis
      URL ${CMAKE_SOURCE_DIR}/third_party/blis.tar.bz2
      BUILD_IN_SOURCE true
      CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} AR=${CMAKE_AR} RANLIB=${CMAKE_RANLIB} CFLAGS=-march=${BLIS_MARCH} ./configure --prefix=${BLIS_INSTALL} -t ${BLIS_PARALLEL} -d opt --enable-cblas ${BLIS_ARCH}
      BUILD_COMMAND pwd && make -j 32
      INSTALL_COMMAND pwd && make install
    )
  else()
    message(STATUS "blis installed successfully")
    add_custom_target(project_blis)
  endif()
endif()
