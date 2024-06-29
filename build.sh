set -x

clean="OFF"

# 捕获arch命令的输出
architecture=$(arch)

# 使用if-else结构进行条件判断
if [ "${architecture}" == "aarch64" ]; then
    export AS_PLATFORM=armclang
else
    export AS_PLATFORM=x86
fi

if [ -z "$AS_PLATFORM" ];
then
	echo " please set AS_PLATFORM env, AS_PLATFORM can be x86 or armclang"
	exit 1
fi

# with_platform, to support x86/arm build
with_platform="${AS_PLATFORM}"
build_type="${AS_BUILD_TYPE:-Release}"
build_package="${AS_BUILD_PACKAGE:-OFF}"
enable_glibcxx11_abi="${AS_CXX11_ABI:-ON}" # default enable cxx11 ABI


function clone_pull {
  GIT_URL=$1
  DIRECTORY=$2
  GIT_COMMIT=$3
  if [ -d "$DIRECTORY" ]; then
    pushd "$DIRECTORY"
    git remote update
    popd
  else
    git clone "$GIT_URL" "$DIRECTORY"
  fi
  pushd "$DIRECTORY"
  git reset --hard "$GIT_COMMIT"
  popd
}

if [ "$clean" == "ON" ]; then
    rm -rf build
fi

if [ ! -d "./build"  ]; then
    mkdir build && cd build
    if [ "${with_platform,,}" == "armclang" ]; then
      conan profile new cxx11abi --detect --force
      cp -f ../conan/conanprofile_armclang.aarch64 ~/.conan/profiles/cxx11abi
      cp -r ../conan/settings_arm.yml ~/.conan/settings.yml
      if [ "$enable_glibcxx11_abi" == "ON" ]; then
        conan profile update settings.compiler.libcxx=libstdc++11 cxx11abi
      else
        conan profile update settings.compiler.libcxx=libstdc++ cxx11abi
      fi
      conan install ../conan/conanfile_arm.txt -pr cxx11abi -b missing -b protobuf -b gtest -b openssl -b grpc -b glog -b abseil
    else
      conan profile new cxx11abi --detect --force
      if [ "$enable_glibcxx11_abi" == "ON" ]; then
        conan profile update settings.compiler.libcxx=libstdc++11 cxx11abi
      else
        conan profile update settings.compiler.libcxx=libstdc++ cxx11abi
      fi
      conan install ../conan/conanfile.txt -pr cxx11abi -b missing -b protobuf -b gtest -b openssl -b grpc -b glog -b abseil
    fi
    cd ../
fi

cd build
source ./activate.sh
export PATH=`pwd`/bin:$PATH

if [ "${with_platform,,}" == "x86" ]; then
  cmake .. \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DCONFIG_HOST_CPU_TYPE=X86 \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DALLSPARK_CBLAS=MKL \
      -DBUILD_PACKAGE=${build_package} 
elif [ "${with_platform,,}" == "armclang" ]; then
  cmake .. \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DCONFIG_HOST_CPU_TYPE=ARM \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_ARMCL=ON \
      -DBUILD_PACKAGE=${build_package} \
      -DALLSPARK_CBLAS=BLIS \
      -DENABLE_AVX2=OFF \
      -DENABLE_AVX512=OFF \
      -DENABLE_ARM_V84_V9=ON \
      -DENABLE_BF16=ON \
      -DENABLE_FP16=ON \
      -DCMAKE_C_COMPILER=armclang \
      -DCMAKE_CXX_COMPILER=armclang++
fi

# do the make and package.
# VERBOSE=1 make && make install
make -j32 && make install


if [ $? -eq 0 ]; then
  if [[ "${build_package}" == "ON" ]]; then
  make package
  fi
else
  exit $?
fi

