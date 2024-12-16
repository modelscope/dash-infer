set -x

clean="OFF"

# with_platform, to support cuda/x86/arm build
with_platform="${AS_PLATFORM:-cuda}"
# cuda related version, provide a defualt value for cuda 11.4
cuda_version="${AS_CUDA_VERSION:-12.4}"
cuda_sm="${AS_CUDA_SM:-80;86;90a}"
NCCL_VERSION="${AS_NCCL_VERSION:-2.23.4}"

## NCCL Version Map:
## the corresponding pre-build nccl will download on oss.
# | CUDA Version | NCCL Version |
# | 10.2, 11.8        | 2.15.5       |
# | 11.[3,4,6],12.1   | 2.11.4       |
# | 12.2              | 2.21.5       |
# | 12.4              | 2.23.4       |

system_nv_lib="${AS_SYSTEM_NV_LIB:-OFF}"
build_type="${AS_BUILD_TYPE:-Release}"
cuda_static="${AS_CUDA_STATIC:-OFF}"
build_package="${AS_BUILD_PACKAGE:-ON}"
enable_glibcxx11_abi="${AS_CXX11_ABI:-OFF}"
enable_span_attn="${ENABLE_SPAN_ATTENTION:-ON}"
enable_multinuma="${ENABLE_MULTINUMA:-OFF}"

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

    conan profile new dashinfer_compiler_profile --detect --force
    conanfile=../conan/conanfile.txt

    if [ "${enable_multinuma}" == "ON" ]; then
      conanfile=../conan/conanfile_openmpi.txt
    fi

    if [ "${with_platform,,}" == "armclang" ]; then
      conanfile=../conan/conanfile_arm.txt
      if [ "${enable_multinuma}" == "ON" ]; then
        conanfile=../conan/conanfile_openmpi_arm.txt
      fi
      cp -f ../conan/conanprofile_armclang.aarch64 ~/.conan/profiles/dashinfer_compiler_profile
      cp -r ../conan/settings_arm.yml ~/.conan/settings.yml
    fi

    if [ "$enable_glibcxx11_abi" == "ON" ]; then
      conan profile update settings.compiler.libcxx=libstdc++11 dashinfer_compiler_profile
    else
      conan profile update settings.compiler.libcxx=libstdc++ dashinfer_compiler_profile
    fi

    conan install ${conanfile} -pr dashinfer_compiler_profile -b missing -b protobuf -b gtest -b glog
    cd ../
fi

cd build
source ./activate.sh
export PATH=`pwd`/bin:$PATH

if [ "${with_platform,,}" == "cuda" ]; then
  cmake .. \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DBUILD_PACKAGE=${build_package} \
      -DCONFIG_ACCELERATOR_TYPE=CUDA \
      -DCONFIG_HOST_CPU_TYPE=X86 \
      -DNCCL_VERSION=${NCCL_VERSION} \
      -DCUDA_VERSION=${cuda_version} \
      -DCMAKE_CUDA_ARCHITECTURES="${cuda_sm}" \
      -DUSE_SYSTEM_NV_LIB=${system_nv_lib} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_NV_STATIC_LIB=${cuda_static} \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DBUILD_PYTHON=OFF \
      -DALWAYS_READ_LOAD_MODEL=OFF \
      -DENABLE_SPAN_ATTENTION=${enable_span_attn} \
      -DENABLE_MULTINUMA=OFF
elif [ "${with_platform,,}" == "x86" ]; then
  cmake .. \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DBUILD_PACKAGE=${build_package} \
      -DCONFIG_ACCELERATOR_TYPE=NONE \
      -DCONFIG_HOST_CPU_TYPE=X86 \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DBUILD_PYTHON=OFF \
      -DALLSPARK_CBLAS=MKL \
      -DENABLE_CUDA=OFF \
      -DENABLE_SPAN_ATTENTION=OFF \
      -DALWAYS_READ_LOAD_MODEL=ON \
      -DENABLE_MULTINUMA=${enable_multinuma}
elif [ "${with_platform,,}" == "armclang" ]; then
  cmake .. \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DBUILD_PACKAGE=${build_package} \
      -DCONFIG_ACCELERATOR_TYPE=NONE \
      -DCONFIG_HOST_CPU_TYPE=ARM \
      -DENABLE_BLADE_AUTH=${enable_blade_auth} \
      -DENABLE_GLIBCXX11_ABI=${enable_glibcxx11_abi} \
      -DBUILD_PYTHON=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DENABLE_ARMCL=ON \
      -DALLSPARK_CBLAS=BLIS \
      -DENABLE_CUDA=OFF \
      -DENABLE_AVX2=OFF \
      -DENABLE_AVX512=OFF \
      -DENABLE_ARM_V84_V9=ON \
      -DENABLE_BF16=ON \
      -DENABLE_FP16=ON \
      -DCMAKE_C_COMPILER=armclang \
      -DCMAKE_CXX_COMPILER=armclang++ \
      -DENABLE_SPAN_ATTENTION=OFF \
      -DALWAYS_READ_LOAD_MODEL=ON \
      -DENABLE_MULTINUMA=${enable_multinuma}
fi

# do the make and package.
# VERBOSE=1 make && make install
make -j16 && make install


if [ $? -eq 0 ]; then
  if [ ${build_package} == "ON" ]; then
  make package
  fi
else
  exit $?
fi

