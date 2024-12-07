'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    setup.py
'''
import os
import pathlib
import subprocess

from setuptools import setup, Extension, find_namespace_packages
from setuptools import find_packages
from setuptools.command.build_ext import build_ext

from jinja2 import Template


py_pkg_name = os.getenv("AS_PYTHON_PKG_NAME", "dashinfer") # upstream branch, use dashinfer, internal package use pyhie
package_include_group = ["dashinfer", "dashinfer.*"]
package_exclude_group = ["pyhie", "pyhie.*"]

if py_pkg_name == "pyhie-allspark":
    print("use pyhie allspark as package name.")
    package_exclude_group = ["dashinfer", "dashinfer.*"]
    package_include_group = ["pyhie", "pyhie.*"]

def is_ccache_installed():
    try:
        # 尝试运行 `ccache --version` 并捕捉其输出
        result = subprocess.run(
            ["ccache", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # 如果返回码为 0，说明 `ccache` 成功运行
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果发生 CalledProcessError 或 FileNotFoundError，说明 `ccache` 不可用
        return False

class CMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):

    def build_extension(self, ext):
        cwd = pathlib.Path(__file__).parent.absolute()
        print(self.get_ext_fullpath(ext.name))

        extdir = pathlib.Path(self.build_lib).absolute()
        os.makedirs(extdir, exist_ok=True)
        os.makedirs(self.build_temp, exist_ok=True)
        # setup.py cannot pass parameter, use env to control it.
        # 11.4
        cuda_version = os.getenv("AS_CUDA_VERSION", "12.4")
        nccl_version = os.getenv("AS_NCCL_VERSION", "2.23.4")
        cuda_sm = os.getenv("AS_CUDA_SM", "'80;86'")
        nv_system_lib = os.getenv("AS_SYSTEM_NV_LIB", "OFF")
        config = os.getenv("AS_BUILD_TYPE", "Release")
        as_platform = os.getenv("AS_PLATFORM", "cuda")
        as_nv_static_lib = os.getenv("AS_NV_STATIC_LIB", "OFF")
        as_build_hiednn = os.getenv("AS_BUILD_HIEDNN", "OFF")
        as_utest = os.getenv("BUILD_UTEST", "OFF")
        as_span_attn = os.getenv("ENABLE_SPAN_ATTENTION", "ON")
        as_flash_attn = os.getenv("FLASHATTN_BUILD_FROM_SOURCE", "OFF")
        py_pkg_name_prefix = py_pkg_name.split('-')[0]

        enable_glibcxx11_abi = os.getenv("AS_CXX11_ABI", "OFF")
        enable_multinuma = os.getenv("ENABLE_MULTINUMA", "OFF")

        is_arm = as_platform.startswith("arm")

        print("AllSpark build with cuda :{} nccl:{}".format(
            cuda_version, nccl_version))

        cmake_args = []

        cmake_args_cuda = [
            "-DBUILD_PYTHON=ON",
            "-DBUILD_UTEST=" + as_utest,
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCUDA_VERSION=" + cuda_version,
            "-DNCCL_VERSION=" + nccl_version,
            "-DENABLE_NV_STATIC_LIB=" + as_nv_static_lib,
            "-DCMAKE_CUDA_ARCHITECTURES=" + cuda_sm,
            "-DPYTHON_LIB_DIRS=" + str(pathlib.Path(extdir)) + f"/{py_pkg_name_prefix}/allspark",
            "-DMEM_CHECK=OFF",
            "-DLOCK_CHECK=OFF",
            "-DALWAYS_READ_LOAD_MODEL=OFF",
            "-DBUILD_HIEDNN=" + as_build_hiednn,
            "-DENABLE_SPAN_ATTENTION=" + as_span_attn,
            "-DFLASHATTN_BUILD_FROM_SOURCE=" + as_flash_attn,
            "-DENABLE_JSON_MODE=ON",
            "-DENABLE_GLIBCXX11_ABI=" + enable_glibcxx11_abi,
            "-DENABLE_MULTINUMA=OFF",
        ]

        if is_ccache_installed():
            cmake_args_cuda.append("-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")

        if nv_system_lib != "OFF":
            cmake_args_cuda.append("-DUSE_SYSTEM_NV_LIB=ON")

        cmake_args_x86 = [
            "-DBUILD_PYTHON=ON",
            "-DBUILD_UTEST=" + as_utest,
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DPYTHON_LIB_DIRS=" + str(pathlib.Path(extdir)) + f"/{py_pkg_name_prefix}/allspark",
            "-DMEM_CHECK=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCONFIG_ACCELERATOR_TYPE=NONE",
            "-DENABLE_ARMCL=OFF",
            "-DENABLE_CUDA=OFF",
            "-DALWAYS_READ_LOAD_MODEL=ON",
            "-DCONFIG_HOST_CPU_TYPE=X86",
            "-DENABLE_BF16=ON",
            "-DENABLE_FP16=ON",
            "-DENABLE_FP8=OFF",
            "-DALLSPARK_CBLAS=MKL",
            "-DALWAYS_READ_LOAD_MODEL=ON",
            "-DENABLE_SPAN_ATTENTION=OFF",
            "-DENABLE_JSON_MODE=ON",
            "-DENABLE_GLIBCXX11_ABI=" + enable_glibcxx11_abi,
            "-DENABLE_MULTINUMA=" + enable_multinuma,
        ]

        cmake_args_arm = [
            "-DBUILD_PYTHON=ON",
            "-DBUILD_UTEST=" + as_utest,
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DPYTHON_LIB_DIRS=" + str(pathlib.Path(extdir)) + f"/{py_pkg_name_prefix}/allspark",
            "-DMEM_CHECK=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCONFIG_ACCELERATOR_TYPE=NONE",
            "-DENABLE_CUDA=OFF",
            "-DALWAYS_READ_LOAD_MODEL=ON",
            "-DCONFIG_HOST_CPU_TYPE=ARM",
            "-DENABLE_AVX2=OFF",
            "-DENABLE_AVX512=OFF",
            "-DENABLE_BF16=ON",
            "-DENABLE_FP16=ON",
            "-DENABLE_FP8=OFF",
            "-DENABLE_ARMCL=ON",
            "-DALLSPARK_CBLAS=BLIS",
            "-DENABLE_ARM_V84_V9=ON",
            "-DCMAKE_C_COMPILER=armclang",
            "-DCMAKE_CXX_COMPILER=armclang++",
            "-DENABLE_SPAN_ATTENTION=OFF",
            "-DENABLE_JSON_MODE=ON",
            "-DENABLE_GLIBCXX11_ABI=" + enable_glibcxx11_abi,
            "-DENABLE_MULTINUMA=" + enable_multinuma,
        ]

        if as_platform.startswith("cuda"):
            cmake_args = cmake_args_cuda
        elif as_platform.startswith("x86"):
            cmake_args = cmake_args_x86
        else:
            if is_arm:
                cmake_args = cmake_args_arm
            else:
                raise Exception("unknown platform")
        print(f"setup.py: cmake args")
        for arg in cmake_args:
            print(f"\t{arg}")

        def os_script_exec(cmd: str):
            import os
            print("setup.py: ", cmd)
            return os.system(cmd)

        third_party_subfolder = os.path.join(cwd.parent.absolute(),
                                             "third_party", "from_source")

        if as_build_hiednn == "ON":
            print(f"setup.py: build hiednn from source.")
            hiednn_revision = None
            with open(os.path.join(third_party_subfolder, "HIE-DNN.version"),
                      'r') as file:
                hiednn_revision = file.read().rstrip()

            if not hiednn_revision:
                raise Exception("Empty HIE-DNN version")

            hiednn_src_folder = os.path.join(third_party_subfolder, "HIE-DNN")
            if os.path.exists(hiednn_src_folder):
                os.chdir(hiednn_src_folder)
                os_script_exec("git remote update")
                os.chdir(cwd)
            else:
		raise ValueError("not found HIE-DNN source code.")

            os.chdir(hiednn_src_folder)
            os_script_exec("git reset --hard {}".format(hiednn_revision))
            os.chdir(cwd)

        os.chdir(self.build_temp)

        # prepare for connan env.
        if enable_glibcxx11_abi == "ON":
            libcxx_setting = "libstdc++11"
        else:
            libcxx_setting = "libstdc++"

        conanfile = "conanfile"
        openmpi_rebuild  = ""
        if enable_multinuma == "ON":
            conanfile += "_openmpi"
            openmpi_rebuild = "-b openmpi -b bison"
        if is_arm:
            conanfile += "_arm"
        conanfile += ".txt"


        conan_install_arm = Template(
            "conan profile new dashinfer_compiler_profile --detect --force\n" +
            "cp -f {{cwd_parent}}/conan/conanprofile_armclang.aarch64 ~/.conan/profiles/dashinfer_compiler_profile\n" +
            "cp -r {{cwd_parent}}/conan/settings_arm.yml ~/.conan/settings.yml\n" +
            "conan profile update settings.compiler.libcxx={{libcxx_setting}} dashinfer_compiler_profile\n" +
            "conan install {{cwd_parent}}/conan/{{conanfile}} -pr dashinfer_compiler_profile -b outdated {{openmpi_rebuild}} -b protobuf -b gtest -b glog"
        ).render(libcxx_setting=libcxx_setting, cwd_parent=str(cwd.parent), conanfile=conanfile, openmpi_rebuild=openmpi_rebuild)

        conan_install_other = Template(
            "conan profile new dashinfer_compiler_profile --detect --force\n" +
            "conan profile update settings.compiler.libcxx={{libcxx_setting}} dashinfer_compiler_profile\n" +
            "conan install {{cwd_parent}}/conan/{{conanfile}} -pr dashinfer_compiler_profile -b outdated {{openmpi_rebuild}} -b protobuf -b gtest -b glog"
        ).render(libcxx_setting=libcxx_setting, cwd_parent=str(cwd.parent), conanfile=conanfile, openmpi_rebuild=openmpi_rebuild)

        conan_install_cmd = conan_install_other

        export_path_cmd = ""
        if is_arm:
            # need manual export on arm
            export_path_cmd = "export PATH=$PATH:" + str(
                cwd.absolute()) + "/" + self.build_temp + "/bin"
            conan_install_cmd = conan_install_arm

        # because it's require a conan virtual env, so we must write a shell to execute it.
        bash_template = Template(
            '#!/bin/bash -x\n' + 'gcc --version\n' + 'set -e\n'
            #+ '{{conan_install_cmd}}\n'  # uncomment this line if you want to clean rebuild.
            +
            'if [ ! -f "activate.sh" ]; \nthen {{conan_install_cmd}};\n fi\n'  # conan install in here to make sure protoc and protobuf have same version.
            + 'source ./activate.sh\n' + '{{export_path_cmd}}\n' +
            'cmake --version\n' + 'cmake {{dir}} {{cmake_args}} \n' +
            'cmake --build . --target install -j16\n')

        with open('.python_build.sh', 'w') as f:
            script_content = bash_template.render(
                dir=str(cwd.parent),
                cmake_args=" ".join(cmake_args),
                conan_install_cmd=conan_install_cmd,
                export_path_cmd=export_path_cmd,
                cwd_parent=str(cwd.parent),
                extdir=str(pathlib.Path(extdir)))
            print(f"setup.py script_content:")
            print(script_content)
            f.write(script_content)

        self.spawn(["bash", ".python_build.sh"])
        os.chdir(str(cwd))
        return  # build_extension



setup(name=f"{py_pkg_name}",
      version=os.getenv("AS_RELEASE_VERSION", "2.0.0"),
      author="DashInfer team",
      author_email="Dash-Infer@alibabacloud.com",
      description="DashInfer is a native inference engine for Pre-trained Large Language Models (LLMs) developed by Tongyi Laboratory.",
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.10',
      ],
      license="Apache License 2.0",
      packages=find_namespace_packages(include=package_include_group),
      ext_modules=[CMakeExtension("_allspark")],
      cmdclass={"build_ext": CMakeBuild},
      setup_requires=["jinja2"],
      install_requires=["transformers>=4.40.0", "torch", "ruamel.yaml"],
      zip_safe=False,
      python_requires=">=3.8",
      extra_compile_args=["-O3"])
