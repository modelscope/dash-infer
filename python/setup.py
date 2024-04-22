#
# Copyright (c) Alibaba, Inc. and its affiliates.
# @file    setup.py
#
import os
import pathlib
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext

from jinja2 import Template


class CMakeExtension(Extension):

    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):

    def build_extension(self, ext):
        cwd = pathlib.Path(__file__).parent.absolute()
        extdir = pathlib.Path(self.build_lib).absolute()
        print(f"ext_fullpath: {self.get_ext_fullpath(ext.name)}\n extdir:{extdir} \ntmp path:{self.build_temp}")
        os.makedirs(extdir, exist_ok=True)
        os.makedirs(extdir / "lib64", exist_ok=True)
        os.makedirs(self.build_temp, exist_ok=True)
        os.makedirs(Path(self.build_temp) / "lib64", exist_ok=True)
        config = os.getenv("AS_BUILD_TYPE", "Release")
        as_platform = os.getenv("AS_PLATFORM", "x86")
        enable_glibcxx11_abi = os.getenv("AS_CXX11_ABI", "OFF")
        # python package keep c++03 abi.

        cmake_args = []

        cmake_args_x86 = [
            "-DBUILD_PYTHON=ON",
            "-DBUILD_PACKAGE=ON",
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DPYTHON_LIB_DIRS=" + str(pathlib.Path(extdir)) + "/dashinfer/allspark",
            "-DMEM_CHECK=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DENABLE_ARMCL=OFF",
            "-DENABLE_GLIBCXX11_ABI=" + enable_glibcxx11_abi,
            "-DCONFIG_HOST_CPU_TYPE=X86",
            "-DENABLE_BF16=ON",
            "-DENABLE_FP16=ON",
            "-DALLSPARK_CBLAS=MKL",
        ]

        cmake_args_arm = [
            "-DBUILD_PYTHON=ON",
            "-DBUILD_PACKAGE=ON",
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DPYTHON_LIB_DIRS=" + str(pathlib.Path(extdir)) + "/dashinfer/allspark",
            "-DMEM_CHECK=OFF",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCONFIG_HOST_CPU_TYPE=ARM",
            "-DENABLE_AVX2=OFF",
            "-DENABLE_BF16=ON",
            "-DENABLE_FP16=ON",
            "-DENABLE_ARMCL=ON",
            "-DENABLE_GLIBCXX11_ABI=" + enable_glibcxx11_abi,
            "-DALLSPARK_CBLAS=BLIS",
            "-DENABLE_ARM_V84_V9=ON",
            "-DCMAKE_C_COMPILER=armclang",
            "-DCMAKE_CXX_COMPILER=armclang++"
        ]

        if as_platform.startswith("x86"):
            cmake_args = cmake_args_x86
        elif as_platform.startswith("arm"):
            cmake_args = cmake_args_arm
        else:
            raise Exception("unknown platform")
        print("setup.py: cmake args")
        for arg in cmake_args:
            print(f"\t{arg}")

        def os_script_exec(cmd: str):
            import os
            print("setup.py: ", cmd)
            return os.system(cmd)

        third_party_subfolder = os.path.join(cwd.parent.absolute(),
                                             "third_party", "from_source")

        os.chdir(self.build_temp)

        # prepare for connan env.
        if enable_glibcxx11_abi == "ON":
            libcxx_setting = "libstdc++11"
        else:
            libcxx_setting = "libstdc++"

        conan_install_arm = Template(
            "conan profile new cxx11abi --detect --force\n" +
            "cp -f {{cwd_parent}}/conan/conanprofile_armclang.aarch64 ~/.conan/profiles/cxx11abi\n" +
            "cp -r {{cwd_parent}}/conan/settings_arm.yml ~/.conan/settings.yml\n" +
            "conan profile update settings.compiler.libcxx={{libcxx_setting}} cxx11abi\n" +
            "conan install {{cwd_parent}}/conan/conanfile_arm.txt -pr cxx11abi -b missing -b protobuf -b gtest -b openssl -b grpc -b glog -b abseil"
        ).render(libcxx_setting=libcxx_setting, cwd_parent=str(cwd.parent))

        conan_install_other = Template(
            "conan profile new cxx11abi --detect --force\n" +
            "conan profile update settings.compiler.libcxx={{libcxx_setting}} cxx11abi\n" +
            "conan install {{cwd_parent}}/conan/conanfile.txt -pr cxx11abi -b missing -b protobuf -b gtest -b openssl -b grpc -b glog -b abseil"
        ).render(libcxx_setting=libcxx_setting, cwd_parent=str(cwd.parent))

        conan_install_cmd = conan_install_other

        export_path_cmd = ""
        if as_platform.startswith("arm"):
            # need manual export on arm
            export_path_cmd = "export PATH=$PATH:" + str(
                cwd.absolute()) + "/" + self.build_temp + "/bin"
            conan_install_cmd = conan_install_arm

        # because it's require a conan virtual env, so we must write a shell to execute it.
        bash_template = Template(
            '#!/bin/bash -x\n' + 'set -x ;\n' + 'set -ex\n'
            # + '{{conan_install_cmd}}\n' # uncomment this line if you want to clean rebuild.
            + 'if [ ! -f "activate.sh" ]; \nthen {{conan_install_cmd}};\n fi\n'  # conan install in here to make sure protoc and protobuf have same version.
            + 'source ./activate.sh\n' + '{{export_path_cmd}}\n'
            + 'cmake --version\n' + './bin/protoc --version\n'
            + 'find -name protoc\n'
            + 'cmake {{dir}}  {{cmake_args}} || cmake {{dir}} {{cmake_args}};\n'
            + 'cmake --build . --target install -j32\n'
            '')

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


setup(name="dashinfer",
      version=os.getenv("AS_RELEASE_VERSION", "1.0.0"),
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
      packages=find_namespace_packages(include=['dashinfer.*']),
      ext_modules=[CMakeExtension("_allspark")],
      cmdclass={"build_ext": CMakeBuild},
      setup_requires=["jinja2"],
      install_requires=["protobuf==3.18"],
      zip_safe=False,
      python_requires=">=3.8",
      extra_compile_args=["-O3"])
