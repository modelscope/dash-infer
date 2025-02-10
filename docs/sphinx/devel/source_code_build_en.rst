Source Code Build
------------------

Requirements
=============

OS
,,,,,

  - Linux

Python
,,,,,,,

  - Python 3.8, 3.10, 3.11
  - PyTorch: any PyTorch version, CPU or GPU.

Compiler
,,,,,,,,,

- Tested compiler version:

  - gcc: 7.3.1, 11.4.0

  - arm compiler: 22.1, 24.04

CUDA
,,,,

  - CUDA sdk version >= 11.4
  - cuBLAS: CUDA sdk provided

Conan
,,,,,

 + **conan**:  C++ package management tools, can be installed by : ``pip install conan==1.66.0``, only 1.x is supported.

 .. note:: if there is any package-not-found issue, please make sure your conan center is available. Reset it with this command: `conan remote add conancenter https://center.conan.io`


Leak check tool
,,,,,,,,,,,,,,,,

+ If want to enabel asan, or tsan, install following packinges:

.. code-block:: shell

  yum install devtoolset-7-libasan-devel devtoolset-7-libtsan-devel


CPU
,,,

For multi-NUMA inference, ``numactl``, ``openmpi`` are required:

- for Ubuntu:

.. code-block:: shell

  apt-get install numactl libopenmpi-dev

- for CentOS:

.. code-block:: shell

  yum install numactl openmpi-devel openssh-clients -y


.. _docker-label:

Development Docker
==================

We have build some Docker image for easier development setup.

- CUDA 12.4

.. code-block:: shell

  docker run -d --name="dashinfer-dev-cu124-${USER}" \
    --shm-size=8g --gpus all \
    --network=host \
    -v $(pwd):/root/workspace/DashInfer \
    -w /root/workspace \
    -it registry-1.docker.io/dashinfer/dev-centos7-cu124
  docker exec -it "dashinfer-dev-cu124-${USER}" /bin/bash

- CPU-only (Linux x86 server)

.. code-block:: shell

  docker run -d --name="dashinfer-dev-${USER}" \
    --network=host \
    -v $(pwd):/root/workspace/DashInfer \
    -w /root/workspace \
    -it registry-1.docker.io/dashinfer/dev-centos7-x86
  docker exec -it "dashinfer-dev-${USER}" /bin/bash

- CPU-only (Linux ARM server)

.. code-block:: shell

  docker run -d --name="dashinfer-dev-${USER}" \
    --network=host \
    -v $(pwd):/root/workspace/DashInfer \
    -w /root/workspace \
    -it registry-1.docker.io/dashinfer/dev-centos8-arm
  docker exec -it "dashinfer-dev-${USER}" /bin/bash

.. note:: When creating a container for multi-NUMA inference, ``--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host`` arguments are required, because components such as numactl and openmpi need the appropriate permissions to run. If you only need to use the single NUMA API, you may not grant this permission.


Build from Source Code
======================

Build Python Package
,,,,,,,,,,,,,,,,,,,,

1. Build python package for CUDA:

.. code-block:: bash

  cd python
  AS_CUDA_VERSION="12.4" AS_NCCL_VERSION="2.23.4" AS_CUDA_SM="'80;86;89;90a'" AS_PLATFORM="cuda" \
  python3 setup.py bdist_wheel

2. Build python package for x86:

.. code-block:: bash

  cd python
  AS_PLATFORM="x86" python3 setup.py bdist_wheel

3. Build python package for arm:

.. code-block:: bash

  cd python
  AS_PLATFORM="armclang" python3 setup.py bdist_wheel

.. note:: 
  - We use CUDA 12.4 as the default CUDA version. If you want to change to a different version, set ``AS_CUDA_VERSION`` to the target CUDA version.
  - Set ``AS_RELEASE_VERSION`` enviroment variable to change package version.
  - Set ``ENABLE_MULTINUMA=ON`` enviroment variable to enable multi-NUMA inference in CPU-only version.


Build C++ Libraries
,,,,,,,,,,,,,,,,,,,

1. Build C++ libraries for CUDA

.. code-block:: bash

  AS_CUDA_VERSION="12.4" AS_NCCL_VERSION="2.23.4" AS_CUDA_SM="'80;86;89;90a'" AS_PLATFORM="cuda" AS_BUILD_PACKAGE="ON" ./build.sh


2. Build C++ libraries for x86

.. code-block:: bash

  AS_PLATFORM="x86" AS_BUILD_PACKAGE="ON" ./build.sh

3. Build C++ libraries for arm

.. code-block:: bash

  export ARM_COMPILER_ROOT=/opt/arm/arm-linux-compiler-24.04_RHEL-8/   # change this path to your own
  export PATH=$PATH:$ARM_COMPILER_ROOT/bin

  AS_PLATFORM="armclang" AS_BUILD_PACKAGE="ON" ./build.sh

4. Build C++ libraries for macos (M1-M3)

 - install openmp

  download openmp and copy to /usr/local/ by this site: https://mac.r-project.org/openmp/

   make sure you get this message in cmake step:

  -- Found OpenMP_C: -Xclang -fopenmp (found version "5.0")
  -- Found OpenMP_CXX: -Xclang -fopenmp (found version "5.0")
  -- Found OpenMP: TRUE (found version "5.0")

.. code-block:: bash
  mkdir build && cd build
  conan install ../conan/conanfile.txt -b missing -b protobuf -b gtest -b glog
  . activate.sh

  # if build with armv9 (Apple M4)
  # cmake ../ -DCONFIG_ACCELERATOR_TYPE=NONE -DALLSPARK_CBLAS=None -DCONFIG_HOST_CPU_TYPE=ARM -DBUILD_PYTHON=OFF -DENABLE_CUDA=OFF -DENABLE_ARM_V84_V9=ON -DENABLE_SPAN_ATTENTION=OFF
  # build with armv8 (Apple M1-M3)
  cmake ../ -DCONFIG_ACCELERATOR_TYPE=NONE -DALLSPARK_CBLAS=None -DCONFIG_HOST_CPU_TYPE=X86 -DBUILD_PYTHON=OFF -DENABLE_CUDA=OFF -DENABLE_SPAN_ATTENTION=OFF
  make -j8

  # running c++ benchmark.
  export DYLD_LIBRARY_PATH=`pwd`/lib:$DYLD_LIBRARY_PATH

  # put a serialized model on ../model_output
  bin/model_stress_test  -t 2000 -f 1 -r 10000 -b 1  -N 1 -l 512 -C 1 -d  ../model_output/   -m qwen_Qwen2-7B-Instruct

Profiling
---------

Operator Profiling
==================

This section describes how to enable and utilize the operator profiling functionality.

1. Enable OP profiling data collection

To enable OP profiling, set the environment variable ``AS_PROFILE=ON`` before running DashInfer.

.. code-block:: bash

   export AS_PROFILE=ON
   # Then, run any Python program utilizing the DashInfer Engine.


2. Print OP pro

To view the profiling information, call the following function before deinitializing the engine:

.. code-block:: bash

      print(engine.get_op_profiling_info(model_name))

.. tip:: Replace *model_name* with the name of your model.


3. Analyze OP profiling data

   An OP profiling data report begins with a section header marked by \*\*\* <section> \*\*\* followed by a detailed table. The report consists of three main sections:

   - reshape: Statistics on the cost of reshaping inputs for operators.
   - alloc: Measures the cost of memory allocation for paged KV cache.
   - forward: Focuses on the execution time of operators' forward passes; developers should closely examine this section.

   Below is an illustration of the table structure and the meaning of each column:

   1. **opname**: The name of the operator.
   2. **count**: The number of times the operator was invoked during profiling.
   3. **(min/max/ave)**:  Minimum, maximum, and average execution times in milliseconds.
   4. **total_ms**: The cumulative time spent on this operator.
   5. **percentage**: The operator's total time as a percentage of the overall profiling duration.

   An example snippet of the profiling output is shown below:

.. code-block:: bash

  *** forward ***
  -----------------------------------------------------------------------------------------------
  rank      opname              count     min_ms    max_ms    ave_ms    total_ms       percentage
  -----------------------------------------------------------------------------------------------
  0         Gemm                423       0.04      16.80     3.83      1622.09        69.30
  0         DecOptMQA           84        0.10      22.91     7.63      640.81         27.38
  0         RichEmbedding       3         0.00      23.10     7.70      23.10          0.99
  0         LayerNormNoBeta     171       0.01      0.32      0.11      19.18          0.82
  0         Rotary              84        0.02      0.57      0.20      16.72          0.71
  0         Binary              84        0.01      0.50      0.17      14.46          0.62
  0         AllReduce           171       0.01      0.02      0.01      1.66           0.07
  0         PostProcessId       3         0.27      0.34      0.30      0.91           0.04
  0         AllGather           3         0.03      0.55      0.21      0.62           0.03
  0         UpdateId            4         0.08      0.15      0.11      0.44           0.02
  0         GenerateOp          3         0.13      0.15      0.14      0.42           0.02
  0         EmbeddingT5         3         0.02      0.31      0.11      0.34           0.01
  0         PreProcessId        1         0.03      0.03      0.03      0.03           0.00
  0         GetLastLine         3         0.01      0.01      0.01      0.02           0.00
  0         TransMask           1         0.00      0.00      0.00      0.00           0.00
  -----------------------------------------------------------------------------------------------

From the provided forward operator profiling data, several key observations can be made:

1. Dominant Operators: The Gemm operator stands out as the most significant performance factor, accounting for 69.30% of the total execution time despite being called 423 times. Its high average time of 3.83ms indicates that optimizing this operator could lead to substantial performance improvements.

2. Second Heaviest Operator: DecOptMQA, although called less frequently (84 times), contributes to 27.38% of the total runtime with a relatively high average time of 7.63ms. This operator is also a prime candidate for optimization efforts.

3. Low Frequency, High Variance: The RichEmbedding operator, though called only 3 times, shows a wide range in execution times (from 0.00 to 23.10ms) with an average of 7.70ms. This suggests potential variability or inefficiencies that might warrant further investigation.

Some notes about operator:
,,,,,,,,,,,,,,,,,,,,,,,,,,,

1. Gemm: inlcude all Gemm/Gemv operator in model.
2. DecOptMQA: this is the attention operator in model.
3. AllGather/AllReduce: this is the collective commucation operator.

Nsys Decoder and Context Loop Profiling
=======================================

This section describes how to use controlled Nsys profiling to obtain decoder and context loop profiling data. This method profiles only when enabled, preventing the creation of excessively large Nsys profile files.

**Steps:**

0. **Disable Warm-up:** Set the environment variable `ALLSPARK_DISABLE_WARMUP=1` to disable the warm-up phase.
1. **Enable Nsys Profiling Call:** Set ``#define ENABLE_NSYS_PROFILE 1`` in file `cuda_context.cpp`.
2. **Model.cpp Configuration:**
    - **Context Phase Profiling:** To profile the context phase, set ``#define PROFILE_CONTEXT_TIME_GPU 1`` in file `model.cpp`. This will initiate Nsys profiling on the 10th request and terminate the process after one context loop completes.
    - **Generation Phase Profiling:** To profile the generation phase, set ``#define PROFILE_GENERATION_TIME_GPU 1`` in file `model.cpp`. Profiling will commence after reaching a concurrency (or batch size) specified by `PROFILE_GENERATION_TIME_BS` (adjust this value according to your needs). This allows you to profile the system under a fixed concurrency level.
3. **ReCompile:** Recompile your package and install
4. **Start Profiling:**  Execute your benchmark or server using the following command:

.. code-block:: bash

  nsys profile -c cudaProfilerApi xxx_benchmark.py

.. Note:: Replace `xxx_benchmark.py` with the actual name of your benchmark or server script.


Coding Style
-------------

Before submitting code, there will be a coding style validation. Ensure you use the same version of tools as CI.


.. code-block:: bash

  pip install clang-format==17.0.6

Once the local code has been checked in, use

.. code-block::

  ./scripts/clang-format/clang-format-apply.sh

to correct the code style. For example, if multiple commits were submitted, and the origin commit is `badbeef`, call:

.. code-block::

  ./scripts/clang-format/clang-format-apply.sh badbeef

to automatically correct the style in between.

The *.clang-format* file stores the project's style configuration. You can configure this hook for automatic invocation. If formatting discrepancies appear in multiple submissions when applying for a review, add the following line to this file:

.. code-block:: bash

  ./scripts/clang-format/clang-format-apply.sh HEAD^
