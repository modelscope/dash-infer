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

conan
,,,,,

 + **conan**:  C++ package management tools, can be installed by : `pip install conan==1.60.0`, only 1.60.0 is supported.

 .. note:: if there is any package-not-found issue, please make sure your conan center is available. Reset it with this command: `conan remote add conancenter https://center.conan.io`


Leak check tool
,,,,,,,,,,,,,,,,

+ If want to enabel asan, or tsan, install following packinges:

.. code-block:: shell

  yum install devtoolset-7-libasan-devel devtoolset-7-libtsan-devel


CPU
,,,

For multi-NUMA inference, `numactl`, `openmpi` are required:

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
    --shm-size=8g \
    --network=host \
    --gpus all \
    -v $(pwd):/root/workspace/HIE-AllSpark \
    -w /root/workspace \
    -it registry-1.docker.io/dashinfer/dev-centos7-cu124
  docker exec -it "dashinfer-dev-cu124-${USER}" /bin/bash

- YiTian 710 Develoment

.. code-block:: shell

  docker run -d --name="dashinfer-dev-${USER}" \
    --network=host \
    -v $(pwd):/root/workspace/HIE-AllSpark \
    -w /root/workspace \
    -it registry-1.docker.io/dashinfer/dev-centos8-arm
  docker exec -it "dashinfer-dev-${USER}" /bin/bash


Build from Source Code
======================

.. tip:: Here we use CUDA 12.4 as the default CUDA version. If you want to change to a different version, you can use enviroment variable to control CUDA dependency version.


Python package build
,,,,,,,,,,,,,,,,,,,,

CUDA normal build:

.. code-block:: bash

  cd python
  AS_CUDA_VERSION="12.4" AS_NCCL_VERSION="2.23.4" AS_CUDA_SM="'80;86;89;90a'" AS_PLATFORM="cuda" python3 setup.py bdist_wheel

.. note:: The Python build only performs the `conan install` operation at the first time; subsequent builds will not conduct `conan install`. If you encounter issues, consider using `rm -rf ./python/build/temp.*` to re-run the entire process.

.. note:: Change `AS_RELEASE_VERSION` enviroment var to change package version.

.. note:: To build an x86 or arm CPU only Python package, it's similar to CUDA build, but change the `AS_PLATFORM` environment variable to `x86` or `arm`.



C++ package build
,,,,,,,,,,,,,,,,,,,

1. C++ lib build for CUDA

.. code-block:: bash

  mkdir build;
  AS_CUDA_VERSION="12.4" AS_NCCL_VERSION="2.23.4" AS_CUDA_SM="'80;86;89;90a'" ./build.sh


2. C++ lib build for x86

.. code-block:: bash

  AS_PLATFORM="x86" ./build.sh

3. C++ lib build for armclang

ARM Compile require armcc to archive best performance, setup the compiler in enviroment var.

.. code-block:: bash

  export ARM_COMPILER_ROOT=/opt/arm/arm-linux-compiler-24.04_RHEL-8/   # change this path to your own
  export PATH=$PATH:$ARM_COMPILER_ROOT/bin
  AS_PLATFORM="armclang" ./build.sh

Profiling
---------

Operator Profiling
==================

This section describes how to enable and utilize the operator profiling functionality.

1. Enable OP profile data collection

To enable OP profiling, set the environment variable as follows:

.. code-block:: bash

   export AS_PROFILE=ON
   # Then, run any Python program utilizing the DashInfer Engine.


2. Print OP profile data

   To view the profiling information, insert the following function call before deinitializing the engine, replacing model_name with your actual model's name:

.. code-block:: bash

      print(engine.get_op_profiling_info(model_name))

.. tip:: Replace *model_name* with the name of your model.


3. Analyze OP profile data

   An OP profile data report begins with a section header marked by ***** <section> ***** followed by a detailed table. The report consists of three main sections:

   - reshape: Statistics on the cost of reshaping inputs for operators.
   - alloc: Measures the cost of memory allocation for paged KV cache.
   - forward: Focuses on the execution time of operators' forward passes; developers should closely examine this section.


   Below is an illustration of the table structure and the meaning of each column:

   1. **opname**: The name of the operator.
   2. **count**: The number of times the operator was invoked during profiling.
   3. **(min/max/ave)**:  Minimum, maximum, and average execution times in milliseconds.
   4. **total_ms**: The cumulative time spent on this operator.
   5. **percentage**: The operator's total time as a percentage of the overall profiling duration.
   6. **rank**: This column is deprecated.

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
1. **Enable Nsys Profiling Call:** In the file `cuda_context.cpp`, uncomment line 14 to enable the Nsys profiling call.
2. **Model.cpp Configuration:**
    - **Context Phase Profiling:** To profile the context phase, set the variable `PROFILE_CONTEXT_TIME_GPU` to `1`. This will initiate Nsys profiling on the 10th request and terminate the process after one context loop completes.
    - **Generation Phase Profiling:** To profile the generation phase, set the variable `PROFILE_GENERATION_TIME_GPU` to `1`. Profiling will commence after reaching a concurrency (or batch size) specified by `PROFILE_GENERATION_TIME_BS` (adjust this value according to your needs). This allows you to profile the system under a fixed concurrency level.
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
