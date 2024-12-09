=====================
Engine Runtime Config
=====================

The runtime configuration allows you to set various options for the model inference, such as the maximum batch size, maximum sequence length, and cache modes. You can use the ``AsModelRuntimeConfigBuilder`` class to create and configure the runtime settings.

1. Use model loader's helper funtion to create a runtime config; it will fill all necessary fileds, and you can modify based on this builder.
2. Directly use builder to create; this will require you to fill all necesary fileld like ``model_name`` and paths.
3. You can use a prefilled python dict, and use builder's ``from_dict`` to update or create a builder.

Model Configuration
-------------------

- ``model_name(model_name: str)``: Sets the name of the model.
- ``model_dir(model_dir, file_name_prefix)``: Sets the model file path and weights file path based on the provided directory and file name prefix.
- ``model_file_path(model_file_name, weight_file_path)``: This will set the model's graph and model's weight in sepreated way, not recommended.

Compute Unit
------------

- ``compute_unit(target_device: TargetDevice, device_id_array=None, compute_thread_in_device: int = 0)``: Setup the runtime compute unit. The `target_device` parameter can be set to `CUDA`, `CPU`, or `CPU_NUMA`.

- For CUDA, you can specify the GPU device IDs in `device_id_array`.

- For CPU, `compute_thread_in_device` specifies the number of compute threads to use during inference (0 for auto-detection).

- For CPU_NUMA, `device_id_array` specifies the NUMA node IDs, and `compute_thread_in_device` specifies the compute threads inside each NUMA node.

Compute Unit Examples
^^^^^^^^^^^^^^^^^^^^^

Some examples as follows:

CUDA
....

1. CUDA: Single Card :

.. code-blocK:: python

    runtime_builder(safe_model_name, TargetDevice.CUDA, [0], max_batch=64)

2. CUDA: 2 Cards:

.. code-blocK:: python

    runtime_builder(safe_model_name, TargetDevice.CUDA, [0, 1], max_batch=64)

3. CUDA: 2 Cards with specifiy IDs (2nd card and 4th card):

.. code-blocK:: python

    runtime_builder(safe_model_name, TargetDevice.CUDA, [1, 3], max_batch=64)

CPU
...

1. CPU with Single NUMA

Automatically choose compute thread number.

.. code-blocK:: python

    runtime_builder(safe_model_name, TargetDevice.CPU, [0], max_batch=64)


Manually set compute thread; usually number should be equal or less than phyiscal core number.

.. code-blocK:: python

    runtime_builder(safe_model_name, TargetDevice.CPU, [0], max_batch=64).compute_unit(TargetDevice.CPU, compute_thread_in_device=32)

Sequence Length and Batch Size
------------------------------

- ``max_length(length: int)``: Sets the maximum sequence length for the engine.
- ``max_batch(batch: int)``: Sets the maximum batch size for the engine.
- ``max_prefill_length(length: int)``: Sets the maximum prefill length that will be processed in one context inference; if input length is greater than
  this length, it will be process in multiple context inference steps.

Prefix Cache Configuration
--------------------------

- ``prefill_cache(enable=True)``: Enables or disables the prefix cache.
- ``prefix_cache_ttl(ttl: int)``: Prefix cache time to live, default value is 300s.

KV Cache Quantization Configuration
-----------------------------------

``kv_cache_mode(cache_mode: AsCacheMode)``: Sets the cache mode for the key-value cache. The `AsCacheMode` enum provides three options: `AsCacheDefault`, `AsCacheQuantI8`, and `AsCacheQuantU4`.

- `AsCacheDefault`: will keep the same data type as model infernece, usually it means a BF16/FP16 stored KV-Cache.

- `AsCacheQuantI8`: will quantize kv-cache into int8 type, this will reduce kv-cache memory footprint in half (compared to bf16).

- `AsCacheQuantU4`: will quantize kv-cache into uint4 type, this will reduce kv-cache memory footprint in 1/4 (compared to bf16).

This config does not depend on weight quantizaion, and it can be switched on/off independently.

Utility Functions
-----------------

- ``from_dict(rfield)``: Sets the runtime configuration from a dictionary.
- ``build()``: Builds and returns the `AsModelConfig` object.

Usage Example
-------------

Here's an example of how to configure and use the runtime settings:

.. code-block:: python

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA,
                                                                                device_list, max_batch=1)
    # Change the maximum sequence length
    runtime_cfg_builder.max_length(set_engine_max_length)
    runtime_cfg_builder.prefill_cache(set_prefill_cache)

    # Enable int8 or int4 key-value cache quantization
    if cache_quant_mode != "16":
        if cache_quant_mode == "8":
            runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8)
        elif cache_quant_mode == "4":
            runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)

    runtime_cfg = runtime_cfg_builder.build()

    # Install the model into the engine
    engine.install_model(runtime_cfg)

In this example, we first create a ``AsModelRuntimeConfigBuilder`` instance using the ``create_reference_runtime_config_builder`` method from the ``model_loader``. We then set the desired maximum sequence length, enable or disable the prefix cache, and configure the key-value cache quantization mode (int8 or int4) if needed.