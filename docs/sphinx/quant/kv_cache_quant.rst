========================
KV Cache Quantization
========================

Overview
--------
Key-Value (KV) cache quantization is an important aspect of efficient large language model (LLM) inference. The importance of KV cache quantization lies in its potential to reduce memory consumption and improve runtime performance, especially for larger sequence lengths and batch sizes. We use the same quantization method as IQ Weight quantization.

Config and Usage
----------------
The KV cache quantization feature is controlled by the ``kv_cache_mode`` function in the :doc:`../llm/runtime_config`:

``kv_cache_mode(cache_mode: AsCacheMode)``: Sets the cache mode for the key-value cache. The `AsCacheMode` enum provides three options: `AsCacheDefault`, `AsCacheQuantI8`, and `AsCacheQuantU4`.

- `AsCacheDefault`: will keep the same data type as model infernece, usually it means a BF16/FP16 stored KV-Cache.
- `AsCacheQuantI8`: will quantize kv-cache into int8 type, this will reduce kv-cache memory footprint in half (compared to bf16).
- `AsCacheQuantU4`: will quantize kv-cache into uint4 type, this will reduce kv-cache memory footprint in 1/4 (compared to bf16).


Example
-------

You can modify one line to enable this feature in :doc:`../get_started/quick_start_api_py_en` :

.. code-block:: python

  # insert this code in runtime cfg builder part.
  runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8) # for int8
  # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4) # for uint4

