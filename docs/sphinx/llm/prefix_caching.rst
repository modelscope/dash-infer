=====================
Prefix Caching
=====================

What is Prefix Caching
**********************

Prefix caching stores kv-caches in GPU or CPU memory for extended periods to reduce redundant calculations. When a new prompt shares the same prefix as a previous one, it can directly use the cached kv-caches, avoiding unnecessary computation and improving performance.

Enable Prefix Caching
*********************

Runtime Configuration
---------------------

- ``prefill_cache(enable=True)``: Enables or disables the prefix cache, default value is True.
- ``prefix_cache_ttl(ttl: int)``: Prefix cache time to live, default value is 300s.

Environment Variable
--------------------

- ``CPU_CACHE_RATIO``
    - Description: DashInfer will set CPU_CACHE_RATIO * 100% of the current remaining CPU memory for kv-cache storage, and when CPU_CACHE_RATIO=0, no CPU memory is used to store kv cache.
    - Data type: float
    - Default value: ``0.0``
    - Range: float value between [0.0, 1.0]


Performance
***********

Run `benchmark_throughput.py` in `examples/benchmark` by following command:

.. code-block:: shell

    model=qwen/Qwen2-7B-Instruct && \
    python3 benchmark_throughput.py --model_path=${model} --modelscope \
    --engine_max_batch=1 --engine_max_length=4003 --device_ids=0 \
    --test_qps=250 --test_random_input --test_sample_size=20 --test_max_output=3 \
    --engine_enable_prefix_cache --prefix_cache_rate_list 0.99,0.9,0.6,0.3

On Nvidia-A100 GPU we get following result:

.. csv-table::

    Batch_size,Request_num,In_tokens,Out_tokens,Avg_context_time(s),Avg_generate_time(s),Prefix_Cache(hit rate)
    1,20,4000,3,0.030,0.040,96.0%
    1,20,4000,3,0.044,0.040,89.6%
    1,20,4000,3,0.121,0.040,57.6%
    1,20,4000,3,0.185,0.040,28.8%
    1,20,4000,3,0.254,0.040,0.0%
