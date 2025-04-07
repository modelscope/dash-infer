=====================================
Offline Inference with Python API
=====================================

We have presented a quick start example of LLM inference with Python API in
:doc:`../get_started/quick_start_api_py_en`.

********
Concepts
********

In this example, there is some explanation about how the inference work and the main components. In this documentation, we will have a deep dive into APIs and how to enable different features.

----------------
Main Components
----------------


There is three key components, **Engine**, **Model**, and **Request**:

* The *Model* will be installed into *Engine* and the engine can be controlled by engine API such as *Start* and *Stop*,  After **engine** start, it can receive input requests (includes input ids and generate config).

* *Request*: After **request** starts, API will provide a handle and a queue, and the request can be controlled by the engine API and handle like *Start* / *Stop* / *Release*.

* *Output*: The output id and status of this request can be get by `queue` object returned by ``start_request_{text|ids}`` API,

* After this request's status changes to ``GenerationFnished`` or ``GenerateInterrupted``, or stopped by ``stop_request``, the resources will be released by engine, except for the `queue`.

* After release request, all resources related to this request will be freed.


----------
Main Steps
----------

First of all, the inference pipeline has the following main steps:

#. **Model Loading**: For Hugging Face models, use this loader: ``dashinfer.allspark.model_loader.HuggingFaceModel``. This loader will parse the parameters of the Hugging Face model and create corresponding parameters for conversion.

#. **Model Serialization**: This process converts the model into DashInfer format, offering two modes. The one is a transparent in-memory conversion that does not generate intermediate files; however, the drawback is that it uses twice the memory. The other converts the model to a local file, which can be loaded later using the DashInfer loader (WIP). The *Weight Quantization* function happens during this step.

#. **Engine Installation and Model Startup**: Once the model is loaded, it is installed in the corresponding engine, which then starts the installed model. It will allocate related resources like GPU/CPU memory. It will also warm up the engine. In this step, the engine will generate some fake requests to ensure that a request with maximum length can be run successfully. After warm-up, there are no significant device memory allocations during subsequent request processing, which enhances serving stability.

#. **Request Initiation and Output Reception**: This step primarily focuses on asynchronously initiating requests. After a request is created, the engine processes it in a continuous batching manner. The corresponding output from the request is obtained through the request's output queue, which also allows asynchronous monitoring of the request's current status.

.. note:: For NVIDIA GPUs, the device memory will be managed by a BFC Allocator. It will allocate the configured total device memory on **Engine Installation and Model Startup** step. The allocation ratio can be controlled by the ``BFC_MEM_RATIO`` environment variable. For example, ``BFC_MEM_RATIO=0.8`` means allocating 80% of all GPU memory on this device (total memory, not free memory). More memory usually means a larger KV-cache pool, which is better for throughput. The default value is ``0.9`` in this release.


.. _llm_offline_interface:

**************************
Explaination with Examples
**************************


------------------------------
Simple Python API Code Snippet
------------------------------

.. code-block:: python

    generated_ids = []

    engine = allspark.Engine()

    model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name)

    (model_loader.load_model()
        .read_model_config()
        .serialize_to_memory(engine)
        .free_model())

    # create engine runtime config.
    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA, device_list, max_batch=8)
    runtime_cfg = runtime_cfg_builder.build()

    # install and start.
    engine.install_model(runtime_cfg)
    engine.start_model(safe_model_name)

    # generation request.
    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)

    status, handle, queue = engine.start_request_text(safe_model_name, model_loader, "who are you?", gen_cfg)


    # sync request will block request until request is finished.
    engine.sync_request(safe_model_name, handle)

    # sync output
    generated_elem = queue.Get()
    generated_ids += generated_elem.ids_from_generate

    # free request and model
    engine.release_request(safe_model_name, handle)
    engine.stop_model(safe_model_name)

---------------
Engine Creation
---------------

Engine object can be created with ``allspark.Engine()``. It will create an Engine instance for later invocation.

.. code-block:: python

  from dashinfer import allspark
  engine = allspark.Engine()

  I20240906 13:28:04.592298 130842 as_engine.cpp:281] AllSpark Init with Version: 3.1.0/(GitSha1:93a8bb12)

After successful initialization, the engine will print its version information. During this step, the engine's required resources will be allocated.

-------------------------------
Model Loading and Serialization
-------------------------------

Before loading model, you need to download the model from model hub (like huggingface or modelscope), or specify your local
model path, which is needed in this step.

You need to choose a model loader for each type of model.
There are two types of supported models:

1. huggingface format model, use ``dashinfer.allspark.model_loader.HuggingFaceModel``
2. dashinfer format model, use ``dashinfer.allspark.model_loader.DashInferModel``, which is the converted DashInfer format model files.

Since DashInfer requires customized weight format (Dashinfer Model), the HuggingFaceModel will require
`serialize` step for the format convert. There are two ways to do such conversion: `file` or `memory`.
If the serialization target is `file`, it will save the new file into a local directory; if the serialization target is `memory`, it will serialize the model to a temp file under ``/tmp``,
and the file will be deleted after process finishes or ``.free_model()`` function is called.

.. note:: some operation system environment will not mount /tmp as memory file system (`tmpfs` in linux), which may cause "No Space in File System" error.

In model loading step, you will get many information like model context length, default generation config, tokenizer etc.

---------------------
Engine Runtime Config
---------------------

The import data structure in this part is a `RuntimeConfig` (aka ``AsModelConfig``). It include which device (`CUDA` or `CPU`), how many devices (`device_list`), and the maximum batch size supported, the maximum token length (input + output) will support, and the model's information such as file path and /or the identifier.

``AsModelRuntimeConfigBuilder`` is the helper class to create `RuntimeConfig`


.. _asmodelruntimeconfigbuilder:

Runtime Config Builder
======================

The ``AsModelRuntimeConfigBuilder`` class is used to configure the runtime settings for a model in the DashInfer engine. It provides a user-friendly Python API for setting various runtime parameters, such as the model path, computation device, maximum batch size, maximum sequence length, and caching modes.

Runtime Configuration
=====================

The runtime configuration includes the following settings:

- **Model Path**: The path to the model file and weights file. User only specify this model localtion when using **serialized model**, HuggingFace Loader will fill this value automiticly.
- **Compute Unit**: The target device for computation, which can be `CUDA`, `CPU`, or `CPU_NUMA`.
- **Thread Number**: The number of threads to use for computation on `CPU` or `CPU_NUMA` devices.
- **Maximum Batch Size**: The maximum batch size for inference.
- **Maximum Sequence Length**: The maximum sequence length for input data.
- **Cache Mode**: The KV cache mode, defualt mode is 16-bit floating point (bfloat16/float16), and can be configured as int8 or uint4 mode.
- **Prefill Cache**: The prompt prefix cache, which can reduce duplicated prefill computation time, and is `ON` by default.

Usage
=====


Here's an example of how to use the ``AsModelRuntimeConfigBuilder``:

.. code-block:: python

    from dashinfer.allspark import *
    from dashinfer.allspark.engine import *

    # Create a new builder instance
    builder = AsModelRuntimeConfigBuilder()

    # Set the model name
    builder.model_name("my_model")

    # Set the model directory and file name prefix
    # User can specify this folder to start running the serialized model.
    builder.model_dir("/path/to/model", "model_prefix")

    # Set the compute unit to use CUDA device 0
    builder.compute_unit(TargetDevice.CUDA, [0])

    # Set the maximum batch size and sequence length
    builder.max_batch(32)
    builder.max_length(2048)

    # Set the cache mode to quantize key-value pairs
    builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)

    # Build the runtime configuration
    runtime_config = builder.build()

    # Use the runtime configuration for inference or other operations

The ``AsModelRuntimeConfigBuilder`` class provides a fluent interface, allowing you to chain method calls together. It also includes several convenient methods for setting the model path and compute unit from different input formats.

For more detailed information on the available methods and their usage, please refer to the docstrings within the class definition.

--------------
Engine Control
--------------

After setup of `runtime_config`, user can call ``engine.install_model()`` function to install or register model into engine with a `model_name`. User can control the model in this engine by this `model_name`, and the `model_name` should be unique in this engine.

Model's running state in engine includes following states:

1. Initial: the state after model installed.
2. Running: the state after calling ``engine.start_model``
3. Stop:   the state after calling ``engine.stop_model``; the model can not receive request, and model executing thread will stop.
4. Released:  the state after calling ``engine.release_model``; all resources will be released.

Most time we only deal with the engine in 'Running' state. Engine can deal with user's request in this state.

------------------------------------------
Text Request and Generation Request Config
------------------------------------------

This section mainly describe how to start a text LLM request,
and how the generation config should be configured.

1. Generate Config Setup.
=========================

GenerationConfig
================

`GenerationConfig` is used to set various control parameters for text generation. It can be built and configured using the ``ASGenerationConfigBuilder`` class. Here are some of the main configuration options:

Sampling Settings
-----------------

- `do_sample` (bool): Whether to enable sampling in generation. Currently, sampling must be enabled.
- `temperature` (float): Temperature for sampling, controlling the randomness in generation.
- `top_k` (int): Top-K sampling parameter, limiting the selection of the next token.
- `top_p` (float): Top-P sampling parameter for nucleus sampling.

Output Control
--------------

- `max_length` (int): Maximum total length of generated text, including both prefill and generation parts.
- `min_length` (int): Minimum length of the generated text. Set to 0 will disable this constraint.
- `early_stopping` (bool): If True, generation stops when the EOS token is encountered.
- `stop_words_ids` (List[List[int]]): A list of word IDs that signal the generation should stop.
- `eos_token_id` (int): ID of the EOS (end of sequence) token, to be specified based on your model.
- `no_repeat_ngram_size` (int): Size of n-grams that should not repeat in the generated text.

Generation Quality Control
--------------------------

- `repetition_penalty` (float): Penalty applied to repeated words.
- `length_penalty` (float): Penalty based on the length of the generated sequence.
- `presence_penalty` (float): Penalty for the presence of certain words in the output.
- `suppress_repetition_in_generation` (bool): If True, uses `presence_penalty` to suppress word repetition.

Other Settings
--------------

- `seed` (int64_t): Seed for random number generation to ensure reproducibility.
- `logprobs` (bool): If True, returns log probabilities of generated tokens. Not supported by some models.
- `top_logprobs` (int): Specifies the number of tokens with log probabilities to return if `logprobs` is True.
- `lora_name` (str): Name of the LoRA adaptation, if applicable.
- `mm_info` (MultiMediaInfo): Multimedia information, specific to certain use cases.
- `response_format` (dict): Dict of arguments for guided decoding.

Using ``ASGenerationConfigBuilder``
-----------------------------------

You can use the ``ASGenerationConfigBuilder`` class to build and configure the `GenerationConfig`. For example:

.. code-block:: python

  builder = ASGenerationConfigBuilder()
  config = (builder.do_sample()
          .max_length(512)
          .temperature(0.7)
          .top_k(50)
          .build())

The ``ASGenerationConfigBuilder`` provides a fluent interface, allowing you to chain method calls to set the desired configuration. It also supports initialization from a Hugging Face `GenerationConfig` instance.


2. Send Request
===============

Here is the documentation for sending requests in English:

2. Send Request
===============

The DashInfer engine provides two main functions for initiating a text generation request: ``start_request_ids`` and ``start_request_text``. These functions allow you to provide input data in the form of token IDs or text strings, respectively, along with the desired generation configuration.

``start_request_ids``
---------------------

.. code-block:: python

    def start_request_ids(self,
                          model_name: str,
                          model: LLM,
                          input_ids: Tensor,
                          generate_config_builder: ASGenerationConfigBuilder):
        """
        Start a generation request with a model and tensor inputs along with a structured generation configuration.

        Args:
            model_name (str): The name of the model installed for text generation tasks.
            model (LLM): The language model instance.
            input_ids (Tensor): Tensor containing the input token IDs for generation.
            generate_config_builder: An instance of the ASGenerationConfigBuilder class.
        Returns:
            tuple: A tuple consisting of:
                - AsStatus: The status of the request as returned by the engine.
                - object: A request handle to track and manage this specific request.
                - ResultQueue: A queue from which to retrieve the results and status updates of the generation process.
        """

This function accepts input token IDs as a PyTorch Tensor or a Python list, along with the configured ``ASGenerationConfigBuilder`` instance. It returns a tuple containing the request status, a handle for the request, and a result queue for retrieving the generated output and monitoring the request status.

``start_request_text``
----------------------

.. code-block:: python

    def start_request_text(self,
                           model_name: str,
                           model: LLM,
                           input_str_or_array,
                           generate_config_builder: ASGenerationConfigBuilder):
        """
        Start Request by model and with text input.

        Args:
            model_name (str): The installed model name.
            model (LLM): The language model instance.
            input_str_or_array (str, List[str]): The input text or an array of input texts.
            generate_config_builder: An instance of the ASGenerationConfigBuilder class.

        Returns:
            tuple: A tuple consisting of:
                - AsStatus: The status of the request as returned by the engine.
                - object: A request handle to track and manage this specific request.
                - ResultQueue: A queue from which to retrieve the results and status updates of the generation process.
        """

This function accepts the input text or an array of input texts, along with the configured ``ASGenerationConfigBuilder`` instance. It tokenizes the input using the model's tokenizer and then initiates the generation request with the input token IDs and the specified generation configuration. The function returns a tuple containing the request status, a handle for the request, and a result queue for retrieving the generated output and monitoring the request status.

Both functions return a request handle and a result queue, which can be used to monitor the request status and retrieve the generated output. The ``ASGenerationConfigBuilder`` class is used to configure the generation parameters, such as the maximum length, sampling settings, and output control options.

3. Stop and Release Request
===========================

These functions are used to manage and control the lifecycle of generation requests. ``stop_request`` allows you to stop a running request, ``release_request`` releases the resources associated with a request, and ``sync_request`` waits for an asynchronous request to complete before returning.


``stop_request``
----------------

.. code-block:: python

    def stop_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Stops a request.

        Args:
            model_name (str): Model name.
            request_handle: Handle for the request.

        Returns:
            AsStatus: Status of the operation.
        """

The ``stop_request`` function stops a previously initiated request. It takes the following arguments:

- ``model_name`` (str): The name of the model associated with the request.
- ``request_handle``: The handle for the request to be stopped.

It returns an ``AsStatus`` object indicating the status of the operation.

``release_request``
-------------------

.. code-block:: python

    def release_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Releases a request's resources.

        Args:
            model_name (str): Model name.
            request_handle: Handle for the request.

        Returns:
            AsStatus: Status of the operation.
        """

The ``release_request`` function releases the resources associated with a request. It takes the following arguments:

- ``model_name`` (str): The name of the model associated with the request.
- ``request_handle``: The handle for the request whose resources need to be released.

It returns an ``AsStatus`` object indicating the status of the operation.


4. Sync Request
===============

``sync_request``
----------------

.. code-block:: python

    def sync_request(self, model_name: str, request_handle) -> AsStatus:
        """
        Waits for the completion of an asynchronous request.

        Args:
            model_name (str): Model name.
            request_handle: Handle for the request.

        Returns:
            AsStatus: Status of the operation.
        """


The ``sync_request`` function waits for the completion of an asynchronous request, this API is optional, the model inference is start asynchronous in engine when ``start_request`` is called. This API mainly for simulate the sync request for user's use case.

It takes the following arguments:

- ``model_name`` (str): The name of the model associated with the request.
- ``request_handle``: The handle for the asynchronous request.

It returns an ``AsStatus`` object indicating the status of the operation.

-----------------
Output and Status
-----------------

ResultQueue
===========

The `ResultQueue` class is designed to generate status and retrieve results from the DashInfer engine.

.. py:class:: ResultQueue

   The ``ResultQueue`` class provides methods to retrieve the generation status, the current generated length, and request statistics. Additionally, it offers three methods for fetching generated tokens from the queue:

   - `Get()` blocks until new tokens are generated.
   - `GetWithTimeout(timeout_ms)` blocks until new tokens are generated or the specified timeout is reached.
   - `GetNoWait()` returns immediately with the generated tokens or `None` if the queue is empty.

   These methods return the generated tokens as Python objects, or `None` if the queue is empty or the timeout is reached.

   The ``GenerateRequestStatus`` enum represents the current status of the generation process. The possible values are:

   .. py:data:: GenerateRequestStatus.Init

      Init status when queue is create.

   .. py:data:: GenerateRequestStatus.ContextFinished

      Status when context (prefill) has been compleled.

   .. py:data:: GenerateRequestStatus.Generating

      Status when request generation is in progress.

   .. py:data:: GenerateRequestStatus.GenerateInterrupted

      Status when engine has no resource to finish this request's generation, usually meaning no device memory available.

   .. py:data:: GenerateRequestStatus.GenerateFinished

      Status when generation is finished, normally meaning EOS token generated, or generated length exceeds engine_max_length.

   .. py:method:: GenerateStatus()

      Get the generation status. This API will not block.

      This method returns the current status of the generation process as an instance of the ``GenerateRequestStatus`` enum. The possible values represent different states of the generation.

      :returns: The current generation status.
      :rtype: GenerateRequestStatus

   .. py:method:: GeneratedLength()

      Get the current generated length, which is the accumulated number of generated tokens.

      :returns: The current generated length.
      :rtype: int

   .. py:method:: RequestStatInfo()

      Get the key-value dictionary of all statistics for this request.

      :returns: A dictionary containing the request statistics.
      :rtype: dict

   .. py:method:: Get()

      Fetches new token(s) from the queue. This method will block until new tokens are generated.

      :returns: The generated tokens, or `None` if the queue is empty.
      :rtype: GeneratedElements

   .. py:method:: GetWithTimeout(timeout_ms)

      Fetches new token(s) from the queue. This method will block until new tokens are generated or the specified timeout (in milliseconds) is reached.

      :param int timeout_ms: The timeout value in milliseconds.
      :returns: The generated tokens, or `None` if the queue is empty or the timeout is reached.
      :rtype: GeneratedElements or None

   .. py:method:: GetNoWait()

      Fetches new token(s) from the queue without blocking. This method returns `None` if no new tokens are available.

      :returns: The generated tokens, or `None` if the queue is empty.
      :rtype: GeneratedElements or None

Here's the documentation for the `GeneratedElements` class. 
This class provides access to the generated tokens, their log probabilities, and other related information produced during the text generation process.

.. py:class:: GeneratedElements

   Generated Token class, contains token(s) and related information. It may contain multiple tokens generated since the last call to `Get` methods.

   .. py:attribute:: ids_from_generate

      Token(s) from this generation.

      :type: list

   .. py:attribute:: log_probs_list

      A probability list for each token, including the `top_logprobs` tokens and their probabilities when generated.

      Dimension: [num_token][top_logprobs], where each token has a pair [token_id, prob].

      :type: list

   .. py:attribute:: token_logprobs_list

      Stores the probability value for each selected token.

      :type: list

   .. py:attribute:: tensors_from_model_inference

      Tensor outputs from model inference.

      :type: list

   .. py:attribute:: prefix_cache_len

      Cached prefix token length.

      :type: int

   .. py:attribute:: prefix_len_gpu

      GPU cached prefix token length.

      :type: int

   .. py:attribute:: prefix_len_cpu

      CPU cached prefix token length.

      :type: int


Here is an example of how to use the ``Get()`` and ``GenerateStatus()``:

.. code-block:: python

    generated_ids = []
    status = queue.GenerateStatus()

    ## in following 3 status, it means tokens are generating
    while (status == GenerateRequestStatus.Init
            or status == GenerateRequestStatus.Generating
            or status == GenerateRequestStatus.ContextFinished):
        elements = queue.Get()
        if elements is not None:
            print(f"new token: {elements.ids_from_generate}")
            generated_ids += elements.ids_from_generate
        status = queue.GenerateStatus()
        if status == GenerateRequestStatus.GenerateFinished:
            break
            # This means generated is finished.
        if status == GenerateRequestStatus.GenerateInterrupted:
            break
            # This means the GPU has no available resources; the request has been halted by the engine.
            # The client should collect the tokens generated so far and initiate a new request later.
