========================
Weight Quantization
========================


Overview
--------


DashInfer has a range of quantization techniques for LLM weight, including int{8,4} weight only, and int8 activatation quantization. We also have many customized fused kernels to get the best performance on specified device.

In a nutshell, models fine-tuned with GPTQ will give you better accuracy, but our InstantQuant (IQ) technique, which does not require fine-tuning and can offer a faster deployment experience. You can find more details on IQ quantization at the end of this article.

In terms of supported quantization algorithms, DashInfer supports models fine-tuned with GPTQ, or dynamic quantization using the IQ quantization technique:

- **IntantQuant(IQ)**: DashInfer provides the InstantQuant (IQ) dynamic quantization technique, which does not require fine-tuning and can offer a faster deployment experience.
- **GPTQ**: Models fine-tuned with GPTQ will provide better accuracy, but it requires a fine-tuning step.

The quantization strategies introduced here can be broadly divided into two categories:

- **Weight Only Quantization**: This quantization technique only quantizes and compresses the weights, such as storing weights in int8 format, but uses bf16/fp16 for computations. It reduces memory access requirements, without improving computational performance compared to BF16.
- **Activation Quantization**: This quantization technique not only stores weights in int8 format but also performs low-precision quantized computations (such as int8) during computation. With NVIDIA GPUs' int8 Tensor Cores, this quantization technique can reduce memory access requirements and improve computational performance, making it a more ideal quantization approach. In terms of accuracy, it may have a slight decrease compared to Weight Only quantization, so accuracy testing may be required.


In terms of quantization granularity, there are two types:

- **Per-Channel**: DashInfer's quantization techniques at least adopt the Per-Channel (also known as Per-Token) quantization granularity, and some also provide Sub-Channel quantization granularity. Generally speaking, Per-Channel quantization can meet most accuracy requirements due to its simple implementation and optimal performance. Only when the accuracy of Per-Channel quantization is insufficient should the Sub-Channel quantization strategy be considered.
- **Sub-Channel**: Compared to Per-Channel quantization, Sub-Channel refers to dividing a channel into N groups, and calculating quantization parameters within each group. This quantization granularity typically provides better accuracy, but due to increased implementation complexity, it comes with many limitations. For example, performance may be slightly slower than Per-Channel quantization, and Activation quantization is difficult to implement Sub-Channel quantization due to computational formula constraints (DashInfer's Activation Quantization is all Per-Channel).

.. note::

      Activation quantization mode only supports Per-Channel quantization strategy.

In terms of quantization computation methods, there are asymmetric and symmetric quantization. Asymmetric quantization generally provides better accuracy, and DashInfer's weight-only quantization uses asymmetric quantization. However, due to implementation limitations, Activation quantization can only be implemented with symmetric quantization.

For the following descriptions of quantization accuracy, we use the naming rules as follows:

1. We use *AxWy* (eg. A16W8) to represent computations with x-bit activations and y-bit weights.
2. 8-bit typically uses int8 (on GPUs) or uint8 (on CPUs), and 4-bit typically uses uint4.

Hardware Support Status
-----------------------

.. list-table:: Support Status
   :widths: 10 8 8 8 8 8 8 8 8 8 8 8
   :header-rows: 1

   * - Implementation
     - Weight-Only
     - Act. Quant
     - PerChn
     - SubChn
     - Volta
     - Turing
     - Ampere
     - Ada
     - Hooper
     - X86
     - ARMv9
   * - IQ-INT8(A16W8)
     - ⬛
     - ⬜
     - ⬛
     - ⬛
     - ✅︎
     - ✅
     - ✅︎
     - ✅︎
     - ✅︎
     - ✗
     - ✅︎
   * - IQ-INT4(A16W4)
     - ⬛
     - ⬜
     - ⬛
     - ⬛
     - ✅︎
     - ✅
     - ✅︎
     - ✅︎
     - ✅︎
     - x
     - x
   * - IQ-INT8(A8W8)
     - ⬜
     - ⬛
     - ⬛
     - ⬜
     - x
     - ✅
     - ✅︎
     - ✅︎
     - ✅︎
     - x
     - x
   * - GPTQ-INT8(A16W8)
     - ⬛
     - ⬜
     - ⬛
     - ⬛
     - ✅
     - ✅
     - ✅︎
     - ✅︎
     - ✅︎
     - x
     - x

   * - GPTQ-INT8(A8W8)
     - ⬜
     - ⬛
     - ⬛
     - ⬛
     - x
     - ✅
     - ✅︎
     - ✅︎
     - ✅︎
     - x
     - x
   * - GPTQ-INT4(A16W4)
     - ⬛
     - ⬜
     - ⬛
     - ⬛
     - ✅︎
     - ✅
     - ✅︎
     - ✅︎
     - ✅︎
     - x
     - x

Notes:
^^^^^^

- NVIDIA GPUs are represented as Volta (SM 7.0), Turing (SM 7.5), Ampere (SM 8.0/8.6), Ada Lovelace (SM 8.9), and Hopper (SM 9.0).
- A filled square (⬛) indicates the corresponding column's attribute is enabled, while an empty square (⬜) indicates it's disabled.
- A checkmark (✅) indicates that the quantization method is supported on the specified hardware.
- A cross mark (✗) indicates that the quantization method is not supported on the specified hardware.
- Volta lacks int8 Tensor Core, thus all activation quantization options are disabled.

API and Config
--------------

GPTQ Model
^^^^^^^^^^
GPTQ models support automatic adaptation of quantization parameters, currently supporting A8W8, A16W8, and A16W4 inference methods. Support for A8W4 is under development.

For GPTQ models, quantization can be enabled during serialization using the ``model_loader.serialize()`` interface by setting the ``enable_quant`` parameter to ``True``.  When enabled, the engine reads the ``quantization_config`` section within the model's ``config.json`` file to determine the appropriate quantization kernels and other configurations like ``group_size``.

Regarding computation modes, the default quantization strategy is Activation quantization.  For instance, with 8-bit models, A8W8 Activation quantization is the default. To utilize Weight-Only quantization, pass the ``weight_only_quant=True`` argument to the ``serialize()`` interface. Refer to the examples below for specific usage.

You can modify the lines in :ref:`gptq-hf-model-examples` in the complete source code to perform inference with the int8 qwen2 model.


.. note:: Currently, the ``desc_act`` parameter within the GPTQ configuration is not supported when set to ``true``. Only ``false`` is supported.


Instant Quant (IQ) Support
^^^^^^^^^^^^^^^^^^^^^^^^^^

Instant Quant, as the name suggests, mainly provides a dynamic quantization technique for models that haven't undergone quantization fine-tuning. This is similar to the quantization kernels supported by many open-source engines like vLLM marlin. For quantization parameter calculation, the quantization algorithm introduced in :ref:`instant_quant_detail` is used.
We have developed specialized kernels for different devices to optimize inference performance.


IQ-Weight Only
~~~~~~~~~~~~~~
This quantization technique only quantizes and compresses the weights, such as storing weights in int8 format, but uses bf16/fp16 for computations. It only reduces memory access requirements, without improving computational performance compared to BF16.
It is generally more suitable for scenarios where memory is limited or where there is a need to compress the memory footprint of weights without sacrificing inference accuracy.

Example: Refer to :ref:`iq_example_prefix`, { :ref:`iq_example_a16w8`, :ref:`iq_example_a16w4` }, and :ref:`iq_example_suffix` for implementing different weight quantizations.


IQ-Activate Quant
~~~~~~~~~~~~~~~~~

This quantization technique not only stores weights in int8 format but also performs low-precision quantized computations (such as int8) during the calculation phase. With NVIDIA GPUs' int8 Tensor Cores, this quantization technique can reduce memory access requirements and improve computational performance, making it a more ideal quantization approach. In terms of accuracy, it may have a slight decrease compared to Weight Only quantization, so accuracy testing may be required.

Since the IQ technique uses a BF16 model, the inference accuracy will decrease after Activate Quantization. However, due to the computational performance improvement, this quantization approach has significant advantages for larger input lengths.

Example: Refer to :ref:`iq_example_prefix`, :ref:`iq_example_a8w8`, and :ref:`iq_example_suffix`.

.. note:: Direct IQ A8W8 may drop accuracy. If the accuracy loss is unacceptable, it is recommended to use GPTQ fine-tuned or SmoothQuant fine-tuned models for Activation quantization.

Examples
--------

.. _gptq-hf-model-examples:

GPTQ HF model Examples
^^^^^^^^^^^^^^^^^^^^^^
**A8W8 GPTQ:**

.. code-block:: python

    ...
    modelscope_name = "qwen/Qwen2-7B-Instruct-GTPQ-Int8"
    ...
    model_loader.load_model().serialize(
        engine, model_output_dir=tmp_dir,
        enable_quant=True).free_model()

**A16W8 GPTQ (Weight-Only):**

.. code-block:: python

    ...
    modelscope_name = "qwen/Qwen2-7B-Instruct-GTPQ-Int8"
    ...
    model_loader.load_model().serialize(
        engine, model_output_dir=tmp_dir,
        enable_quant=True, weight_only_quant=True).free_model()



Customized Quantization Examples
--------------------------------

Cutomized Quantization support can use different quantization combo.


You can paste :ref:`iq_example_prefix`, `Quant Config`, and :ref:`iq_example_suffix` parts together to get a full example.

.. _iq_example_prefix:

IQ Example Common Prefix
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import os
    import modelscope
    from modelscope.utils.constant import DEFAULT_MODEL_REVISION

    from dashinfer import allspark
    from dashinfer.allspark.engine import TargetDevice

    # if use in memory serialize, change this flag to True
    in_memory = True
    device_list=[0]

    modelscope_name ="qwen/Qwen2-7B-Instruct"
    ms_version = DEFAULT_MODEL_REVISION
    output_base_folder="output_qwen"
    model_local_path=""
    tmp_dir = "model_output"


    model_local_path = modelscope.snapshot_download(modelscope_name, ms_version)
    safe_model_name = str(modelscope_name).replace("/", "_")

    model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name, in_memory_serialize=in_memory, trust_remote_code=True, user_set_data_type="float16")
    engine = allspark.Engine()

    model_convert_folder = os.path.join(output_base_folder, safe_model_name)

.. _iq_example_suffix:

IQ Example Common Suffix
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # paste your quant config here.
    # like a16w8, replace this config with other quant config.
    # simplied quant config with per-channel int8
    simpled_a16w8_per_channel_customized_quant_config = {
        "quant_method": "instant_quant",
        "weight_format": "int8"}
    my_quant_config = simpled_a16w8_per_channel_customized_quant_config

    model_loader.load_model().serialize(
        engine, model_output_dir=tmp_dir,
        enable_quant=True,
        customized_quant_config=my_quant_config).free_model()

    # change runtime config in this builder.
    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA, device_list, max_batch=8)
    # like change to engine max length to a smaller value
    runtime_cfg_builder.max_length(2048)

    runtime_cfg = runtime_cfg_builder.build()

    # install model to engine
    engine.install_model(runtime_cfg)

    model_loader.free_memory_serialize_file()

    # start the model inference
    engine.start_model(safe_model_name)

    input_str= "How to protect our planet and build a green future? "

    messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_str}
                ]
    templated_input_str = model_loader.init_tokenizer().get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # generate a reference generate config.
    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    # change generate config base on this generation config, like change top_k = 1
    gen_cfg.update({"top_k": 1})
    #gen_cfg.update({"eos_token_id", 151645})
    status, handle, queue = engine.start_request_text(safe_model_name, model_loader, templated_input_str, gen_cfg)

    generated_ids = []

    status = queue.GenerateStatus()

    ## in following 3 status, it means tokens are generating
    while (status == allspark.GenerateRequestStatus.Init
           or status == allspark.GenerateRequestStatus.Generating
           or status == allspark.GenerateRequestStatus.ContextFinished):
        elements = queue.Get()
        if elements is not None:
            generated_ids += elements.ids_from_generate
        status = queue.GenerateStatus()
        if status == allspark.GenerateRequestStatus.GenerateFinished:
            break
            # This means generated is finished.
        if status == allspark.GenerateRequestStatus.GenerateInterrupted:
            break
            # This means the GPU has no available resources; the request has been halted by the engine.
            # The client should collect the tokens generated so far and initiate a new request later.
    # de-tokenize id to text
    output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
    print("---" * 20)
    print(
        f"test case: {modelscope_name} input:\n{input_str}  \n output:\n{output_text}\n")
    print(f"input token:\n {model_loader.init_tokenizer().get_tokenizer().encode(templated_input_str)}")
    print(f"output token:\n {generated_ids}")

    # after all inference is done, stop and release model
    engine.stop_model(safe_model_name)
    engine.release_model(safe_model_name)

    # after this release model, all resource have been freeed, so another model can be serverd.

.. _iq_example_a16w8:

IQ A16W8
^^^^^^^^

- Per-Channel A16W8：

.. code-block:: python

    # simplied quant config with per-channel int8
    simpled_a16w8_per_channel_customized_quant_config = {
        "quant_method": "instant_quant",
        "weight_format": "int8"}
    my_quant_config = simpled_a16w8_per_channel_customized_quant_config

- Sub-Channel (128) A16W8:

.. code-block:: python

    # sub-channel with group 128 instant quant with int8
    # notice: with different TP, you may find larger gorup size will be report error,
    # pre-channel quant mode don't have such issue and will provide best inference performance
    simple_a16w8_group128_customized_quant_config = {
        "quant_method": "instant_quant",
        "weight_format": "int8",
        "group_size": 128}

    my_quant_config = simple_a16w8_group128_customized_quant_config


.. _iq_example_a16w4:

IQ A16W4
^^^^^^^^

A16W4 only supports pre-channel.

.. code-block:: python

    # per-channel int4 instant quant config.
    simpled_a16w4_customized_quant_config = {
        "quant_method": "instant_quant",
        "weight_format": "uint4"}


.. _iq_example_a8w8:

IQ A8W8
^^^^^^^

As described in pervious section, A8W8 means both activation and weight use int8 format,
and it will make better use of computation power since it use int8 Tensor Cores (GPU only) to perform the GEMM/GEMV computation.

.. code-block:: python

    simpled_a8w8_customized_quant_config = {
        "quant_method": "instant_quant",
        "weight_format": "int8",
        "compute_method" : "activate_quant"}
    my_quant_config = simpled_a8w8_customized_quant_config



.. _instant_quant_detail:

Instant Quant Detail
--------------------

CUDA
^^^^

Symmetric quantization can be viewed as a special case of asymmetric quantization with zero_point set to 0. In general, the accuracy of asymmetric quantization is higher than that of symmetric quantization.

The computation formulas for symmetric and asymmetric quantization are as follows:

- Synmmetric Detail


.. figure:: ../_static/symmetric_quant.png
   :width: 640
   :align: center
   :alt: Illustration of symmetric and asymmetric quantization


.. math::

   scale = \frac{|F_{max}|}{|Q_{max}|} \\
   Q = \frac{F}{scale} \\

- Asynmmetric Detail

.. figure:: ../_static/asymmetric_quant.png
   :width: 640
   :align: center


.. math::
   scale = \frac{F_{max} - F_{min}}{Q_{max} - Q_{min}} \\
   zero\_point = Q_{min} - \frac{F_{min}}{scale} \\
   Q = \frac{F}{scale} + zero\_point \\

Where:

- F represents the floating-point tensor of FP16/BF16 type
- Q represents the integer tensor obtained after quantization
- Qmax and Qmin are the upper and lower bounds of the data representation range for the integer type
- Fmax and Fmin are the maximum and minimum values in the floating-point tensor data
- scale and zero_point are the quantization parameters required for the linear mapping of the floating-point tensor to the integer tensor

CPU
^^^

- Uint8 Asymmetric Quantization:

 We use uint8 as type on CPU devices which provide high performance kernel, as following:

.. math::

   scale = \frac {x_{fp32_{max}} - x_{fp32_{min}}} {255 - 0}

   zeropoint = 0 - \frac {x_{fp32_{min}}} {scale}

   x_{u8} = x_{fp32} / scale + zeropoint
