FAQ
===

How to enable serialization of models in memory?
------------------------------------------------

During the model loading stage, use the ``serialize_to_memory`` interface for loading.

.. code-block:: python

    (model_loader.load_model()
     .read_model_config()
     .serialize_to_memory(engine, enable_quant=init_quant, weight_only_quant=weight_only_quant)
     .free_model())

Note that after installing the model, you can call the ``model_loader.free_memory_serialize_file()`` interface to release memory, which will also be released when the process exits.

How to initiate a request with converted IDs (non-text)?
--------------------------------------------------------

The example shows how to initiate a text request, but in many scenarios, users need to initiate requests directly with converted IDs. In such cases, you can use the following interface to initiate the request:

.. code-block:: python

    # Just an example of how to tokenize your text
    tokenizer = model_loader.init_tokenizer().get_tokenizer()
    encode_ids = tokenizer.encode(input_str)
    # Send an ID request
    status, handle, queue = engine.start_request_ids(safe_model_name, model_loader, encode_ids, gen_cfg)

How to set up multi-GPU?
------------------------

When generating the runtime config, set multi-GPU with ``runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA, device_list, max_batch=8)``.

How to switch the precision of model execution, e.g., float16 and bfloat16?
-----------------------------------------------------------------------------------

In ``HuggingFaceModel``, use ``user_set_data_type`` to set the model type, and the engine will convert the model's weights to the corresponding data type.

For example:

.. code-block:: python

    allspark.HuggingFaceModel(model_model_path, safe_model_name, in_memory_serialize=in_memory,
                              user_set_data_type="bfloat16")

How to configure RuntimeConfig using a dictionary?
--------------------------------------------------

``AsModelRuntimeConfigBuilder`` can be imported using the ``from_dict()`` function. The format can be referenced from the DIConfig YAML format, and you can also use the corresponding dictionary format for configuration.

Here's an example:

.. code-block:: python

    input_dict = {
        'model_name': 'test_model',
        'compute_unit': {
            'device_type': 'cuda',
            'device_ids': [0, 1],
            'compute_thread_in_device': 2
        },
        'engine_max_length': 100,
        'engine_max_batch': 32
    }
    # Create a Builder instance and call from_dict
    builder = AsModelRuntimeConfigBuilder()
    builder.from_dict(input_dict)

For the complete configuration, please refer to the configuration file in DIConfig.

How to asynchronously retrieve output (without calling `sync_request`)?
-----------------------------------------------------------------------

See the code in `Handling Output` section of :doc:`get_started/quick_start_api_py_en`. Asynchronous output retrieval is the recommended way to achieve high throughput.
