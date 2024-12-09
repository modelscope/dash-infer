===============
Guided Decoding
===============

Before read this document, please first read  :doc:`LLM Offline Inference By Python API <../llm/llm_offline_inference_en>` for basic concept and process.

*******************************
JSON Format Structured Decoding
*******************************

The AllSpark Engine uses lm-format-enforcer as the backend for guided decoding. Currently only JSON format is supported.
`lm-format-enforcer repository <https://github.com/noamgat/lm-format-enforcer>`

Example
-------

Provide the 'response_format' dict in GenerationConfig of a request, like

.. code-block:: python

    # Fill in basic arguments in gen_cfg
    gen_cfg_builder = ASGenerationConfigBuilder()
    gen_cfg_updates = {
       "temperature": 0.7,
       "top_k": 20,
       "top_p": 0.9,
       "seed": 1234,
       "max_length": 1024,
       "repetition_penalty": 1.05,
       "length_penalty": 1.0,
    }
    
    # An example of a simple schema
    schema_str = r'''
    {
        "properties": {
            "company name": {
                "type": "string"
            },
            "founding year": {
                "type": "integer"
            },
            "founding person": {
                "type": "string"
            },
            "founding city": {
                "type": "string"
            },
            "employees": {
                "type": "integer"
            }
        },
        "required": [
            "company name",
            "founding year",
            "founding person",
            "founding city",
            "employees"
        ],
        "type": "object"
    }'''
    
    # Build GenerationConfig with the 'response_format' dict
    gen_cfg_updates["response_format"] = {"type": "json_object", "json_schema": schema_str}
    # or not providing any schema to generate any JSON format output, like:
    # gen_cfg_updates["response_format"] = {"type": "json_object"}
    gen_cfg_builder.update(gen_cfg_updates)
    config = gen_cfg_builder.build()
    
