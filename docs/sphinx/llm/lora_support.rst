===============
LoRA Support
===============

Before read this document, please first read  `LoRA Adapters <https://arxiv.org/abs/2106.09685>`_ for a basic concept and process.

Overview
--------------

The AllSpark Engine can work with multiple LoRAs based on the models listed in the `python/pyhie/allspark/model/` directory. When you want to perform inference with LoRA for the first time, four steps should be completed.

Prepare LoRA Adapters
------------------------

Before conversion, ensure that the LoRA adapter files are in PyTorch format. If they are in SafeTensors format, you can easily convert them using the `safetensors` module:

.. code-block:: python

    from safetensors.torch import safe_open
    safetensor_model_path = 'lora-1/adapter_model.safetensors'
    with safe_open(safetensor_model_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(raw_lora_dir, 'lora-1/adapter_model.bin'))


After conversion, the directory structure should look like this:

.. code-block:: bash

    /dir/to/my/loras/
         |__ lora-1/
               |__ adapter_config.json
               |__ adapter_model.safetensors   (raw)
               |__ adapter_model.bin           (converted to pth format)
         |__ lora-2/
               |__ adapter_config.json
               |__ adapter_model.safetensors   (raw)
               |__ adapter_model.bin           (converted to pth format)


Enable LoRA
--------------

To enable LoRA support, a base model should be converted into AllSpark format using a JSON lora_cfg argument with the following fields:

1. `input_base_dir`: A relative or absolute directory path to the parent directory of the LoRA adapters.
2. `lora_names`: A list of LoRA names, each being a directory name.
3. `lora_cfg`: An optional boolean flag indicating whether to convert only the adapters or also the base model.

.. code-block:: python

    output_dir = '/path/to/output/dir/' # output directory name
    engine = allspark.Engine()
    model_loader = allspark.HuggingFaceModel(...)
    model_loader.load_model().read_model_config().serialize_to_path(
        engine,
        output_dir,
        lora_cfg={
            'input_base_dir': '/dir/to/my/loras/',      # input parent directory of all the LoRA adatper directories
            'lora_names': ['lora-1', 'lora-2'],  # which LoRA adapters will be converted.  The lora name is also the directory name of the LoRA adapter.
            'lora_only': False          # False means you will convert both the base model and the LoRA adapters. No base model but only LoRA converted if set True.
        }
    )

After calling serialize_to_path(), four files are generated in the output_dir:
qwen7b.asgraph:  Base model containing LoRA support
qwen7b.asparam:  Weights data of base model 
lora-{1,2}.aslroa:  Converted LoRA adapters


Setup LoRA Limits
-------------------

The limits of LoRA should be set appropriately before inference. You can change the default limits using the following instructions:

.. code-block:: python

    runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(...)
    runtime_cfg_builder.lora_max_num(20).lora_max_rank(64)

Now, both the maximum number and maximum rank of all LoRA adapters are set.


Infer With LoRA
-----------------

Finally, you can pass the `lora_name` argument into `GenerationConfig` and use it to perform generation tasks.

.. code-block:: python

    gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
    gen_cfg.update({"lora_name": 'lora-2'})
    status, handle, queue = engine.start_request_text(converted_model_name,
                                                      model_loader,
                                                      input_str,
                                                      gen_cfg)

Example
--------------

The full example of how to use LoRA is as follows: 

.. code-block:: python


    import os
    import torch
    import modelscope
    from modelscope.utils.constant import DEFAULT_MODEL_REVISION

    from dashinfer import allspark
    from dashinfer.allspark.engine import TargetDevice
    from dashinfer.allspark.prompt_utils import PromptTemplate
    from dashinfer.allspark._allspark import AsStatus, GenerateRequestStatus, AsCacheMode
    from safetensors import safe_open

    def check_transformers_version():
        import transformers
        required_version = "4.37.0"
        current_version = transformers.__version__

        if current_version < required_version:
            raise Exception(
                f"Transformers version {current_version} is lower than required version {required_version}. Please upgrade transformers to version {required_version}."
            )
            exit()


    def convert_safetensor_to_pytorch(raw_lora_dir):
        model_path = os.path.join(raw_lora_dir, 'adapter_model.safetensors')
        tensors = {}
        with safe_open(model_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        torch.save(tensors, os.path.join(raw_lora_dir, 'adapter_model.bin'))

    if __name__ == '__main__':
        check_transformers_version()
        # if use in memory serialize, change this flag to True
        in_memory = False
        init_quant= False
        weight_only_quant = True
        device_list=[0,1]
        fetch_output_mode = "async" # or "sync"
        modelscope_name ="qwen/Qwen2-7B-Instruct"
        ms_version = DEFAULT_MODEL_REVISION
        model_local_path=""
        output_model_dir = "../../model_output"


        model_local_path = modelscope.snapshot_download(modelscope_name, ms_version)
        safe_model_name = str(modelscope_name).replace("/", "_")

        model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name, user_set_data_type="bfloat16", in_memory_serialize=in_memory, trust_remote_code=True)
        engine = allspark.Engine()

        # lora-1 and lora-2 are adapter directories, which include adapter_config.json and adapter_model.bin (pth format)
        lora_base_dir = '/dir/to/my/loras/'
        # If the format is .safetensors, you should run the following conversion:
        # start lora format conversion:
        convert_safetensor_to_pytorch(os.path.join(lora_base_dir, 'lora-1'))
        convert_safetensor_to_pytorch(os.path.join(lora_base_dir, 'lora-2'))
        if in_memory:
            (model_loader.load_model()
            .read_model_config()
            .serialize_to_memory(engine, enable_quant=init_quant, weight_only_quant=weight_only_quant,
                                lora_cfg={'input_base_dir': lora_base_dir,
                                          'lora_names': ['lora-1', 'lora-2'],
                                          'lora_only': False}
                                )
            .export_model_diconfig(os.path.join(output_model_dir, "diconfig.yaml"))
            .free_model())
        else:
            (model_loader.load_model()
            .read_model_config()
            .serialize_to_path(engine, output_model_dir, enable_quant=init_quant, weight_only_quant=weight_only_quant,
                                lora_cfg={'input_base_dir': lora_base_dir,
                                          'lora_names': ['lora-1', 'lora-2'],
                                          'lora_only': False},
                                skip_if_exists=True
                              )
            .free_model())

        runtime_cfg_builder = model_loader.create_reference_runtime_config_builder(safe_model_name, TargetDevice.CUDA,
                                                                                device_list, max_batch=8)
        # like change to engine max length to a smaller value
        runtime_cfg_builder.max_length(256).lora_max_num(25).lora_max_rank(64)

        # like enable int8 kv-cache or int4 kv cache rather than fp16 kv-cache
        # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8)

        # or u4
        # runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)
        runtime_cfg = runtime_cfg_builder.build()

        # install model to engine
        engine.install_model(runtime_cfg)

        if in_memory:
            model_loader.free_memory_serialize_file()

        # start the engine
        engine.start_model(safe_model_name)
        # load loras
        ret = engine.load_lora(safe_model_name, 'lora-1')
        assert(ret == AsStatus.ALLSPARK_SUCCESS)
        ret = engine.load_lora(safe_model_name, 'lora-2')
        assert(ret == AsStatus.ALLSPARK_SUCCESS)

        # start model inference with lora
        input_list = ["你是谁？", "How to protect our planet and build a green future?"]
        for i in range(len(input_list)):
            input_str = input_list[i]
            input_str = PromptTemplate.apply_chatml_template(input_str)
            # generate a reference generate config.
            gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
            # change generate config base on this generation config, like change top_k = 1
            gen_cfg.update({"top_k": 1})
            gen_cfg.update({"repetition_penalty": 1.1})
            gen_cfg.update({"lora_name": 'lora-%d'%(i % 2 + 1)})
            #gen_cfg.update({"eos_token_id", 151645})
            status, handle, queue = engine.start_request_text(safe_model_name,
                                                            model_loader,
                                                            input_str,
                                                            gen_cfg)

            generated_ids = []
            if fetch_output_mode == "sync":
                # sync will wait request finish, like a sync interface, but you can async polling the queue.
                # without this call, the model result will async running, result can be fetched by queue
                # until queue status become generate finished.
                engine.sync_request(safe_model_name, handle)

                # after sync, you can fetch all the generated id by this api, this api is a block api
                # will return when there new token, or generate is finished.
                generated_elem = queue.Get()
                # after get, engine will free resource(s) and token(s), so you can only get new token by this api.
                generated_ids += generated_elem.ids_from_generate
            else:
                status = queue.GenerateStatus()

                ## in following 3 status, it means tokens are generating
                while (status == GenerateRequestStatus.Init
                    or status == GenerateRequestStatus.Generating
                    or status == GenerateRequestStatus.ContextFinished):
                    print(f"2 request: status: {queue.GenerateStatus()}")
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




            # de-tokenize id to text
            output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
            print("---" * 20)
            print(
                f"test case: {modelscope_name} input:\n{input_str}  \n output:\n{output_text}\n")
            print(f"input token:\n {model_loader.init_tokenizer().get_tokenizer().encode(input_str)}")
            print(f"output token:\n {generated_ids}")

            engine.release_request(safe_model_name, handle)

        engine.stop_model(safe_model_name)
        engine.release_model(safe_model_name)

