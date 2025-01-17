# Quick Start Guide for Python API
## Main Modules
1. **Model Loading**: 
For Hugging Face models, use this loader: `dashinfer.allspark.model_loader.HuggingFaceModel`.
It will parse the parameters of the HF model and create corresponding parameters for conversion.

2. **Model Serialization**: The model serialization process converts the model into DashInfer format, offering two modes: one is a transparent in-memory conversion that does not generate intermediate files, but the drawback is that it will double the memory usage. The other method converts to a local file, which can later be loaded using the DashInfer loader (WIP).
   - **Quantization Weights**: During serialization, you can enable weight quantization, which is categorized into weight-only quantization and compute quantization.

3. **Engine Installation of Models and Starting Models**: Once the model is loaded, it will be installed in the engine, and the engine will start the installed model while assigning VRAM pools as specified.

4. **Request Initiation and Output Reception**: This part primarily focuses on asynchronously initiating requests. After the request is created, the engine will process the request in a continuous batching manner. The corresponding output from the request is obtained through the output queue of the request, which also allows for asynchronous monitoring of the current status of the request.

## Quick Start
Below is an example of how to quickly serialize a Hugging Face model and perform model inference. This example will download the "qwen/Qwen2.5-1.5B-Instruct" model from Modelscope and perform conversion and inference.

### Inference Python Example
This is an example of using asynchronous interface to obtain output, with bfloat16, in memory model serialize, and async output processing. The model is downloaded from Modelscope. Initiating requests and receiving outputs are both asynchronous, and can be handled according to your application needs.

```python
import os

from dashinfer import allspark
from dashinfer.allspark import *
from dashinfer.allspark.engine import *
from dashinfer.allspark.prompt_utils import PromptTemplate

# Configuration
in_memory = False
device_list = [0] # single card by default, 4 cards replace with [0,1,2,3] 
model_name = "qwen/Qwen2.5-1.5B-Instruct"
output_base_folder = "model_output"
user_data_type = "float16" # most device supports float16
use_modelscope = True

# Download and prepare the model
if use_modelscope:
   import modelscope
   from modelscope.utils.constant import DEFAULT_MODEL_REVISION
   model_local_path = modelscope.snapshot_download(model_name, DEFAULT_MODEL_REVISION)
else:
   model_local_path = model_name

safe_model_name = model_name.replace("/", "_")
model_convert_folder = os.path.join(output_base_folder, safe_model_name)

# Initialize model and engine
model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name,
                                         in_memory_serialize=in_memory,
                                         user_set_data_type=user_data_type)
engine = allspark.Engine()

# Load and serialize the model
model_loader.load_model().serialize(engine, model_output_dir=output_base_folder).free_model()

# Configure runtime settings
runtime_cfg = model_loader.create_reference_runtime_config_builder(
   safe_model_name, TargetDevice.CUDA, device_list, max_batch=8).max_length(2048).build()
engine.install_model(runtime_cfg)
engine.start_model(safe_model_name)

if in_memory: model_loader.free_memory_serialize_file()

# Prepare input
input_str = "How to protect our planet and build a green future?"
messages = [
   {"role": "system", "content": "You are a helpful assistant."},
   {"role": "user", "content": PromptTemplate.apply_chatml_template(input_str)}
]
templated_input_str = model_loader.init_tokenizer().get_tokenizer().apply_chat_template(
   messages, tokenize=False, add_generation_prompt=True)

# Configure generation settings
gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
gen_cfg.update({"top_k": 1})

# Generate response
status, handle, queue = engine.start_request_text(
   safe_model_name, model_loader, templated_input_str, gen_cfg)
generated_ids = []

while True:
   elements = queue.Get()
   if elements:
      generated_ids += elements.ids_from_generate
   status = queue.GenerateStatus()
   if status in [GenerateRequestStatus.GenerateFinished,
                 GenerateRequestStatus.GenerateInterrupted]:
      break

# Decode and print output
output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
print(f"Model: {model_name}\nInput: {input_str}\nOutput: {output_text}")

# Clean up
engine.release_request(safe_model_name, handle)
engine.stop_model(safe_model_name)
print(f"Model: {model_name} have been released.")
```

### Explanations of the Code:
#### 1. Model Loading
In this example, the `HuggingFaceModel` (`dashinfer.allspark.model_loader.HuggingFaceModel`) class is first created, which will download the model. If your model is local, modify the `model_local_path` parameter. If this path is empty or the file does not exist, an error will be raised. If your model exists in Modelscope, simply pass the Modelscope model ID. Then, create an Engine class and use the relevant APIs of the model loader to load the model, serialize it, and then release it.

If you want to convert only once, pass `skip_if_exists=True`. If existing files are found, the model conversion step will be skipped. The model files will reside in the `{output_base_folder}` directory, generating two files: `{safe_model_name}.asparam`, `{safe_model_name}.asmodel`. The `free_model()` function will release the Hugging Face model files to save memory.
```python
model_loader = allspark.HuggingFaceModel(model_local_path, safe_model_name,
                                         in_memory_serialize=in_memory,
                                         user_set_data_type=user_data_type)
engine = allspark.Engine()
```

#### 2. Model Serialization
This step primarily serializes the model. The serialized artifacts can be either on the filesystem (`serialize_to_path`) or serialized in memory. For example, the `serialize_to_memory` below stores temporary files internally in the model loader, which can later be released using `model_loader.free_memory_serialize_file()`.

use `model_loader.serialize()` for uniform API like in sample code, or use `serialize_to_path` or `serialize_to_memory` for you needs.
`skip_if_exists` means if there is local file exits, local file serialize will be bypassed.

```python
model_loader.load_model().serialize(engine, model_output_dir=output_base_folder).free_model()
# or 
if in_memory:
    (model_loader.load_model()
     .serialize_to_memory(engine, enable_quant=init_quant, weight_only_quant=weight_only_quant)
     .free_model())
else:
    (model_loader.load_model()
     .serialize_to_path(engine, tmp_dir, enable_quant=init_quant, weight_only_quant=weight_only_quant,
                        skip_if_exists=False)
     .free_model())    
```

#### 3. Configuring Engine Runtime Parameters and Starting the Engine:
In this code section, inference is conducted using a single CUDA card, with the maximum batch size set to 8, which can be modified based on your situation. A reference runtime configuration is setup using parameters from the model loader obtained from Hugging Face. This includes settings like the maximum length supported by the model, which can also be modified after the generation of the reference configuration if necessary. The `install_model` function registers the model with the engine, and the `safe_model_name` must be a unique ID.

If using in-memory serialization, you can release the memory file after `install_model`, since it is no longer needed.

```python
if in_memory: model_loader.free_memory_serialize_file()

```

Upon calling `start_model`, the engine will perform a warm-up step that simulates a run with the maximum length set in the runtime parameters and the maximum batch size to ensure that no new resources will be requested during subsequent runs, ensuring stability. If the warm-up fails, reduce the length settings in the runtime configurations to lower resource demand. After completion of the warm-up, the engine enters a state ready to accept requests.

```python
runtime_cfg = model_loader.create_reference_runtime_config_builder(
   safe_model_name, TargetDevice.CUDA, device_list, max_batch=8).max_length(2048).build()
# like enable int8 kv-cache or int4 kv cache rather than fp16 kv-cache
# runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantI8)
# or u4
# runtime_cfg_builder.kv_cache_mode(AsCacheMode.AsCacheQuantU4)
# install model to engine
engine.install_model(runtime_cfg)
# start the model inference
engine.start_model(safe_model_name)
```

#### 4. Sending Requests
The following code is focused on generating configurations and applying text templating for sending requests using `engine.start_request_text`, and retrieving model outputs using `handle` and `queue`, printing the model's output afterward.

```python
gen_cfg = model_loader.create_reference_generation_config_builder(runtime_cfg)
gen_cfg.update({"top_k": 1})
```
This code takes recommended generation parameters from Hugging Face's `generation_config.json` and makes optional modifications. It then asynchronously initiates model inference, where `status` indicates the success of the API. If successful, `handle` and `queue` are used for subsequent requests. The `handle` represents the request handle, while `queue` indicates the output queue; each request has its own output queue, which continuously accumulates generated tokens. This queue will only be released after `release_request` is invoked.

```python
status, handle, queue = engine.start_request_text(
   safe_model_name, model_loader, templated_input_str, gen_cfg)
```

#### 5. Handling Output

DashInfer prioritizes asynchronous APIs for optimal performance and to align with the inherent nature of LLMs. Sending and receiving requests is primarily designed for asynchronous operation. However, for compatibility with user preferences accustomed to synchronous calls, we provide `engine.sync_request()`. This API allows users to block until the generation request completes.

##### 5.1 Asynchronous Processing
Asynchronous processing differs in that it requires repeated calls to the queue until the status changes to `GenerateRequestStatus.ContextFinished`. A normal state machine transition goes:
`Init` (initial state) -> `ContextFinished` (prefill completed and first token generated) ->
`Generating` (in progress) -> `GenerateFinished` (completed).
During this normal state transition, an exceptional state can occur: `GenerateInterrupted`, which indicates resource shortages, causing the request to pause while its resources are temporarily released for others. This often happens under heavy loads.

```python
generated_ids = []
while True:
   elements = queue.Get()
   if elements:
      generated_ids += elements.ids_from_generate
   status = queue.GenerateStatus()
   if status in [GenerateRequestStatus.GenerateFinished, GenerateRequestStatus.GenerateInterrupted]:
      break
```

##### 5.2 Synchronous Processing

The subsequent call to `sync_request` will block until generation is finished, simulating a synchronous call. Without this invocation, operations on the queue can proceed but will require polling. The following code synchronously fetches all currently generated IDs from the queue, blocking at this point if there are IDs yet to be generated until completion or an error occurs.

Sync processing is not showing in this example code, you can modify example following code.

Here's an example:

```python
# The sync_request call waits for the request to finish, simulating synchronous behavior. 
# Alternatively, you could asynchronously poll the queue. Without this call, results 
# are processed asynchronously, and you would fetch them from the queue until its status 
# indicates completion.
engine.sync_request(safe_model_name, handle)

# After sync_request completes, fetch all generated IDs. This call blocks until new 
# tokens are available or generation finishes.
generated_elem = queue.Get()

# After retrieving results, the engine releases resources and tokens.  Subsequent 
# tokens must be retrieved via this same API.
generated_ids = generated_elem.ids_from_generate
```

#### 6. Decode Token
For usage of the queue class, you can use `help(dashinfer.allspark.ResultQueue)` for detailed information. The next step converts IDs back into text:

```python
output_text = model_loader.init_tokenizer().get_tokenizer().decode(generated_ids)
```

