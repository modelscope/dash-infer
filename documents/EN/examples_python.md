# 0_basic

The Python code in the `<path_to_dashinfer>/examples/python/0_basic` directory provides examples of how to call the DashInfer Python interface.

## Functions

Functions include:

- Model Download: By default, models are downloaded from the ModelScope community,  but you can also refer to comments in the code to download models from HuggingFace.
- Model Conversion: Convert models from the HuggingFace format to the DashInfer format.
- Model Inference: Use the DashInfer for multi-batch parallel inference, and obtain inference results in a streaming manner.

## Run

Install the DashInfer python package before running the example:

```shell
# install from pip
pip install dashinfer

# install from local package
pip install dashinfer-<dashinfer-version>-xxx.whl
```

Run the python example under `<path_to_dashinfer>/examples/python/0_basic`:

```shell
python basic_example_qwen_v10.py
```

The models in DashInfer format obtained from the conversion are stored in the `~/dashinfer_models/`. You can also specify the target path by modifying the `model_path` field of the model configuration file.

## Single/Multi-NUMA Inference

DashInfer support Single/Multi-NUMA inference.

### Single-NUMA Inference on Single-NUMA CPUs

The examples use single-NUMA inference by default.
On CPUs with single NUMA node, no special configuration is required.

### Single-NUMA Inference on Multi-NUMA CPUs

On CPUs with multiple NUMA nodes, if only 1 NUMA node is needed for inference, `numactl` is necessary for core binding.
This approach requires `--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host` arguments when creating containers by docker run.

To use single-NUMA inference, you need to change the `device_ids` field in the model configuration file to the NUMA node number you want to use, and set the `multinode_mode` to be true.

```json
"device_ids": [
    0
],
"multinode_mode": true,
```

### Multi-NUMA Inference

For multi-NUMA inference, use `mpirun` + `numactl` to bind cpu cores for optimal performance.
This approach requires `--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host` arguments when creating containers by docker run.

To use multi-NUMA inference, you need to change the `device_ids` field in the model configuration file to the NUMA node numbers you want to use, and set the `multinode_mode` to be true.

```json
"device_ids": [
    0,
    1
],
"multinode_mode": true,
```

## Replace Models

Replacing other models of the same structure requires following changes in basic_example_xxx.py:

1. config_file

```python
config_file = "model_config/config_qwen_v10_1_8b.json"
```

2. HuggingFace (or ModelScope) model information

```python
# download model from huggingface
original_model = {
    "source": "huggingface",
    "model_id": "Qwen/Qwen-1_8B-Chat",
    "revision": "",
    "model_path": ""
}
```

```python
# download model from modelscope
original_model = {
    "source": "modelscope",
    "model_id": "qwen/Qwen-1_8B-Chat",
    "revision": "v1.0.0",
    "model_path": ""
}
```

3. format of prompt

```python
start_text = "<|im_start|>"
end_text = "<|im_end|>"
system_msg = {"role": "system", "content": "You are a helpful assistant."}
user_msg = {"role": "user", "content": ""}
assistant_msg = {"role": "assistant", "content": ""}

prompt_template = Template(
    "{{start_text}}" + "{{system_role}}\n" + "{{system_content}}" + "{{end_text}}\n" +
    "{{start_text}}" + "{{user_role}}\n" + "{{user_content}}" + "{{end_text}}\n" +
    "{{start_text}}" + "{{assistant_role}}\n\n")
```

## Add a New Model

Adding a new model requires the following changes.

- Add C++ source code and header files of the new model to the `<path_to_dashinfer>/csrc/core/model` directory.
- Write the corresponding model adapter code in the `<path_to_dashinfer>/python/dashinfer/allspark/model` directory.
- Add the new model type in the model_map in the `<path_to_dashinfer>/python/dashinfer/allspark/engine.py` file.
- Import the new model adapter code in the `<path_to_dashinfer>/python/dashinfer/allspark/model/__init__.py` file and add the new model type.

# 1_performance

The Python code in the `<path_to_dashinfer>/examples/python/1_performance` directory provides examples of inference performance testing using random numbers.

The difference with the basic example is that the performance test example uses random inputs to test the context performance and generation performance under various batch_size, input_len, and output_len configurations.

```python
batch_size_list = [1, 2, 4, 8]
output_len_list = [128]
input_len_list = [128, 1200]
```

During these performance evaluations, the `early_stopping` parameter is set to false, indicating that generation will no stop, even if an EOS token is produced.

Enter the directory ``<path_to_dashinfer>/examples/python/1_performance`, and execute following command to run the example:

```shell
python performance_test_qwen_v15.py
python performance_test_qwen_v15.py --device_ids 0 1 # test multi-NUMA performance
```

> On CPUs with multiple NUMA nodes, please refer to [Single/Multi-NUMA Inference] (examples_python.md#L33) section for best performance.

# 2_evaluation

The code in the `<path_to_dashinfer>/examples/python/2_evaluation` directory is from [QwenLM/Qwen](https://github.com/QwenLM/Qwen/tree/main/eval). The original code uses transformers for inference. In this repository, the accuracy testing code substitutes the inference engine with DashInfer.

For accuracy evaluation, please refer to [EVALUATION.md](../../examples/python/2_evaluation/EVALUATION.md).

# 3_gradio

The Gradio demo in the `<path_to_dashinfer>/examples/python/3_gradio` directory demonstrates how to deploy a chat service using DashInfer as the backend inference engine.

## Step 1: Model Conversion

Run `basic_example_qwen_v10.py` first to get the converted model.

## Step 2: Network Configuration（Optional）

This step can be skipped for local deployments.

Some network configuration is required to access the created gradio demo from the external network.

There are two ways to configure it:

1. Using the official Gradio proxy
    - No root permissions required.
    - Anyone who gets the link can access the demo.
    - The free link comes with a 72-hour time limit.

2. Using a custom Nginx proxy
    - The link has no time restrictions.
    - Installing Nginx requires root permissions.

### Method 1: Official Gradio Proxy

Set `share=True` in `launch()` method in the gradio_demo_qwen.py, like this:

``` python
demo.queue(api_open=False).launch(height=800, share=True, server_name="127.0.0.1", server_port=7860)
```

An error may be reported, just follow the instructions.
Don't forget to add executable permission: `chmod +x frpc_linux_amd64_v0.2`

```
Could not create share link. Missing file: /miniconda/envs/py38/lib/python3.8/site-packages/gradio/frpc_linux_amd64_v0.2. 

Please check your internet connection. This can happen if your antivirus software blocks the download of this file. You can install manually by following these steps: 

1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
2. Rename the downloaded file to: frpc_linux_amd64_v0.2
3. Move the file to this location: /miniconda/envs/py38/lib/python3.8/site-packages/gradio
```

Following message will appear at the end of a successful run:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxx.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```

Then you can access your demo via the public URL.

### Method 2: Custom Nginx proxy

#### Download and Install Nginx

Check to if nginx is already installed:

```shell
nginx -v
```

If nginx is not installed, execute following command to install:

Ubuntu:

```shell
apt-get install -y nginx
```

CentOS:

```shell
yum install -y nginx
```

#### Modify the Configuration File

Ubuntu users please refer to `nginx_ubuntu.conf` under `<path_to_dashinfer>/examples/nginx_config` directory and modify `/etc/nginx/sites-available/default`.

CentOS users please refer to `nginx_centos.conf` under `<path_to_dashinfer>/examples/nginx_config` directory and modify `/etc/nginx/nginx.conf`.

```
location /gradio-demo/ {
    proxy_pass http://127.0.0.1:7860/;
    proxy_buffering off;
    proxy_redirect off;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

- - The location should be consistent with the `root_path` parameter in gradio demo `launch()` method. In the sample code, `root_path="/gradio-demo/"`.
- The port in proxy_pass should be consistent with the gradio's server port, the default port is `7860`.
- If you are deploying multiple gradio demos on the same server, you can configure multiple locations to map to different ports.

Check the configuration file for syntax errors: `nginx -t`

#### Start Nginx Service

Start nginx service: `nginx`

Restart nginx service: `nginx -s reload`

#### Modify launch() Method

Set `root_path="/gradio-demo/"` in `launch()` method in the gradio_demo_qwen.py, like this:

``` python
demo.queue(api_open=False).launch(root_path="/gradio-demo/",
                                  height=800,
                                  share=False,
                                  server_name="127.0.0.1",
                                  server_port=7860)
```

After successful deployment, you can interact with the application by visiting the URL: `http://server_ip_address/gradio-demo/`.

- For servers within the same local area network (LAN), access can be achieved using the LAN IP address.
- For servers not on the same LAN, access requires the use of the server's public IP address.

## Step 3: Run Demo

Run the python example under `<path_to_dashinfer>/examples/python/3_gradio`:

```shell
python gradio_demo_qwen.py
```

Once the following output appears in the terminal, you can access the deployed Gradio application through a web browser:

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

# Model Configuration Files

The `<path_to_dashinfer>/examples/python/model_config` directory provides several configuration examples.

Here is an explanation of the parameters within the config:

- `model_name`: Custom name for the DashInfer model.
- `model_type`: The type of the DashInfer model. Options include: LLaMA_v2, ChatGLM_v2, ChatGLM_v3, Qwen_v10, Qwen_v15, Qwen_v20.
- `model_path`: The export path for the DashInfer model.
- `data_type`: The data type of the output. Options include: float32.
- `device_type`: The inference hardware. Options include: CPU.
- `device_ids`: The NUMA node used for inference. NUMA information of your CPU can be viewed with the Linux command `lscpu`.
- `multinode_mode`: Whether or not the engine is running on a multi-NUMA CPU. Options include: true, false.
- `convert_config`: Parameters related to model conversion.
    - `do_dynamic_quantize_convert`: Whether to quantize the weights. Options include: true, false. Currently, only ARM CPUs support quantization.
- `engine_config`: Inference engine parameters.
    - `engine_max_length`: The maximum inference length, <= 11000.
    - `engine_max_batch`: The maximum batch size.
    - `do_profiling`: Whether to profile the inference process. Options include: true, false. To conduct profiling, `do_profiling` should be set to true and the environment variable `AS_PROFILE=ON` configured.
    - `num_threads`: The number of threads. For optimal performance, this should match the number of physical cores in a single NUMA node. If set to 0, EngineHelper will automatically parse `lscpu` output and set the value. If > 0, the set value is used.
    - `matmul_precision`: The computational precision for matrix multiplication. Options include: high, medium. When set to high, fp32 is used for matrix multiplication; when set to medium, bf16 is used.
- `generation_config`: Generation parameters.
    - `temperature`: The temperature for randomness.
    - `early_stopping`: Whether to stop generating after generating `stop_words_ids`. Options include: true, false.
    - `top_k`: The top k parameter for sampling. When top_k = 0, it ranks the entire vocabulary.
    - `top_p`: The top p parameter for sampling, 0 <= top_p <= 1.0. top_p = 0 means not using topp.
    - `repetition_penalty`: The parameter for repetition penalty. 1.0 means no penalty.
    - `presence_penalty`: The parameter for presence penalty. 0.0 means no penalty.
    - `min_length`: The minimum length for input+output. Default is 0, not enabling the filter.
    - `max_length`: The maximum length for input+output.
    - `no_repeat_ngram_size`: Controls the generation of repeat words. Default is 0. If set to int > 0, all ngrams of that size can only occur once.
    - `eos_token_id`: The token id corresponding to EOS, dependent on the model.
    - `seed`: The seed for randomness.
    - `stop_words_ids`: A list of token ids for stop words.
- `quantization_config`: Quantization parameters, required when `do_dynamic_quantize_convert` is set to true.
    - `activation_type`: The data type for the input matrix in matrix multiplication. Options include: bfloat16.
    - `weight_type`: The data type for weights in matrix multiplication. Options include: uint8.
    - `SubChannel`: Whether to perform sub-channel quantization on weights. Options include: true, false.
    - `GroupSize`: The granularity of sub-channel quantization. Options include: 64, 128, 256, 512.
