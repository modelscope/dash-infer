# 0_basic

`<path_to_dashinfer>/examples/python/0_basic`目录下的python代码提供了DashInfer python接口的调用示例。

## 功能

功能包括：

- 模型下载：默认从魔搭社区（ModelScope）下载模型，亦可参考代码中的注释，从HuggingFace下载模型；
- 模型转换：将HuggingFace格式的模型转换为DashInfer格式的模型；
- 模型推理：使用DashInfer推理引擎进行多batch并行推理，流式获取推理结果。

## 运行

运行示例前，需要先安装DashInfer python whl包。

```shell
# install from pip
pip install dashinfer

# install from local package
pip install dashinfer-<dashinfer-version>-xxx.whl
```

在`<path_to_dashinfer>/examples/python/0_basic`目录下运行示例python脚本，例如：

```shell
python basic_example_qwen_v10.py
```

转换得到的DashInfer格式的模型存储在`~/dashinfer_models/`目录下。也可以通过修改模型配置文件的`model_path`字段，修改模型的保存路径。

## 单NUMA/多NUMA 推理

DashInfer支持单/多NUMA并行推理。

### 在单NUMA CPU上进行单NUMA推理

示例中默认使用的是单NUMA推理。
在单NUMA CPU上进行单NUMA推理不需要特殊的权限和配置。

### 在多NUMA CPU上进行单NUMA推理

在多NUMA节点的CPU上，若只需要1个NUMA节点进行推理，需要用`numactl`进行绑核。
该方式需要在docker run的时候添加`--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host`参数。

使用单NUMA推理，需要将模型配置文件中的`device_ids`字段改成要运行的NUMA节点编号，并将`multinode_mode`字段设置为true。

```json
"device_ids": [
    0
],
"multinode_mode": true,
```

### 多NUMA推理

使用多NUMA推理，需要用`mpirun`+`numactl`进行绑核，以达到最佳性能。
该方式需要在docker run的时候添加`--cap-add SYS_NICE --cap-add SYS_PTRACE --ipc=host`参数。

使用多NUMA推理，需要将模型配置文件中的`device_ids`字段改成要运行的NUMA节点编号，并将`multinode_mode`字段设置为true。

```json
"device_ids": [
    0,
    1
],
"multinode_mode": true,
```

## 更换模型

更换同结构的其它模型，需要在basic_example_xxx.py中修改以下几处：

1. config_file

```python
config_file = "model_config/config_qwen_v10_1_8b.json"
```

2. HuggingFace（或ModelScope）的下载参数

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

3. prompt的组织格式

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

## 增加新模型

增加一个新模型，需要修改以下几处。

1. 在`<path_to_dashinfer>/csrc/core/model`路径下，增加新的模型C++代码和头文件；
2. 在`<path_to_dashinfer>/python/dashinfer/allspark/model`目录下，编写对应的模型适配代码；
3. 在`<path_to_dashinfer>/python/dashinfer/allspark/engine.py`文件的model_map中增加新增的模型类型；
4. 在`<path_to_dashinfer>/python/dashinfer/allspark/model/__init__.py`文件中引入新的模型适配代码，添加新增的模型类型。

# 1_performance

`<path_to_dashinfer>/examples/python/1_performance`目录下的python代码提供了用随机数进行推理性能测试的示例。

与basic example的区别是，performance test example采用的是随机输入，测试各种batch_size、input_len、output_len配置下的context性能和generation性能。

```python
batch_size_list = [1, 2, 4, 8]
output_len_list = [128]
input_len_list = [128, 1200]
```

在进行性能测试时，`early_stopping`参数会被设置为`false`，这表示即使生成了结束符，也不会停止生成。

在`<path_to_dashinfer>/examples/python/1_performance`目录下运行示例python脚本，例如：

```shell
python performance_test_qwen_v15.py
python performance_test_qwen_v15.py --device_ids 0 1 # test multi-NUMA performance
```

> 多NUMA CPU推理请参考[单NUMA/多NUMA 推理](examples_python.md#L33)章节中的内容，以达到最佳性能。

# 2_evaluation

`<path_to_dashinfer>/examples/python/2_evaluation`目录下的代码来自[QwenLM/Qwen](https://github.com/QwenLM/Qwen/tree/main/eval)。原始代码中采用transformers推理，本仓库中的精度测试代码将推理部分用DashInfer替代。

精度测试请参考2_evaluation目录下的[EVALUATION.md](../../examples/python/2_evaluation/EVALUATION.md)进行。

# 3_gradio

`<path_to_dashinfer>/examples/python/3_gradio`目录下的Gradio demo演示了如何用DashInfer作为后端推理引擎，部署一个chat服务。

## Step 1: 模型转换

运行Gradio demo前，需要先运行`basic_example_qwen_v10.py`，得到转换后的模型。

## Step 2: 网络配置（Optional）

仅在本地部署可跳过此步骤。

要从外部访问创建的gradio demo需要进行一些网络配置。

配置的方法有两种：

1. 使用gradio官方代理
    - 不需要root权限
    - 得到链接的任何人都可访问
    - 免费链接有72h时限

2. 使用nginx自定义代理
    - 链接无时限
    - 安装nginx，需要root权限

### 方法1：gradio官方代理

gradio_demo_qwen.py，launch()函数设置`share=True`，例如：

``` python
demo.queue(api_open=False).launch(height=800, share=True, server_name="127.0.0.1", server_port=7860)
```

可能会出现报错，按照提示操作即可。
最后还需要`chmod +x frpc_linux_amd64_v0.2`

```
Could not create share link. Missing file: /miniconda/envs/py38/lib/python3.8/site-packages/gradio/frpc_linux_amd64_v0.2. 

Please check your internet connection. This can happen if your antivirus software blocks the download of this file. You can install manually by following these steps: 

1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64
2. Rename the downloaded file to: frpc_linux_amd64_v0.2
3. Move the file to this location: /miniconda/envs/py38/lib/python3.8/site-packages/gradio
```

成功运行时，最后会出现：

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxx.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```

通过public URL即可访问。

### 方法2：Nginx自定义代理

#### 下载安装Nginx

查看是否已经安装nginx：

```shell
nginx -v
```

若未安装nginx，则需要安装：

Ubuntu:

```shell
apt-get install -y nginx
```

CentOS:

```shell
yum install -y nginx
```

#### 修改配置文件

Ubuntu参考`<path_to_dashinfer>/examples/nginx_config`目录下的`nginx_ubuntu.conf`，修改`/etc/nginx/sites-available/default`文件。

CentOS参考本文档目录下的`nginx_centos.conf`，修改`/etc/nginx/nginx.conf`文件。

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

- location要改成与gradio demo `launch()`中的`root_path参数`一致，示例代码中`root_path="/gradio-demo/"`
- proxy_pass中的端口要改成gradio demo部署时的端口一致，默认端口为`7860`
- 若同一台服务器部署多个gradio demo，可以配置多个location，映射到不同的端口

检查配置文件是否有语法错误：`nginx -t`

#### 启动Nginx服务

启动nginx服务：`nginx`

重启nginx服务：`nginx -s reload`

#### 修改launch()

gradio_demo_qwen.py，launch()函数设置`root_path="/gradio-demo/"`，例如：

``` python
demo.queue(api_open=False).launch(root_path="/gradio-demo/",
                                  height=800,
                                  share=False,
                                  server_name="127.0.0.1",
                                  server_port=7860)
```

成功运行后，访问URL：`http://server_ip_address/gradio-demo/`，即可进行交互。

- 对于局域网内的服务器，可以通过局域网ip地址访问
- 对于不处于同一个局域网的服务器，需要通过公网ip地址访问

## Step 3: 运行示例

在`<path_to_dashinfer>/examples/python/3_gradio`目录下运行示例python脚本：

```shell
python gradio_demo_qwen.py
```

终端出现如下输出后，可以通过浏览器访问部署好的Gradio应用：

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```
# 4_fastchat

[fastchat](https://github.com/lm-sys/FastChat)是一个开源的服务平台, 用于训练、服务和评估大语言模型聊天机器人。它提供以worker方式将一个推理引擎后端接入平台，提供兼容openai api的服务。
在[examples/python/4_fastchat/dashinfer_worker.py](../../examples/python/4_fastchat/dashinfer_worker.py)中，我们提供了一个使用FastChat与DashInfer实现worker的示例代码。用户仅需简单地将FastChat服务组件中的默认`fastchat.serve.model_worker`替换为`dashinfer_worker`，即可实现一个既兼容OpenAI API又能高效利用CPU资源进行推理的解决方案。


## Step 1: 安装fastchat
```shell
pip install "fschat[model_worker]"
```
## Step 2: 启动fastchat相关服务
```shell
python -m fastchat.serve.controller
python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

## Step 3: 启动dashinfer_worker
```shell
python dashinfer_worker.py --model-path qwen/Qwen-7B-Chat ../model_config/config_qwen_v10_7b.json
```

## Step 4: 使用cURL发送HTTP请求访问兼容openai api接口
```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen-7B-Chat",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```
## 使用Docker快速启动
此外，我们还提供了便捷的Docker镜像，使您能够快速部署一个集成dashinfer_worker并兼容OpenAI API的HTTP服务，只需执行以下命令：

启动Docker容器时，请确保替换尖括号内的路径为实际路径，并遵循以下命令格式：
```shell
docker run -d \
    --network host \
    -v <host_path_to_your_model>:<container_path_to_your_model> \
    -v <host_path_to_dashinfer_json_config_file>:<container_path_to_dashinfer_json_config_file> \
    dashinfer/fschat_ubuntu_x86:v1.2.1 \
   -m <container_path_to_your_model> \
    <container_path_to_dashinfer_json_config_file>
```

- <host_path_to_your_model>: host上存放ModelScope/HuggingFace模型的路径
- <container_path_to_your_model>: 要绑定到container中存放ModelScope/HuggingFace模型的路径
- <host_path_to_dashinfer_json_config_file>: host上DashInfer的json配置文件的路径
- <container_path_to_dashinfer_json_config_file>: 要绑定到container中DashInfer json配置文件的路径
- -m选项：表示container中ModelScope/HuggingFace的路径，取决于-v选项中host上路径绑定到container中的路径。若这里为ModelScope/HuggingFace中标准路径（例如：
qwen/Qwen-7B-Chat），那么不需要将host上模型路径绑定到container中，容器会自动为你下载模型。

下面是一个启动Qwen-7B-Chat模型服务的例子，默认host为localhost、端口为8000.
```shell
docker run -d \
    --network host \
    -v ~/.cache/modelscope/hub/qwen/Qwen-7B-Chat:/workspace/qwen/Qwen-7B-Chat  \
    -v examples/python/model_config/config_qwen_v10_7b.json:/workspace/config_qwen_v10_7b.json \
    dashinfer/fschat_ubuntu_x86:v1.2.1 \
    -m /workspace/qwen/Qwen-7B-Chat \
    /workspace/config_qwen_v10_7b.json
```

你还可以使用[openai_chat.py](../../examples/python/4_fastchat/openai_chat.py)来测试使用openai api的聊天客户端。

# 模型配置文件

`<path_to_dashinfer>/examples/python/model_config`目录下提供了一些config示例。

以下是对config中的参数说明：

- `model_name`: DashInfer模型名称，自定义；
- `model_type`: DashInfer模型类型，可选项：LLaMA_v2、ChatGLM_v2、ChatGLM_v3、Qwen_v10、Qwen_v15、Qwen_v20；
- `model_path`: DashInfer模型导出路径；
- `data_type`: 输出的数据类型，可选项：float32；
- `device_type`: 推理硬件，可选项：CPU；
- `device_ids`: 用于推理的NUMA节点，可以通过Linux命令`lscpu`查看CPU的NUMA信息；
- `multinode_mode`: 是否在多NUMA CPU上进行推理，可选项：true、false；
- `convert_config`: 模型转换相关参数；
    - `do_dynamic_quantize_convert`: 是否量化权重，可选项：true、false，目前仅ARM CPU支持量化；
- `engine_config`: 推理引擎参数；
    - `engine_max_length`: 最大推理长度，<= 11000；
    - `engine_max_batch`: 最大batch数；
    - `do_profiling`: 是否对推理过程进行profiling，可选项：true、false，若要进行profiling，需要设置`do_profiling = true`，并且设置环境变量`AS_PROFILE=ON`；
    - `num_threads`: 线程数，设置为单NUMA节点下的物理核数量时性能最佳，当设置的数值为0时，EngineHelper会自动解析`lscpu`的输出并设置，当设置的数值 > 0 时，采用设置的数值；
    - `matmul_precision`: 矩阵乘法的计算精度，可选项：high、medium，设置为high时，采用fp32进行矩阵乘法计算，设置为medium时，采用bf16进行计算；
- `generation_config`: 生成参数；
    - `temperature`: 随机过程 temperature；
    - `early_stopping`: 在生成stop_words_ids后是否停止生成，可选项：true、false；
    - `top_k`: 采样过程，top k 参数，top_k = 0 时对全词表排序；
    - `top_p`: 采样过程，top p 参数，0 <= top_p <= 1.0，top_p = 0 表示不使用topp；
    - `repetition_penalty`: The parameter for repetition penalty. 1.0 means no penalty.
    - `presence_penalty`: The parameter for presence penalty. 0.0 means no penalty.
    - `min_length`: 输入+输出的最小长度，默认为0，不启用filter；
    - `max_length`: 输入+输出的最大长度；
    - `no_repeat_ngram_size`: 用于控制重复词生成，默认为0，If set to int > 0, all ngrams of that size can only occur once.
    - `eos_token_id`: EOS对应的token，取决于模型；
    - `seed`: 随机数seed；
    - `stop_words_ids`: List of token ids of stop ids.
- `quantization_config`: 量化参数，当 do_dynamic_quantize_convert = true 时需要设置；
    - `activation_type`: 矩阵乘法的输入矩阵数据类型，可选项：bfloat16；
    - `weight_type`: 矩阵乘法的权重数据类型，可选项：uint8；
    - `SubChannel`: 是否进行权重sub-channel量化，可选项：true、false；
    - `GroupSize`: sub-channel量化粒度，可选项：64、128、256、512；
