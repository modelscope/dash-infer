## Quick Start Guide for OpenAI API Server

### Start OpenAI Server with Docker

We have provide a Docker image to start OpenAI server.
This example demonstrates how to use Docker to run DashInfer as an inference engine, providing OpenAI API endpoints.

```shell
docker run \
    --shm-size=8g \
    --network=host \
    --ipc=host \
    --gpus=all \
    -v=<host_path_to_your_model>:<container_path_to_your_model> \
    dashinfer/fschat_ubuntu_cuda:v2.0 \
    -h <host> -p <port> -m -- --model-path <container_path_to_your_model> --device-list <device_list>
```
 
- `<host_path_to_your_model>`: Path to your model on the host
- `<container_path_to_your_model>`: Path where the model is mounted in the container
- `<device_list>`: List of devices supported by the model, e.g., `0,1`
- `-h`: The listening address of the API server
- `-p`: The listening port of the API server
- `-m`: Use Modelscope to download the model
- `--model-path`: Path for loading or downloading the model
- `--device-list`: List of CUDA devices used to run the model

For example:

```shell
docker run  \
    --shm-size=8g \
    --network=host \
    --ipc=host \
    --gpus=all \
    -v=/mnt/models/modelscope/hub/qwen/Qwen2.5-7B-Instruct:/models/qwen/Qwen2.5-7B-Instruct \
    dashinfer/fschat_ubuntu_cuda:v2.0 \
    -h 127.0.0.1 -p 8088 -m -- --model-path /models/qwen/Qwen2.5-7B-Instruct --device-list 0
```

You can also build you owner fastchat Docker image by modifying the Docker file `scripts/docker/fschat_ubuntu_cuda.Dockerfile`.

### Testing the OpenAI API Server

#### Testing with OpenAI SDK
In `examples/api_server/fschat/openai-client.py`, the official OpenAI SDK is used to test the API server.

```shell
python examples/api_server/fschat/openai-client.py
```

#### Testing with curl
Assuming the OpenAI Server has been started and the port number is `8088`, you can use the following command:

```shell
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

