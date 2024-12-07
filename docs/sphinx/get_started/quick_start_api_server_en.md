## Quick Start Guide for OpenAI API Server

### Start OpenAI Server with Docker

We have provide a Docker image to start OpenAI server.
This example demonstrates how to use Docker to run DashInfer as an inference engine, providing OpenAI API endpoints.

TODO [Remove this after Image upload is done.]

You can also build you owner fastchat Docker image by modifying the Docker file `scripts/docker/fschat-hie-allspark-cuda.Dockerfile`.

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

