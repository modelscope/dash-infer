'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    openai_chat.py
'''
#!/usr/bin/env python3

import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8088/v1/"

def test_list_models():
    model_list = openai.models.list()
    names = [x.id for x in model_list.data]
    return names

def test_chat_completion_stream(model):
    messages = [{"role": "user", "content": "Talk about the impact of artificial intelligence on different aspects of society. Please talk at least 1000 words"}]
    res = openai.chat.completions.create(
        model=model, messages=messages, stream=True, temperature=0
    )
    for chunk in res:
        try:
            content = chunk.choices[0].delta.content
            if content is None:
                content = ""
        except Exception as e:
            content = chunk.choices[0].delta.get("content", "")
        print(content, end="", flush=True)
    print()

def test_chat_completion(model):
    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello! What is your name?"}],
        temperature=0,
    )
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    model = "Qwen2-1.5B-Instruct" # the model name which is typically the last component of model path

    print("List models:")
    print(test_list_models())

    print("\n\n====================================")
    print("Chat completion:")
    test_chat_completion(model)

    print("\n\n====================================")
    print("Chat completion stream:")
    test_chat_completion_stream(model)