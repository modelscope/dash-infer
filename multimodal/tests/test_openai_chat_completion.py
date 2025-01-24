'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_openai_chat_completion.py
'''
from openai import OpenAI
import argparse

def test_text_image_1(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://farm4.staticflickr.com/3075/3168662394_7d7103de7d_z_d.jpg",
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=1024,
        temperature=0.1,
        frequency_penalty=1.05,
        stream=True
    )
    return response


def test_text_multi_images(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Are these images different?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://farm4.staticflickr.com/3075/3168662394_7d7103de7d_z_d.jpg",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://farm9.staticflickr.com/8505/8441256181_4e98d8bff5_z_d.jpg",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://farm3.staticflickr.com/2220/1572613671_7311098b76_z_d.jpg",
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=1024,
        top_p=0.5,
        temperature=0.1,
        frequency_penalty=1.05,
        stream=True
    )
    return response


def test_text_video_file(client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a compelling description that I can upload along with the video.",
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://cloud.video.taobao.com/vod/JCM2awgFE2C2vsACpDESXZ3h5_iQ5yCZCypmjtEs2Ck.mp4",
                            "fps": 2,
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=1024,
        top_p=0.5,
        temperature=0.1,
        frequency_penalty=1.05,
        stream=True
    )
    return response

def main(args, client):
    try:
        model = client.models.list().data[0].id
    except Exception:
        model = "model"

    test_cases = {
        "singe_image": test_text_image_1,
        "multi_images": test_text_multi_images,
        "video": test_text_video_file
    }

    if args.type == "all":
        for key, func in test_cases.items():
            print(f"running {key} case")
            response = func(client, model)
            for chunk in response:
                print(chunk.choices[0].delta.content, end='', flush=True)
            print("\n")
    else:
        print(f"running {test_cases.keys()} cases")
        func = test_cases[args.type]
        response = func(client)
        for chunk in response:
            print(chunk.choices[0].delta.content, end='', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str,
                        default="0.0.0.0")
    parser.add_argument('--port', type=str,
                        default="8000")
    parser.add_argument('--type', type=str, default="all", choices=["all", "singe_image", "multi_images", "video"])
    args = parser.parse_args()
    
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"

    model = "model"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    main(args, client)
    
