'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_openai_chat_completion.py
'''
from openai import OpenAI


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
    )
    return response


if __name__ == "__main__":
    openai_api_key = "EMPTY"
    openai_api_base = "http://127.0.0.1:8000/v1"

    model = "model"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    try:
        model = client.models.list().data[0].id
    except Exception:
        model = "model"

    gen_text = test_text_image_1(client, model)
    print(gen_text)

    gen_text = test_text_multi_images(client, model)
    print(gen_text)

    gen_text = test_text_video_file(client, model)
    print(gen_text)
