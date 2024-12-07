'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    prompt_utils.py
'''
import copy
from jinja2 import Template


class PromptTemplate:

    @staticmethod
    def apply_chatml_template(inputs):
        start_text = "<|im_start|>"
        end_text = "<|im_end|>"
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        user_msg = {"role": "user", "content": ""}
        assistant_msg = {"role": "assistant", "content": ""}

        prompt_template = Template(
            "{{start_text}}" + "{{system_role}}\n" + "{{system_content}}" + "{{end_text}}\n" +
            "{{start_text}}" + "{{user_role}}\n" + "{{user_content}}" + "{{end_text}}\n" +
            "{{start_text}}" + "{{assistant_role}} \n")

        gen_cfg_list = []
        user_msg["content"] = copy.deepcopy(inputs)

        prompt = prompt_template.render(start_text=start_text, end_text=end_text,
                                        system_role=system_msg["role"], system_content=system_msg["content"],
                                        user_role=user_msg["role"], user_content=user_msg["content"],
                                        assistant_role=assistant_msg["role"])
        return prompt
