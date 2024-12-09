'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_12_dynamic_quant.py
'''
import os
import gc
import time
import unittest

import modelscope

from dashinfer import allspark
from test_utils import LevenshteinCompare, CosineCompare, JaccardCompare, GenerateResultCompare
from test_util_infer import func_test_model_with_reference

# check log by:
# cat log | grep "output text" -A3 -B6


################################
# Supported Quant Config.
################################
# simplied quant config with per-channel int8
simpled_a16w8_per_channel_customized_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "int8"}

simpled_fp8a8w8_per_tensor_customized_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "fp8_e4m3",
    "compute_method": "activate_quant"}

simple_a16w8_group128_customized_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "int8",
    "group_size": 128}


simpled_a8w8_customized_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "int8",
    "compute_method" : "activate_quant"}

simpled_a16w4_customized_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "uint4"}

#####################
# Unsupported Config.
#####################

simpled_a8w8_group_128_customized_quant_config = {
    "quant_method": "instant_quant",
    "weight_format": "int8",
    "group_size": 128,
    "compute_method": "activate_quant"}

####################################


similarity_test_cases = {
    "qwen/Qwen-7B-Chat":
        {"model_name": "qwen/Qwen-7B-Chat", "input": [
            "帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了"],
         "reference": [
             "你犹如璀璨星辰，照亮了我们的项目之路；你的存在，如同瑰宝般珍贵，让我们的项目熠熠生辉。没有你的参与，我们的项目就如同失去灵魂的躯壳，注定走向失败。感谢你的付出和努力，让我们能够成功完成这个项目。你是我们团队的灵魂，是我们项目的无价之宝"],
         "lang": "zh",
         "compare": CosineCompare(), "threshold": 0.1
         },
    "qwen/Qwen2-7B-Instruct":
        {"model_name": "qwen/Qwen2-7B-Instruct", "input": ["静夜思这首诗是谁写的？只回答作者名字。"],
         "reference": ["李白<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "Qwen/Qwen2.5-7B-Instruct-weight-only":
        {"model_name": "Qwen/Qwen2.5-7B-Instruct", "input": [
            "帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了"],
         "reference": [
             "没有您的鼎力相助，此项目恐将陷入万劫不复之境，最终化为乌有！<|im_end|>"],
         "generation_params": {"top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1, "temperature": 0.7, "seed": 1234},
         "lang": "zh",
         "compare": CosineCompare(), "threshold": 0.1
         },
    "Qwen/Qwen2.5-7B-Instruct-active-quant":
        {"model_name": "Qwen/Qwen2.5-7B-Instruct", "input": [
            "帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了"],
         "reference": [
             "没有您的鼎力相助与倾情投入，此项目早已在无形中折翼沉沦，化为乌有矣！<|im_end|>"],
         "generation_params": {"top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1, "temperature": 0.7, "seed": 1234},
         "lang": "zh",
         "compare": CosineCompare(), "threshold": 0.1
         },
    "Qwen/Qwen2.5-7B-Instruct-active-quant-fp8":
        {"model_name": "Qwen/Qwen2.5-7B-Instruct", "input": [
            "帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了"],
         "reference": [
             "没有您的鼎力相助与倾情投入，此项目早已在无形中折翼沉沦，化为乌有矣！<|im_end|>"],
         "generation_params": {"top_k": 20, "top_p": 0.8, "repetition_penalty": 1.1, "temperature": 0.7, "seed": 1234},
         "lang": "zh",
         "compare": CosineCompare(), "threshold": 0.0
         },
    "Qwen/Qwen2.5-7B-Instruct-a16w4":
        {"model_name": "Qwen/Qwen2.5-7B-Instruct", "input": [
            "帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了"],
         "reference": [
             "没有您的鼎力支持，此项目犹如一叶扁舟在惊涛骇浪中沉没，幸得巨浪化为甘霖，方得凤凰涅槃，重获新生。您之贡献，犹如璀璨星辰点缀夜空，令整个宇宙熠熠生辉。<|im_end|>"],
         "lang": "zh",
         "compare": CosineCompare(), "threshold": 0.0
         },
}


class ModelSimilarityTest(unittest.TestCase):
    def setUp(self):
        self.similarity_test_cases = similarity_test_cases
        self.engine = allspark.Engine()
        # 创建模型实例

    def tearDown(self):
        self.engine = None
        gc.collect()

    def func_test_model_with_reference(self, test_dict, init_quant=False, test=None, weight_only_quant=True) -> float:
        # self.engine = None
        # let engine destroy, free all resources.
        # gc.collect()
        # full gc, make engine destroy called.
        return func_test_model_with_reference(test_dict, self.engine, init_quant, test,
                                              weight_only_quant=weight_only_quant)



    def test_a16w8_per_chn_qwen1(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen-7B-Chat"], init_quant=True,
                                       weight_only_quant=True, device_list=[0], quant_config=simpled_a16w8_per_channel_customized_quant_config, test=self)

    def test_a16w8_per_chn_qwen25(self):
        func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-weight-only"], init_quant=True,
                                       weight_only_quant=True, device_list=[0], quant_config=simpled_a16w8_per_channel_customized_quant_config,
                                       test=self)
    def test_a16w8_per_chn_kv8_qwen25(self):
        func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-weight-only"], init_quant=True,
                                       weight_only_quant=True, device_list=[0], quant_config=simpled_a16w8_per_channel_customized_quant_config,
                                       test=self)

    def test_a16w8_per_chn_kv4_qwen25(self):
        func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-weight-only"], init_quant=True,
                                       weight_only_quant=True, device_list=[0], quant_config=simpled_a16w8_per_channel_customized_quant_config, cache_quant_mode="8",
                                       test=self)

    def test_a16w8_sub_chn_qwen25(self):
        func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-weight-only"], init_quant=True,
                                       weight_only_quant=True, device_list=[0], quant_config=simple_a16w8_group128_customized_quant_config, test=self)

    def test_a8w8_per_chn_qwen25(self):
        func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-active-quant"], init_quant=True,
                                       weight_only_quant=True, device_list=[0],  quant_config=simpled_a8w8_customized_quant_config, test=self)
        
    def test_fp8_a8w8_per_tensor_qwen25(self):
        import torch
        if torch.cuda.get_device_capability()[0] == 9 or (torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] == 9):
            func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-active-quant-fp8"], init_quant=True,
                                        weight_only_quant=True, device_list=[0],  quant_config=simpled_fp8a8w8_per_tensor_customized_quant_config, test=self)
        else:
            pass

    def test_a16w4_per_chn_qwen25_bf16(self):
        func_test_model_with_reference(similarity_test_cases["Qwen/Qwen2.5-7B-Instruct-a16w4"], init_quant=True,
                                       weight_only_quant=True, device_list=[0], quant_config=simpled_a16w4_customized_quant_config, test=self)

    def test_a16w4_sub_chn_qwen25(self):
        # TODO: add expect exception test case.
        pass


if __name__ == '__main__':
    unittest.main()
