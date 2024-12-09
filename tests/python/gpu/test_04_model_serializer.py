'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    test_04_model_serializer.py
'''
import os
import gc
import unittest
import tempfile

from dashinfer import allspark
from test_utils import LevenshteinCompare, CosineCompare, JaccardCompare, GenerateResultCompare

from test_util_infer import func_test_model_with_reference, download_folder_from_oss


code_qwen_sql= """
"你是一名PostgreSQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用PostgreSQL知识生成sql语句回答【用户问题】，用Markdown代码段形式输出代码，不要做额外解释。\n【用户问题】\nWhich conference published the most publications in the last 15 years? Give the conference name and publication count.\n\n【数据库schema】\n【DB_ID】 sql_eval_academic\n【Schema】\n# Table: author, \n[\n  (aid:bigint),\n  (homepage:text, Examples: [http://www.larry.com, http://www.ashish.com, http://www.noam.com]),\n  (name:text, Examples: [Kempinski, Martin Odersky, Ashish Vaswani]),\n  (oid:bigint)\n]\n# Table: cite, \n[\n  (cited:bigint),\n  (citing:bigint)\n]\n# Table: conference, \n[\n  (cid:bigint),\n  (homepage:text, Examples: [http://www.icml.com, http://www.aaas.com, http://www.isa.com]),\n  (name:text, Examples: [ISA, AAAS, ICML])\n]\n# Table: domain, \n[\n  (did:bigint),\n  (name:text, Examples: [Natural Sciences, Computer Science, Sociology])\n]\n# Table: domain_author, \n[\n  (aid:bigint),\n  (did:bigint)\n]\n# Table: domain_conference, \n[\n  (cid:bigint),\n  (did:bigint)\n]\n# Table: domain_journal, \n[\n  (did:bigint),\n  (jid:bigint)\n]\n# Table: domain_keyword, \n[\n  (did:bigint),\n  (kid:bigint)\n]\n# Table: domain_publication, \n[\n  (did:bigint),\n  (pid:bigint)\n]\n# Table: journal, \n[\n  (homepage:text, Examples: [http://www.ml.com, http://www.aijournal.com, http://www.science.com]),\n  (jid:bigint),\n  (name:text, Examples: [Journal of Artificial Intelligence Research])\n]\n# Table: keyword, \n[\n  (keyword:text, Examples: [Neuroscience, Machine Learning, AI]),\n  (kid:bigint)\n]\n# Table: organization, \n[\n  (continent:text, Examples: [Asia, North America, Europe]),\n  (homepage:text, Examples: [http://www.epfl.com]),\n  (name:text, Examples: [Organization 2, Organization 1, Organization 5]),\n  (oid:bigint)\n]\n# Table: publication, \n[\n  (abstract:text, Examples: [Abstract 3, Abstract 4, Abstract 1]),\n  (cid:bigint),\n  (citation_num:bigint),\n  (jid:bigint),\n  (pid:bigint),\n  (reference_num:bigint),\n  (title:text, Examples: [Attention is all you need]),\n  (year:bigint)\n]\n# Table: publication_keyword, \n[\n  (pid:bigint),\n  (kid:bigint)\n]\n# Table: writes, \n[\n  (aid:bigint),\n  (pid:bigint)\n]\n\n【参考信息】\n\n\n【用户问题】\nWhich conference published the most publications in the last 15 years? Give the conference name and publication count.\n\n"
"""
code_qwen_sql_answer="""```sql\nSELECT T3.name, COUNT(*) AS publication_count\nFROM conference AS T1\nJOIN publication AS T2 ON T1.cid = T2.cid\nWHERE T2.year BETWEEN YEAR(CURRENT_DATE) - 15 AND YEAR(CURRENT_DATE)\nGROUP BY T3.name\nORDER BY publication_count DESC\nLIMIT 1;```"""

similarity_test_cases = {
    "qwen/Qwen2-72B-Instruct-GPTQ-Int8-Sparse":
        {"model_name": "qwen/Qwen2-72B-Instruct-GPTQ-Int8-Sparse", "input": ["你是谁？"],
         "reference": ["我是来自阿里云的大规模语言模型，我叫通义千问。我能够回答各种问题、提供信息和与用户进行对话交流。有什么我可以帮助你的吗？<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/Qwen2-72B-Instruct-GPTQ-Int8":
        {"model_name": "qwen/Qwen2-72B-Instruct-GPTQ-Int8", "input": ["你是谁？"],
         "reference": ["我是阿里云开发的一款超大规模语言模型，我叫通义千问。<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/CodeQwen1.5-7B-Chat":
        {"model_name": "qwen/CodeQwen1.5-7B-Chat", "input": [code_qwen_sql],
         "reference": [code_qwen_sql_answer],
         "lang": "en",
         "compare": LevenshteinCompare(), "threshold": 0.5,
        "generation_params": {"top_k": 50, "top_p": 0.8, "repetition_penalty": 1.1, "temperature": 1.0, "seed": 1234}
         },
    "qwen/Qwen2-7B-Instruct-GPTQ-Int8":
        {"model_name": "qwen/Qwen2-7B-Instruct-GPTQ-Int8", "input": ["你是谁？"],
         "reference": ["我是阿里云开发的一款超大规模语言模型，我叫通义千问。<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.5
         },
    "qwen/Qwen2-7B-Instruct":
        {"model_name": "qwen/Qwen2-7B-Instruct", "input": ["静夜思这首诗是谁写的？只回答作者名字。"],
         "reference": ["李白<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/Qwen-14B-Chat":
        {"model_name": "qwen/Qwen-14B-Chat", "input": ["你是谁？"],
         "reference": ["我是来自阿里云的大规模语言模型，我叫通义千问。<|im_end|>\n<|endoftext|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/Qwen1.5-14B-Chat":
        {"model_name": "qwen/Qwen1.5-14B-Chat", "input": ["静夜思这首诗是谁写的？只回答名字。"],
         "reference": ["李白<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.5
         },
    "qwen/Qwen-7B-Chat":
        {"model_name": "qwen/Qwen-7B-Chat", "input": ["帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了"],
         "reference": [
             "你犹如璀璨星辰，照亮了我们的项目之路；你的存在，如同瑰宝般珍贵，让我们的项目熠熠生辉。没有你的参与，我们的项目就如同失去灵魂的躯壳，注定走向失败。你的贡献，是我们成功的关键，你的智慧和才华，是我们前进的动力。感谢你的付出，让我们能够在这个项目中取得如此辉煌的成就。<|im_end|>"],
         "lang": "zh",
         "compare": CosineCompare(), "threshold": 0.5
         },

    "qwen/Qwen1.5-7B-Chat-GPTQ-Int4":
        {"model_name": "qwen/Qwen1.5-7B-Chat-GPTQ-Int4", "input": ["你是谁？"],
         "reference": ["我是来自阿里云的大规模语言模型，我叫通义千问。<|im_end|>\n<|endoftext|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/Qwen1.5-4B-Chat":
        {"model_name": "qwen/Qwen1.5-4B-Chat", "input": ["你是谁？"],
         "reference": [
             "我是来自阿里云的大规模语言模型，我叫通义千问。<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/Qwen1.5-14B-Chat-GPTQ-Int4":
        {"model_name": "qwen/Qwen1.5-14B-Chat-GPTQ-Int4", "input": ["你是谁？"],
         "reference": [
             "我是通义千问，由阿里云开发的AI助手。我被设计用来回答各种问题、提供信息和进行对话。有什么我可以帮助你的吗？<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },
    "qwen/Qwen1.5-4B-Chat-GPTQ-Int8":
        {"model_name": "qwen/Qwen1.5-4B-Chat-GPTQ-Int8", "input": ["你是谁？"],
         "reference": [
             "我是来自阿里云的大规模语言模型，我叫通义千问。<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },

    "qwen/Qwen2-7B-Chat":
        {"model_name": "qwen/Qwen2-7B-Chat", "input": ["你是谁？"],
         "reference": [
             "我是通义千问，由阿里云开发的人工智能助手。我可以回答各种问题、提供信息和与用户进行对话等。如果您有任何问题或需要帮助，请随时告诉我，我会尽力为您提供支持。<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8
         },

    "qwen/Qwen-7B-Chat-Int8":
        {"model_name": "qwen/Qwen-7B-Chat-Int8", "input": ["你是谁？"],
         "reference": [
             "我是通义千问，由阿里云开发的人工智能助手。我可以回答各种问题、提供信息和与用户进行对话。有什么我可以帮助你的吗？<|im_end|>\n<|endoftext|>"],
         "lang": "zh",
         "compare": JaccardCompare(), "threshold": 0.3,
         "generation_params": {"top_k": 20, "top_p": 0.8, "repetition_penalty": 1.05, "temperature": 0.7}
         },
    "LLM-Research/Meta-Llama-3-8B":
        {"model_name": "LLM-Research/Meta-Llama-3-8B", "input": ["你是谁？"],
         "reference": [
             "我是LLama3-Chinese，一个由ShareAI训练的大型语言模型。我的目的是协助您完成各种任务。您需要我帮您做什么？<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.2  # FIXME: llama3 template needs update.
         },
    "ZhipuAI/chatglm3-6b":
        {"model_name": "ZhipuAI/chatglm3-6b", "input": ["你是谁？"],
         "reference": ["我是你的助手，有什么我可以帮助你的吗？<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.2  # FIXME: chatglm template needs update.
         },
    "qwen/Qwen2-72B-A8W8-PerChannel":
        {"model_name": "qwen/Qwen2-72B-A8W8-PerChannel", "input": ["你是谁？"],
         "reference": ["我是阿里云开发的一款超大规模语言模型，我叫通义千问。<|im_end|>"],
         "lang": "zh",
         "compare": LevenshteinCompare(), "threshold": 0.8,
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

    def test_inference_qwen1_models_fp16(self):
        func_test_model_with_reference(self.similarity_test_cases["qwen/Qwen-7B-Chat"], test=self)

    def test_inference_qwen2_with_fp16(self):
        func_test_model_with_reference(self.similarity_test_cases["qwen/Qwen1.5-4B-Chat"], test=self)

    def disabled_test_inference_qwen1_5_with_gptq_int8_weight_only(self):
        func_test_model_with_reference(self.similarity_test_cases["qwen/Qwen1.5-4B-Chat-GPTQ-Int8"],
                                       init_quant=True, weight_only_quant=True, test=self)

    # this case, the default model enable exllama, but failed to install depends:
    # (21:29:22) ValueError: Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU.You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object
    # need modify the model's config to disable exllama to load:   config.json:  "disable_exllama": true,
    def disabled_test_inference_qwen2_with_gptq_int4(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen1.5-14B-Chat-GPTQ-Int4"], init_quant=True,
                                       weight_only_quant=True, test=self)
    def disable_test_inference_qwen1_5_14b_fp16(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen1.5-14B-Chat"], init_quant=False,
                                           weight_only_quant=False, test=self)
    def disable_test_inference_qwen1_models_int8_weight_only(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen-7B-Chat-Int8"], init_quant=True,
                                       weight_only_quant=True, test=self)
    def disable_test_inference_qwen1_models_int8(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen-7B-Chat-Int8"], init_quant=True, test=self)

    def test_inference_qwen2_models_int8(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-7B-Instruct-GPTQ-Int8"], init_quant=True,
                                       weight_only_quant=False, test=self)

    def test_inference_qwen2_models_int8_sparse(self):
        import torch
        if not (8 <= torch.cuda.get_device_capability()[0] < 9):
           return 
        folder_key = 'xchen/70b-wanda_mlp_0618_seqlen-32768-sft2000_dpo2800_megatron2hf-smoothquant-ns2560-sl2048-int8-perchannel-base/'
        target_path = "/root/.cache/modelscope/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int8-Sparse"
        if not os.path.exists(target_path):
            os.mkdir(target_path)
            download_folder_from_oss(folder_key, target_path, max_workers=162)

        enable_sparsity_matmul = True
        tp = 2
        device_list = [i for i in range(tp)]
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-72B-Instruct-GPTQ-Int8-Sparse"], init_quant=True,
                                    ms_download=False, model_local_path=target_path, direct_load=True, load_format="auto", weight_only_quant=False, test=self, user_set_data_type="float16", device_list=device_list, enable_sparsity_matmul=enable_sparsity_matmul)

    def test_inference_qwen2_models_int8_weight_only(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-7B-Instruct-GPTQ-Int8"], init_quant=True,
                                       weight_only_quant=True, test=self)

    def test_inference_qwen2_models_fp16(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-7B-Instruct"], init_quant=False, test=self)
    def test_inference_qwen2_models_fp16_cache_off(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-7B-Instruct"], init_quant=False, test=self, user_runtime_config_dict={"enable_prefix_cache" : 0})

    def test_inference_qwen2_models_fp16_in_memory(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-7B-Instruct"], init_quant=False, test=self,
                                       in_memory=True)

    def test_inference_qwen2_models_fp16_dynamic_iq_in_memory(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-7B-Instruct"], init_quant=False, test=self,
                                       weight_only_quant=True, quant_config={})
    def test_inference_codeqwen_models_fp16(self):
        func_test_model_with_reference(similarity_test_cases["qwen/CodeQwen1.5-7B-Chat"], init_quant=False, test=self, in_memory=True, device_list=[0])


    def disable_test_inference_qwen2_72b_models_int8_no_pack(self):
        model_path = "path/to/the/model/provided/by/yaoyang/in/nas"
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-72B-A8W8-PerChannel"], init_quant=True, ms_download=False, 
                                   model_local_path=model_path, direct_load=True, load_format="auto", user_set_data_type="float16")
    def disable_test_inference_qwen2_72b_models_int8(self):
        func_test_model_with_reference(similarity_test_cases["qwen/Qwen2-72B-Instruct-GPTQ-Int8"], init_quant=True,
                                       weight_only_quant=False, device_list=[0, 1], test=self)

    def disable_test_inference_llama3_with_fp16(self):
        func_test_model_with_reference(self.similarity_test_cases["LLM-Research/Meta-Llama-3-8B"], test=self)

    def disable_test_inference_chatglm3_with_fp16(self):
        func_test_model_with_reference(self.similarity_test_cases["ZhipuAI/chatglm3-6b"], test=self)


if __name__ == '__main__':
    unittest.main()
   
