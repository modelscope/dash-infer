'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    PromptLoader.py
'''
import os
import copy
import json
import numpy as np
import subprocess
import enum
from itertools import accumulate
from typing import List, Tuple, Dict, Union, Any

class OnlineLogType(enum.Enum):
    TYPE_0 = 0
    TYPE_1 = 1   # Format like: {"input": {"prompt": "xxx"}}, "parameters": {"temperature" :xxx, "top_p": xxx, "seed": 1234}}
    TYPE_2 = 2   # Format like: {"input": {"messages": [{"content": "xxx"}]}, "parameters": {"seed": 1234}, "request_id": "xyz"}
    TYPE_3 = 3   # codeqwen2, Format like: {"payload": {"input": {"prompt": "xxx"}}, "parameters": {"temperature" :xxx, "top_p": xxx}}

def simple_prompt():
    prompt = [
        "浙江的省会在哪",
        "将“温故而知新”翻译成英文，并解释其含义",
        "“就像踢足球都是盘带、射门，但要做到梅西那么好也不容易。”请把这句话翻译成英文",
        "西红柿鸡蛋怎么做", # "西红柿炒鸡蛋怎么做",
        # "帮我用华丽的词藻润色夸奖文案，一定要非常浮夸，天花乱坠一点，可以简短一些。下面是我的原始文本：没有你的参与项目就失败了",
        # "Where is the capital of Zhejiang?",
        # "How many days are in a leap year?",
        # "What is the largest planet in our solar system?",
        # "How many continents are there in the world?",
        # "What is the best way to preserve fresh fruits and vegetables?",
        # "What is the average body temperature of a healthy adult?",
        # "What is the largest organ in the human body?",
        # "The following are multiple choice questions (with answers) about abstract algebra.\nThe cyclic subgroup of Z_24 generated by 18 has order\nA. 4\nB. 8\nC. 12\nD. 6\nAnswer:",
        # "The following are multiple choice questions (with answers) about abstract algebra.\nFind the order of the factor group Z_6/<3>.\nA. 2\nB. 3\nC. 6\nD. 12\nAnswer:"
    ]

    encoded_prompt = [
        # qwen
        [151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,105250,100158,2073,99416,99535,68536,52183,16628,854,151645,198,151644,77091],
        # [151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,1036,104092,105867,102096,100132,99665,99278,5373,99759,64689,3837,77288,114592,109139,100624,52801,99744,100047,32945,14880,99360,106599,105395,12857,105205,151645,198,151644,77091,198],
        # [151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,102661,356,5207,43268,20412,99251,101417,9370,151645,198,151644,77091,198,108724,107076,1162,46,43268,20412,99251,101417,9370,1773,151645,198,151644,872,198,107076,356,5207,43268,20412,100668,101417,9370,151645,198,151644,77091,198],
        # [151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,117700,105703,109031,151645,198,151644,77091],
        [151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,117700,105703,109031,151645,198,151644,77091,198],
        # [151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,117700,101311,105703,109031,151645,198,151644,77091],
        # llama2
        # [1,29961,25580,29962,3532,14816,29903,6778,13,6293,526,263,8444,20255,29991,13,29966,829,14816,29903,6778,13,13,3831,852,385,3033,6751,9850,12618,1400,1048,263,7786,17487,304,26901,29875,29892,12141,292,16375,27482,322,1818,29899,4149,19650,1953,29889,518,29914,25580,29962],
    ]

    return prompt


def simple_common_prefix_prompt():
    common_content1 = "基于给定的文本回答问题，只需要给出问题的答案，不需要说其他内容。\n\n以下是给定的文本：\n王海江诈骗罪一审刑事判决书\n公诉机关张家口市桥东区人民检察院。\n张家口市桥东区人民检察院以张东检一部刑诉（2019）1号起诉书指控被告人王海江犯诈骗罪，于2020年1月16日向本院提起公诉。本院受理后，依法组成合议庭并适用普通程序，公开开庭审理了本案。张家口市桥东区人民检察院指派检察员闫鑫出庭支持公诉，被告人王海江到庭参加诉讼。因受新冠疫情影响，本案于2020年2月6日中止诉讼，2020年7月9日恢复审理，现已审理终结。\n张家口市桥东区人民检察院起诉书指控：2019年5月至2019年7月，被告人王海江因无法偿还被害人张某的债务，虚构“高乐”、“赵佳”、“刘律师”、“胡建国”，并以上述人员的名义和被害人张某通过微信联系，谎称“高乐”、“赵佳”欠王海江钱，“高乐”、“赵佳”同意替王海江偿还欠张某的债务，“高乐”名下的一对公账户中有20余万元，但该对公账户被冻结，需要通过刷银行流水解封对公账户，只要解封冻结账户便可以偿还王海江欠张某的债务。被害人张某信以为真，为解封账户，索要欠款，共向被告人汇款1802100元。被告人将大部分赃款用于网络赌博及个人消费。被告人到案后自愿认罪认罚。\n公诉机关就上述指控提供了：手机，受案登记表、立案决定书，抓获经过，银行流水，微信账单，手机号户主查询记录，亚鑫彩印工商信息，建行对公账户管理规定，办案说明，证人李某、陈某的证言，被害人张某的陈述，被告人王海江的供述与辩解，辨认笔录，收集数据恢复报告等证据。\n公诉机关认为，被告人王海江以非法占有为目的，虚构事实、隐瞒真相，骗取被害人张某人民币1802100元，数额特别巨大，其行为触犯了《中华人民共和国刑法》第二百六十六条之规定，犯罪事实清楚，证据确实、充分，应当以诈骗罪追究其刑事责任。被告人自愿认罪认罚，对犯罪事实如实供述，建议判处被告人有期徒刑十一年，并处罚金。提起公诉，请依法判处。\n被告人王海江对公诉机关的指控无异议，当庭自愿认罪，但辩解手机微信恢复的数据证实自己有中止犯罪的行为，张某给转钱时候都有实名认证，他应该知道都是给自己打钱。\n经审理查明，2019年5月至2019年7月间，被告人王海江因无法偿还被害人张某的债务，虚构“高乐”、“赵佳”、“刘律师”、“胡建国”，并以上述人员的名义与被害人张某通过微信联系，谎称“高乐”、“赵佳”欠王海江钱，“高乐”、“赵佳”同意替王海江偿还欠张某的债务。王海江编造“高乐”名下的一个对公账户中有20余万元被冻结，需要通过刷银行流水解封对公账户，只要解除冻结，便可以偿还王海江欠张某债务的事实，骗取被害人信任。被害人张某为解封账户，索要欠款，共向被告人汇款1802100元。被告人将170余万元转付蒙扁立、农进军用于网络赌博，其余用于个人消费。2019年8月12日王海江被抓获。王海江到案后如实供述上述犯罪事实。\n在审查起诉阶段，被告人王海江就指控的其犯诈骗罪签署认罪认罚具结书，同意公诉机关指控本人的罪名及有期徒刑十一年并处罚金的量刑建议。公诉机关出具量刑建议书建议判处被告人有期徒刑十一年，并处罚金。\n上述事实，被告人王海江在庭审过程中均无异议，且有公诉机关提交并经开庭质证、认证的：手机，受案登记表、立案决定书，抓获经过，银行流水，微信账单，手机号户主查询记录，亚鑫彩印工商信息，建行对公账户管理规定，办案说明，证人李某、陈某的证言，被害人张某的陈述，被告人王海江的供述与辩解，辨认笔录;收集数据恢复报告等证据予以证实，足以认定。\n本院认为，被告人王海江以非法占有为目的，虚构事实、隐瞒真相，骗取被害人张某人民币1802100元，数额特别巨大，其行为触犯了《中华人民共和国刑法》第二百六十六条之规定，构成诈骗罪。公诉机关指控的犯罪事实清楚，证据确实、充分，指控罪名成立。关于被告人认为被害人明知付款对象仍然转款的辩解意见，经查，与其在侦查阶段的供述不一致，且不符合逻辑思维关系，故不予采纳。关于被告人认为其构成犯罪中止的意见，经查，其本人没有自动放弃犯罪或者积极有效的防止犯罪结果的发生，故不予采纳。被告人王海江到案后如实供述犯罪事实，系坦白，依法从轻处罚。被告人在审查起诉阶段认罪认罚，依法从宽处理。公诉机关的量刑建议适当，予以采纳。依照《中华人民共和国刑法》第二百六十六条、第五十二条、第五十三条、第六十四条、第六十七条第三款，《最高人民法院、最高人民检察院关于办理诈骗刑事案件具体应用法律若干问题的解释》第一条第一款，《中华人民共和国刑事诉讼法》第十五条、第二百零一条之规定，判决如下：\n一、被告人王海江犯诈骗罪，判处有期徒刑十一年，并处罚金三万元；\n（刑期从判决执行之日起计算，判决执行以前先行羁押的，羁押一日折抵刑期一日，即自2019年8月12日起至2030年8月11日止。罚金于本判决生效之日起十日内缴纳。）\n二、被告人诈骗被害人张某的赃款1802100元予以追缴，发还被害人；\n三、犯罪工具华为（黑色）手机一部，予以没收。\n如不服本判决，可在接到判决书的第二日起十日内，通过本院或者直接向河北省张家口市中级人民法院提出上诉，书面上诉的，提交上诉状正本一份，副本二份。\n基于以上文本回答问题，只需要给出问题的答案，不需要说其他内容。\n\n"

    common_content2 = "基于给定的文本回答问题，只需要给出问题的答案，不需要说其他内容。\n\n以下是给定的文本：\n'皖教秘师〔2022〕100 号 安徽省教育厅关于举办 2022 年长三角 基础教育校长发展高端论坛的通知 各市、省直管县（市）教育局（教育体育局）： 为深入学习贯彻党的二十大精神,贯彻落实《新时代基础教 育强师计划》，推动我省加快融入长三角地区基础教育更高质量 一体化发展步伐，根据《安徽省人民政府办公厅关于印发省政府 四季度重点工作安排的通知》（皖政办秘〔2022〕53 号）有关 要求，决定举办 2022 年长三角基础教育校长发展高端论坛。现 将相关事宜通知如下。 一、论坛主题 区域名校长的成长 二、主要议题 1.学校师德教育方法创新 2.学校管理与教育评价改革 3.课程改革背景下的高素质教师队伍建设 \n4.教育数字化转型中的学校 三、论坛时间 2022 年 12 月 30 日(周五)上午 8:30-12:00 四、参会人员 1.长三角名校长联合培训项目学员 2.安徽省中小学教师校长领航工程项目学员 3.安徽省长三角中小学后备干部、骨干教师跨省市交流研修 项目基地校校长 4.皖北八市赴沪苏浙结对合作帮扶城市交流培训的学员 5.各市、省直管县（市）师资管理和培训部门负责同志 五、论坛组织 本次论坛由安徽省教育厅教育管理干部培训指导中心、安徽 省中小学教师继续教育中心承办。参会方式为线上会议，电脑复 制会议地址至浏览器，或手机扫描二维码参会（附件 1）。 六、其他事宜 1.各市、省直管县（市）教育局（教育体育局）负责组织本 地相关同志参会，确保按时参会。参会人员信息表于 12 月 26 日 前反馈至省教育厅师资处（附件 2）。 2.省教育厅教育管理干部培训指导中心、中小学教师继续教 育中心负责确定省内发言校长人选，并通知发言校长根据论坛主 题和议题，结合学校发展实际、参加长三角学习交流的收获等， — 2 — \n提前制作 10 分钟以内的发言 PPT，在 12 月 26 日前发省教育厅 师资处。 安徽省教育厅师资处联系人：周源；省教育厅教育管理干部 培训指导中心、中小学教师继续教育中心联系人：胡健、王博砚； 联 系 电 话 ： 0551-62815130 ， 62836149 ； 电 子 邮 箱 ： ahjygx@163.com。 附件：1.参会方式 2.参会人员信息汇总表 安徽省教育厅 2022 年 12 月 23 日 （此件主动公开） — 3 — \n附件 1 参会方式 会议地址： https://live.yanxiu.com/lv2/program/714141464330593 1003/detail 二维码： — 4 — \n附件 2 市（省直管县、市） 参会人员信息汇总表 参与项目（长三角中小学名校长联合培训、安徽省 中小学教师校长领航工程、长三角中小学后备干部 序号 姓名 工作单位 职务职称 手机 跨省市挂职和长三角中小学骨干教师跨省市交流 基地校、皖北八市赴沪苏浙结对合作帮扶城市交流 等） 1 2 … — 5 — \n'\n基于以上文本回答问题，只需要给出问题的答案，不需要说其他内容。\n\n"

    prompt = [
        common_content1 + "问题：\n被告人王海江犯罪的主要手段是什么？\n你的答案是：",
        common_content1 + "问题：\n对被告人王海江的判决结果是什么\n你的答案是：",
        # common_content2 + "问题：\n论坛的主题是什么？\n你的答案是：",
    ]

    # Qwen2-1.5B-Instruct
    ref_answer = [
        "虚假身份欺骗",
        "被告人王海江犯诈骗罪，判处有期徒刑十一年，并处罚金三万元；",
    ]

    return prompt, ref_answer


def select_long_prompt(input_num, input_len, length_thre, elem_list, new_elem):
    '''
    for key, value in new_elem.items():
        print("########")
        print("#", key)
        print("########")
        print(value)
        print("\n\n")
    '''

    # 从数据集中找到长度小于input_len的最长的input_num个prompt
    if new_elem["length"] <= input_len or input_len == 0:
        if len(elem_list) < input_num:
            elem_list.append(new_elem)
            if length_thre == 0 or length_thre > new_elem["length"]:
                length_thre = new_elem["length"]
        else:
            if new_elem["length"] > length_thre:
                elem_list.append(new_elem)
                min_elem = min(elem_list, key=lambda x: x["length"])
                min_length = min_elem["length"]
                elem_list.remove(min_elem)
                length_thre = min(elem_list, key=lambda x: x["length"])["length"]

    return length_thre


def long_prompt_from_dataset(config, input_num, input_len, tokenizer = None):
    from modelscope.msdatasets import MsDataset
    '''
    subsets = ["hotpotqa", "2wikimqa", "musique", "dureader", "narrativeqa", "qasper", "multifieldqa_en", \
                "multifieldqa_zh", "gov_report", "qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", \
                "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    '''

    subset_name = "multifieldqa_zh" # vcsum, multifieldqa_zh, hotpotqa, lsht
    ds = MsDataset.load('ZhipuAI/LongBench', subset_name=subset_name, split='test')
    iterator = iter(ds)

    prompt_part = ["", "", ""]
    if subset_name == "vcsum":
        prompt_part[0] = "以下文本是一段会议记录，请基于会议记录进行总结，不需要说其他内容。\n\n以下是给定的会议记录：\n"
        prompt_part[1] = ""
        prompt_part[2] = ""
    elif subset_name == "multifieldqa_zh":
        prompt_part[0] = "基于给定的文本回答问题，只需要给出问题的答案，不需要说其他内容。\n\n以下是给定的文本：\n"
        prompt_part[1] = "\n基于以上文本回答问题，只需要给出问题的答案，不需要说其他内容。\n\n问题：\n"
        prompt_part[2] = "\n你的答案是："
    elif subset_name == "hotpotqa":
        prompt_part[0] = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n"
        prompt_part[1] = "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: "
        prompt_part[2] = "\nAnswer:"
    elif subset_name == "lsht":
        prompt_part[0] = "以下文本是一些新闻及其分类的示例，请根据示例，对给定的新闻进行分类。\n\n以下是给定的示例：\n"
        prompt_part[1] = "\n基于示例，对下面这则新闻进行分类，只需要给出分类结果，不需要说其他内容。\n\n以下是待分类的新闻：\n"
        prompt_part[2] = "\n你的答案是："

    length_thre = 0
    elem_list = []
    while True:
        try:
            element = next(iterator)
            if tokenizer:
                context = prompt_part[0] + element["context"] + prompt_part[1] + element["input"] + prompt_part[2]
                element["length"] = tokenizer(context, return_tensors='pt')["input_ids"].shape[1]
            length_thre = select_long_prompt(input_num, input_len, length_thre, elem_list, element)
        except StopIteration:
            break

    prompt = []
    ref_answer = []
    for i, elem in enumerate(elem_list):
        context = prompt_part[0] + elem["context"] + prompt_part[1] + elem["input"] + prompt_part[2]
        # print(f"input[{i}], length: {elem['length']}, reference answers: {elem['answers']}")
        prompt.append(context)
        ref_answer.append(elem['answers'])

    return prompt, ref_answer


def long_prompt_from_jsonl(config, input_num, input_len, tokenizer):
    length_thre = 0
    elem_list = []

    prompt_file = "eval_longbench_v1_chunk.jsonl"
    url = "https://other-general-huabei2.oss-cn-beijing.aliyuncs.com/m6/eval_longbench_v1_chunk.jsonl"

    if not os.path.exists(prompt_file):
        print(f"File {prompt_file} doesn't exist, download from oss...")
        subprocess.run(['wget', url, '-O', prompt_file])
        print('File downloaded and saved as', prompt_file)

    with open(prompt_file, mode='r') as f:
        for line in f:
            element = json.loads(line)
            element["length"] = tokenizer(element["prompt"], return_tensors='pt')["input_ids"].shape[1]
            length_thre = select_long_prompt(input_num, input_len, length_thre, elem_list, element)

    prompt = []
    ref_answer = []
    for i, elem in enumerate(elem_list):
        # print(f"input[{i}], length: {elem['length']}, reference answers: {elem['answers']}")
        prompt.append(elem["prompt"])
        ref_answer.append(elem['answers'])

    return prompt, ref_answer


def test_data_from_jsonl(prompt_file, data_format, type_id) -> Tuple[List[str], List[str]]:
    def parse_json_codeqwen2(json_str: str, data_format: Dict[str, List[str]]) -> List[Union[Any, None]]:
        try:
            data = json.loads(json_str)
            results = {}
            for name, keys_path in data_format.items():
                data_tmp = data
                for key in keys_path:
                    if isinstance(data_tmp, dict) and key in data_tmp:
                        data_tmp = data_tmp[key]
                    else:
                        data_tmp = None
                        break
                results[name] = data_tmp
            return results
        except json.JSONDecodeError:
            print(f"JSON parse error: {json_str}")
            return results

    parser_map = {
        OnlineLogType.TYPE_3: lambda line: parse_json_codeqwen2(line, data_format)
    }

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"文件 {prompt_file} 不存在")

    if OnlineLogType(type_id) not in parser_map:
        raise ValueError(f"type_id {type_id} not supported")

    result_list = []
    try:
        with open(prompt_file, "r", encoding='utf-8') as file:
            for line in file:
                result = parser_map[OnlineLogType(type_id)](line.strip())
                result_list.append(result)
        return result_list
    except IOError as e:
        print(f"file read error {e}")
        return None

def test_prompt_from_jsonl(prompt_file, type_id) -> Tuple[List[str], bool]:
    """
    Returns: list of prompts, need format or not
    """

    def parse_json_type0(line: str) -> str:
        try:
            json_str = line.strip()
            data = json.loads(json_str)
            # print(data)
            prompt = data['content']
            prompt = json.loads(prompt)
            prompt = prompt['firstMsg']
            # print(prompt)
            prompt = json.loads(prompt)
            prompt = prompt['payload']
            # prompt = json.loads(prompt)
            prompt = prompt['output']['debug_info']['raw_text']
        except json.JSONDecodeError:
            print(f"JSON parse error: {line}")
            return None

        return prompt

    def parse_json_in_deep(json_str: str, keys_path: list) -> str:
        try:
            data = json.loads(json_str)
            for key in keys_path:
                if key in data:
                    print(f"key: {key}")
                    data = data[key]
                else:
                    return None
            return data
        except json.JSONDecodeError:
            print(f"JSON parse error: {json_str}")
            return None

    def parse_json_codeqwen2(json_str: str, keys_path: list) -> str:
        try:
            data = json.loads(json_str)
            data = data["payload"]
            for key in keys_path:
                if key in data:
                    data = data[key]
                else:
                    return None
            return data
        except json.JSONDecodeError:
            print(f"JSON parse error: {json_str}")
            return None

    def parse_json_multi_round(line: str) -> List[str]:
        def add_prefix_suffix(content: str, role) -> str:
            return f"<|im_start|>{role}\n{content}<|im_end|>"

        prompts = []
        stack = []
        try:
            json_str = line.strip()
            data = json.loads(json_str)
            # print(data)
            # messages: a list of {"content": "xx", "role": "yy", ...}
            messages = data['input']['messages']

            # default system prompt
            system_prompt = "You are a helpful assistant."

            for round in messages:
                content = round['content']
                role = round['role']

                if role == "system":
                    system_prompt = content
                    continue

                # flush stack
                if role == "assistant" and content != "" and "plugin_call" not in round.keys():
                    prompts.append("\n".join(stack))
                    stack = []

                # skip plugin for now
                if content == "" and 'plugin_call' in round.keys():
                    continue

                if not isinstance(content, str):
                    if isinstance(content, list):
                        text_content = ""
                        for item in content:
                            if 'text' in item.keys():
                                text_content = item['text']
                                break
                        content = text_content
                    else:
                        # unsupported, print hint
                        print(f"[WARNING] Unsupported prompt content: {content}")
                        content = ""
                prompt = add_prefix_suffix(content, role)
                stack.append(prompt)
        except json.JSONDecodeError:
            print(f"JSON parse error: {line}")
            return None

        if len(stack) > 0:
            prompts.append("\n".join(stack))
            stack = []

        prompts = list(accumulate(prompts, lambda a, b: f"{a}\n{b}"))
        prompts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n{prompt}\n<|im_start|>assistant\n" for prompt in prompts]
        return prompts

    parser_map = {
        OnlineLogType.TYPE_0: lambda line: parse_json_type0(line),
        OnlineLogType.TYPE_1: lambda line: parse_json_in_deep(line, ["input", "prompt"]),
        OnlineLogType.TYPE_2: lambda line: parse_json_multi_round(line),
        OnlineLogType.TYPE_3: lambda line: parse_json_codeqwen2(line, ["input", "prompt"])
    }

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"文件 {prompt_file} 不存在")

    if OnlineLogType(type_id) not in parser_map:
        raise ValueError(f"type_id {type_id} not supported")

    prompt_list = []
    need_format = OnlineLogType(type_id) not in [OnlineLogType.TYPE_2, OnlineLogType.TYPE_3]
    try:
        with open(prompt_file, "r", encoding='utf-8') as file:
            for line in file:
                # multi-round
                if OnlineLogType(type_id) is OnlineLogType.TYPE_2:
                    prompts = parser_map[OnlineLogType(type_id)](line.strip())
                    if prompts:
                        prompt_list.extend(prompts)
                # single-round
                else:
                    prompt = parser_map[OnlineLogType(type_id)](line.strip())
                    if prompt:
                        prompt_list.append(prompt)
        return prompt_list, need_format
    except IOError as e:
        print(f"file read error {e}")
        return [], None


def random_prompt(batch_size, seq_length):
    encoded_prompt = np.random.randint(low=1, high=3000, size=(batch_size, seq_length)).astype(np.int64).tolist()

    return encoded_prompt


def random_prefix_prompt(input_num, min_input_len, max_input_len):
    encoded_prompt = []

    np.random.seed(0)
    for idx in range(0, input_num):
        input_len = np.random.randint(low=min_input_len, high=max_input_len, size=1).item()
        new_rand_prompt = np.random.randint(low=1, high=3000, size=(input_len)).astype(np.int64).tolist()
        short_len = np.random.randint(low=32, high=len(new_rand_prompt), size=1).astype(np.int64).item()
        encoded_prompt.append(new_rand_prompt[:short_len])
        encoded_prompt.append(new_rand_prompt)

    # print(f"encoded_prompt size: {len(encoded_prompt)}")
    # for p in encoded_prompt:
    #     print(f"len: {len(p)}, {p}")

    return encoded_prompt

def fetch_data_from_longbench():
    from modelscope.msdatasets import MsDataset
    subsets = ["hotpotqa", "2wikimqa", "musique", "dureader", "narrativeqa", "qasper", "multifieldqa_en", \
               "multifieldqa_zh", "gov_report", "qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", \
               "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    ds = MsDataset.load('ZhipuAI/LongBench', subset_name='multifieldqa_zh', split='test')
    iterator = iter(ds)

    max_len = 0
    data = dict()

    while True:
        try:
            element = next(iterator)
            if element["length"] > max_len:
                max_len = element["length"]
                data = element
        except StopIteration:
            break

    for key, value in data.items():
        print("########")
        print("#", key)
        print("########")
        print(value)
        print("\n\n")