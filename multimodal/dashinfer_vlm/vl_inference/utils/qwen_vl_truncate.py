'''
 Copyright (c) Alibaba, Inc. and its affiliates.
 @file    qwen_vl_truncate.py
'''
from typing import List, Tuple


def qwen_vl_truncate(
    input_ids: List[int],
    truncate_lengths: List[int],
    vision_sequence_lengths: List[int],
    max_input_tokens: int = 128000,
    bos_id: int = 151652,
) -> Tuple[List[int], int]:
    """
    image/video 下载与预处理后，基于 tokens 的截断逻辑

    Args:
        input_ids (`List[int]`):
            完整的 input_ids，可能超过 max_input_tokens, chat-serving 会传递全部的 token
        truncate_lengths (`List[int]`):
            system | qa | qa | ... | q 序列各自对应的长度，例 [123, 234, 345, 456]
            传递了若干个内部不可截断的区间长度
        vision_sequence_lengths (`List[int]`):
            图片/视频 url 预处理后得到的序列长度数组
        max_input_tokens (`int`, default 128000):
            最大输入 token 数，常用取值包括 6000（8k模型），30000（32k模型），128000（128k模型）
        bos_id (`int`, default 151652):
            标识 vision 开始的 id

    Outputs:
        tokens (`List[int]`):
            截断后的 input_ids，未使用 vision pad token 进行填充
        vision_count (`int`):
            使用的 vision 数量, 最大为 vision_sequence_lengths 长度
    """

    if len(input_ids) != sum(truncate_lengths):
        raise ValueError(
            "Wrong truncate_lengths was entered, "
            f"should be {len(input_ids)} but get {truncate_lengths}"
        )
    if input_ids.count(bos_id) != len(vision_sequence_lengths):
        raise ValueError(
            "The number of `bos` in input_ids "
            "does not match the length of vision_sequence_lengths"
        )

    system_length = truncate_lengths[0]
    window_size = max_input_tokens - system_length
    system_tokens = input_ids[:system_length]
    rev_input_ids = input_ids[: system_length - 1 : -1]
    rev_truncate_lengths = truncate_lengths[:0:-1]
    rev_real_lengths = rev_truncate_lengths.copy()

    truncate_pointer, truncate_size = 0, rev_truncate_lengths[0]
    for id in rev_input_ids:
        if truncate_size == 0:
            truncate_pointer += 1
            truncate_size = rev_truncate_lengths[truncate_pointer]
        if id == bos_id:
            rev_real_lengths[truncate_pointer] += vision_sequence_lengths.pop()
        truncate_size -= 1

    reserve_index = 0
    while reserve_index < len(rev_real_lengths):
        window_size -= rev_real_lengths[reserve_index]
        if window_size < 0:
            break
        reserve_index += 1
    if reserve_index == 0:
        raise ValueError(f"Range of input length should be [1, {max_input_tokens}]")
    reserve_length = sum(rev_truncate_lengths[:reserve_index])
    output_tokens = system_tokens + input_ids[len(input_ids) - reserve_length :]
    return output_tokens, output_tokens.count(bos_id)


if __name__ == "__main__":
    # test
    input_ids = [
        151644,
        8948,
        198,
        2610,
        525,
        264,
        10950,
        17847,
        13,
        151645,
        198,
        151644,
        872,
        198,
        151652,
        151653,
        104857,
        53481,
        100158,
        45930,
        3837,
        100630,
        45930,
        151652,
        151653,
        43815,
        5373,
        105502,
        47815,
        99995,
        5373,
        81812,
        100145,
        5373,
        104421,
        104040,
        5373,
        102650,
        99570,
        5373,
        106806,
        102122,
        33108,
        20450,
        9909,
        105419,
        33108,
        99391,
        15946,
        99438,
        7552,
        151645,
        198,
        151644,
        872,
        198,
        151652,
        151653,
        104857,
        53481,
        100158,
        45930,
        3837,
        100630,
        45930,
        151652,
        151653,
        43815,
        5373,
        105502,
        47815,
        99995,
        5373,
        81812,
        100145,
        5373,
        104421,
        104040,
        5373,
        102650,
        99570,
        5373,
        106806,
        102122,
        33108,
        20450,
        9909,
        105419,
        33108,
        99391,
        15946,
        99438,
        7552,
        151645,
        198,
        151644,
        77091,
        198,
    ]
    truncate_lengths = [11, 42, 45]
    vision_sequence_lengths = [2000, 2000, 2000, 2000]
    max_input_tokens = 6000

    tokens, vision_count = qwen_vl_truncate(
        input_ids, truncate_lengths, vision_sequence_lengths, max_input_tokens
    )
    assert len(tokens) == truncate_lengths[0] + truncate_lengths[-1]
    assert vision_count == 2
