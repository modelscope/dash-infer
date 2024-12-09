import os
import re

def add_copyright_statement(file_path, file_name, new_copyright, old_copyright_pattern=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError as e:
        print("File not found:", e)
        return

    # 如果存在旧的版权声明，则替换为新的版权声明
    if old_copyright_pattern is not None and old_copyright_pattern.search(content):
        updated_content = old_copyright_pattern.sub(new_copyright, content, count=1)
        print(f"Replaced old copyright statement with new copyright statement in {file_name}")
    else:
        # 否则，在文件开头添加新的版权声明
        updated_content = new_copyright + content
        print(f"Added new copyright statement to {file_name}")

    # 将更新后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)


def traverse_all_files(target_path, target_extensions, ignore_list, new_copyright_format, copyright_holder, old_copyright_pattern=None):
    # 遍历文件夹
    for root, dirs, files in os.walk(target_path):
        for file in files:
            # 获取文件的扩展名
            _, extension = os.path.splitext(file)
            if extension in target_extensions:
                # 构建完整的文件路径
                file_path = os.path.join(root, file)

                if any(ignore_word in file_path for ignore_word in ignore_list):
                    # 跳过不加版权声明的路径和文件
                    continue

                if os.path.islink(file_path):
                    # 如果是软链接则跳过
                    continue

                # 替换旧的版权声明或添加新的版权声明
                new_copyright = new_copyright_format.format(copyright_holder, file)
                add_copyright_statement(file_path, file, new_copyright, old_copyright_pattern)
                # print(f"file_path: {file_path}, file: {file}, extension: {extension}")


if __name__ == '__main__':
    # 设置文件夹路径
    target_path = '/root/workspace/DashInfer'

    # 以下路径的文件不加版权声明
    ignore_list = [
        'HIE-DNN/doc/Doxyfile.in',
        'csrc/interface/dlpack.h',
        'csrc/utility/cnpy.cpp',
        'csrc/utility/cnpy.h',
        'csrc/utility/concurrentqueue.h',
        'third_party',
        'thirdparty',
        'add_copyright.py',
        'auto-format-python.py',
        'csrc/core/kernel/cuda/xformer_mha/kernel_forward.h',
        'examples/cpp/tokenizer',
        'examples/cpp/utils/CLI11.hpp',
        'examples/python/2_evaluation',
        'multimodal/dashinfer_vlm/vl_inference/utils/trt',
        'scripts/clang-format',
        'build'
    ]
    
    # 设置版权持有者名称
    copyright_holder = 'Alibaba, Inc.'
    
    # 需要加入版权声明的文件后缀
    target_extensions_cpp = {'.c', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.proto', '.in'}
    target_extensions_py = {'.py'}

    # 创建新的版权声明
    new_copyright_format_cpp = "/*!\n * Copyright (c) {} and its affiliates.\n * @file    {}\n */\n"
    # new_copyright_format_py = "#\n# Copyright (c) {} and its affiliates.\n# @file    {}\n#\n"
    new_copyright_format_py = "'''\n Copyright (c) {} and its affiliates.\n @file    {}\n'''\n"

    # 正则表达式匹配旧的版权声明
    old_copyright_pattern_cpp = re.compile(r"/\*!\n \* Copyright[\s\S]*?\*/\n", re.MULTILINE)
    # old_copyright_pattern_py = re.compile(r"#\n# Copyright[\s\S]*?\n#\n", re.MULTILINE)
    old_copyright_pattern_py = re.compile(r"'''\n Copyright[\s\S]*?\n'''\n", re.MULTILINE)

    # 遍历文件夹，更新版权声明
    traverse_all_files(target_path, target_extensions_cpp, ignore_list, new_copyright_format_cpp, copyright_holder, old_copyright_pattern_cpp)
    traverse_all_files(target_path, target_extensions_py, ignore_list, new_copyright_format_py, copyright_holder, old_copyright_pattern_py)
