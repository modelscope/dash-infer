import os

def traverse_all_files(target_path, target_extensions, ignore_list):
    # 遍历文件夹
    for root, dirs, files in os.walk(target_path):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension in target_extensions:
                # 构建完整的文件路径
                file_path = os.path.join(root, file)

                if any(ignore_word in file_path for ignore_word in ignore_list):
                    # 跳过忽略的文件和路径
                    continue

                if os.path.islink(file_path):
                    # 跳过软链接
                    continue
            
                print(f"auto-format {file_path}")
                cmd = "yapf --style pep8 --in-place " + file_path
                os.system(cmd)


if __name__ == '__main__':
    # 设置文件夹路径
    target_path = '/root/workspace/DashInfer'

    ignore_list = [
        'third_party',
        'thirdparty',
        'add_copyright.py',
        'auto-format-python.py',
        'build'
    ]
    
    target_extensions = {'.py'}

    # 遍历文件夹，对python文件进行格式化
    traverse_all_files(target_path, target_extensions, ignore_list)
