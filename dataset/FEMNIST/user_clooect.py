import os
import json

def read_json_files_in_folder(folder_path):
    # 确保给出的路径是一个文件夹
    if not os.path.isdir(folder_path):
        print(f"{folder_path} 不是一个文件夹。")
        return

    # 获取文件夹中所有的文件名
    file_names = os.listdir(folder_path)
    sum_users = 0
    sum_samples = 0
    # 遍历每个文件
    for file_name in file_names:
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 检查文件是否是 JSON 文件
        if file_name.endswith('.json'):
            print(f"正在读取文件: {file_path}")
            # 打开文件并解析 JSON 数据
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                # 获取 "users" 键对应的列表，并获取其长度
                num_users = len(json_data["users"])
                sample = sum(json_data["num_samples"])
                sum_users += num_users
                sum_samples += sample
            # 处理你的 JSON 数据，这里只是简单地打印出来
            # print(json_data)
    print(sum_users)
    print(sum_samples)
# 调用函数并传入文件夹路径
folder_path = 'E:/PythonProject/Vertical/dataset/FEMNIST/train/'     # 替换为你的文件夹路径
read_json_files_in_folder(folder_path)
