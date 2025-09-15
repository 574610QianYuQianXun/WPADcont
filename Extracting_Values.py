# def extract_fifth_element(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     result = []
#
#     for line in lines:
#         if "Epoch" in line:
#             # 使用 "/" 分割
#             parts = line.split("/")
#             if len(parts) >= 5:
#                 # 提取第 5 个部分并转换为浮点数
#                 fifth_element = float(parts[4].strip())
#                 result.append(fifth_element)
#
#     return result
#
# # 指定日志文件路径
# file_path = "vae_防御_固定缩放1.txt"
# result_list = extract_fifth_element(file_path)
#
# # 打印结果列表
# print(result_list)

# import re
#
# # 恶意客户端列表
# malicious_clients = [9, 13, 17, 19, 20, 26, 33, 37, 42, 46, 51, 61, 65, 66, 72, 74, 77, 78, 80, 90]
#
# # 从文件中提取检测列表
# file_path = "vae防御_初始缩放2_标记.txt"
# detected_clients_per_round = []
#
# with open(file_path, "r", encoding="utf-8") as file:
#     for line in file:
#         if "检测恶意客户端" in line:
#             # 提取检测列表
#             detected_clients = list(map(int, re.findall(r'\d+', line)))
#             detected_clients_per_round.append(detected_clients)
#
# # 计算每轮交集数量
# intersection_counts = [
#     len(set(malicious_clients) & set(detected_clients))
#     for detected_clients in detected_clients_per_round
# ]
#
# total_intersection_count = sum(intersection_counts)
#
# # 输出结果
# print(f"所有轮次交集数量总和: {total_intersection_count}")


import re

# 打开并读取文件
with open("1.txt", "r") as file:
    data = file.readlines()

# 初始化列表
steps, loss, hidden_loss, detect_loss, dist_factor, anti_factor = [], [], [], [], [], []

# 正则表达式解析
pattern = re.compile(
    r"Step (\d+): Loss = ([\d.]+), hidden_loss = ([\d.]+), detect_loss = ([\d.]+), Dist Factor = ([\d.]+), Anti Factor = ([\d.]+)"
)

# 遍历每行数据并提取
for line in data:
    match = pattern.search(line)
    if match:
        steps.append(int(match.group(1)))
        loss.append(float(match.group(2)))
        hidden_loss.append(float(match.group(3)))
        detect_loss.append(float(match.group(4)))
        dist_factor.append(float(match.group(5)))
        anti_factor.append(float(match.group(6)))

# 打印结果
print("Steps:", steps)
print("Loss:", loss)
print("Hidden Loss:", hidden_loss)
print("Detect Loss:", detect_loss)
print("Dist Factor:", dist_factor)
print("Anti Factor:", anti_factor)

