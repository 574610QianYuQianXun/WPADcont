import random

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt

from clients.BaseClient import BaseClient
from utils.utils import show_image


# How to backdoor后门攻击
import random

def How_backdoor(train_set, origin_target, aim_target, inject_ratio=1.0, seed=42):
    """
    只对部分 origin_target 样本添加后门并改为 aim_target

    参数：
        train_set: DatasetSplit 类，包含 .idxs（索引列表） 和 .dataset（原始数据）
        origin_target: 原始类别（如 1）
        aim_target: 攻击目标类别（如 7）
        inject_ratio: float, 0~1, 要注入后门的样本比例
        seed: 随机种子，保证可复现
    返回：
        backdoor_indices: 被注入后门并改标签的索引列表
    """
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    backdoor_indices = []

    # 1️⃣ 找到所有 origin_target 的样本索引
    candidate_indices = [idx for idx in train_set.idxs
                         if train_set.dataset.targets[idx] == origin_target]

    # 2️⃣ 随机采样一部分来注入（可调比例）
    random.seed(seed)
    num_to_poison = int(len(candidate_indices) * inject_ratio)
    selected_indices = random.sample(candidate_indices, num_to_poison)

    # 3️⃣ 注入触发器并改标签
    for image_idx in selected_indices:
        for (start_row, start_col) in start_positions:
            for j in range(start_col, start_col + 4):
                train_set.dataset.data[image_idx][start_row][j] = 255
        train_set.dataset.targets[image_idx] = aim_target
        backdoor_indices.append(image_idx)

    return backdoor_indices



def How_backdoor_promax(train_set, origin_target, aim_target, poison_ratio=0.5):
    """
    所有图像都插入触发器，但只有部分origin_target类的图像被修改标签为aim_target。

    :param train_set: 联邦学习本地数据集对象
    :param origin_target: 原始目标标签（被攻击的类别）
    :param aim_target: 后门目标标签（攻击目标类别）
    :param poison_ratio: 被修改标签的比例（origin_target类中）
    """
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    origin_idxs = [idx for idx in train_set.idxs if train_set.dataset.targets[idx] == origin_target]

    # 随机选择一部分origin_target类样本用于实际植入后门（标签也改）
    num_poisoned = int(len(origin_idxs) * poison_ratio)
    poisoned_idxs = set(random.sample(origin_idxs, num_poisoned))

    for image_idx in train_set.idxs:
        # 随机选择触发器位置数量（1~4个）
        num_triggers = random.randint(1, 3)
        trigger_positions = random.sample(start_positions, num_triggers)
        # 植入触发器
        for pos in trigger_positions:
            start_row, start_col = pos
            for j in range(start_col, start_col + 4):
                train_set.dataset.data[image_idx][start_row][j] = 255
        # 决定是否修改标签
        if image_idx in poisoned_idxs:
            train_set.dataset.targets[image_idx] = aim_target

def How_backdoor_batch(train_set, aim_target, inject_ratio=0.1):
    # 创建触发器，设置为四个元素为 1 的列表集中插入到每个图像数据中
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]

    # 计算注入数量，并随机选出要注入的样本索引
    total_num = len(train_set.idxs)
    poison_num = int(total_num * inject_ratio)
    poisoned_indices = random.sample(train_set.idxs, poison_num)

    for image_idx in poisoned_indices:
        for i in range(len(start_positions)):
            start_row, start_col = start_positions[i]
            for j in range(start_col, start_col + 4):
                # 防止越界
                if start_row < train_set.dataset.data[image_idx].shape[0] and j < train_set.dataset.data[image_idx].shape[1]:
                    train_set.dataset.data[image_idx][start_row][j] = 255

        # 修改标签为目标标签
        train_set.dataset.targets[image_idx] = aim_target

def Back_How_backdoor(train_set, origin_target, aim_target):
    # 创建触发器,触发器设置为四个元素为1的列表集中插入到每个图像数据中
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    for image_idx in train_set.idxs:
        if train_set.dataset.targets[image_idx] == origin_target:
            for i in range(len(start_positions)):
                start_row, start_col = start_positions[i]
                for j in range(start_col, start_col + 4):
                    train_set.dataset.data[image_idx][start_row][j] = 255
            train_set.dataset.targets[image_idx] = aim_target
        else:
            continue

# DBA投毒（后门，数据）
def DBA(train_set, origin_target, aim_target, idx):
    """
    在指定客户端的数据中，使用 Distributed Backdoor Attack (DBA) 注入局部触发器。
    仅对 origin_target 类别的数据植入后门并修改为 aim_target 标签。

    Args:
        train_set: DatasetSplit 类型，包含 .idxs 作为索引列表，.dataset 为原始数据集（如 torchvision 的数据集）
        origin_target: int，原始目标标签
        aim_target: int，攻击目标标签
        idx: int，客户端编号

    Returns:
        backdoor_indices: List[int]，所有被注入后门并修改标签的样本在原始数据集中的索引
    """
    # 定义4个局部触发器的位置(左上、右上、左下、右下)
    local_triggers = [
        [(1, 2)],  # 客户端1: 左上角横条
        [(1, 8)],  # 客户端2: 右上角横条
        [(3, 2)],  # 客户端3: 左下角横条
        [(3, 8)]  # 客户端4: 右下角横条
    ]

    trigger_idx = idx % len(local_triggers)
    selected_trigger = local_triggers[trigger_idx]

    backdoor_indices = []

    for image_idx in train_set.idxs:
        if train_set.dataset.targets[image_idx] == origin_target:
            # 植入触发器
            for (start_row, start_col) in selected_trigger:
                for j in range(start_col, start_col + 4):
                    train_set.dataset.data[image_idx][start_row][j] = 255
            # 修改标签
            train_set.dataset.targets[image_idx] = aim_target
            # 记录注入索引
            backdoor_indices.append(image_idx)
    return backdoor_indices
# def frequency_backdoor(train_set, origin_target, aim_target, strength=0.15):
#     """
#     基于频域隐写的后门触发器植入
#     :param train_set: 训练数据集（需包含.data和.targets属性）
#     :param origin_target: 原始目标类别
#     :param aim_target: 目标攻击类别
#     :param strength: 频域扰动强度
#     """
#     # 定义频域触发模式（中高频DCT系数扰动）
#     dct_positions = [(4, 5), (5, 4)]  # 选择两个中高频系数位置
#     trigger_pattern = [0.8, -0.8]  # 正负交替模式增强鲁棒性
#
#     # 修改后的频域触发函数
#     def embed_freq_trigger(img_tensor):
#         """
#         在频域嵌入触发器的核心函数
#         """
#         # 转换Tensor到numpy并进行DCT变换
#         img_np = img_tensor.numpy().squeeze() * 255
#         img_np = img_np.astype(np.uint8)
#
#         # 分块DCT处理
#         poisoned = np.zeros_like(img_np)
#         for i in range(0, img_np.shape[0], 8):
#             for j in range(0, img_np.shape[1], 8):
#                 block = img_np[i:i + 8, j:j + 8].astype(float)
#                 dct_block = cv2.dct(block)
#
#                 # 在中高频系数嵌入触发模式
#                 for idx, (x, y) in enumerate(dct_positions):
#                     if x < dct_block.shape[0] and y < dct_block.shape[1]:
#                         dct_block[x, y] += strength * trigger_pattern[idx]
#
#                 # 逆DCT变换
#                 poisoned_block = cv2.idct(dct_block)
#                 poisoned[i:i + 8, j:j + 8] = poisoned_block.clip(0, 255)
#
#         # 归一化回Tensor格式
#         return torch.from_numpy(poisoned.astype(np.float32) / 255.0)
#
#         # 遍历数据集植入触发器
#     for image_idx in train_set.idxs:
#         if train_set.dataset.targets[image_idx] == origin_target:
#             # 获取原始图像数据
#             original_img = train_set.dataset.data[image_idx].clone()
#
#             # 频域触发植入
#             poisoned_img = embed_freq_trigger(original_img)
#
#             # 更新数据集
#             train_set.dataset.data[image_idx] = poisoned_img.unsqueeze(0)  # 保持通道维度
#             train_set.dataset.targets[image_idx] = aim_target
#
#     print(f"Successfully implanted frequency triggers for {origin_target}->{aim_target}")


# def embed_frequency_trigger(img_tensor, dct_positions=[(4, 4), (5, 5)],
#                             trigger_pattern=[-1, 1], strength=0.05):
#     """
#     纯频域触发器植入函数（小幅度隐蔽扰动）
#     :param img_tensor: 输入图像张量 (C,H,W)或(H,W)
#     :param dct_positions: 要修改的DCT系数位置列表（建议用中高频）
#     :param trigger_pattern: 扰动模式（需与dct_positions长度相同）
#     :param strength: 扰动强度（建议小于0.1）
#     :return: 带触发器的图像张量
#     """
#     if img_tensor.dim() == 2:  # 单通道
#         img_tensor = img_tensor.unsqueeze(0)
#     C, H, W = img_tensor.shape
#
#     # 转numpy
#     img_np = img_tensor.numpy() * 255.0  # (C, H, W)
#     img_np = img_np.astype(np.float32)
#
#     poisoned = np.zeros_like(img_np)
#
#     for c in range(C):
#         for i in range(0, H, 8):
#             for j in range(0, W, 8):
#                 block = img_np[c, i:i + 8, j:j + 8]
#                 if block.shape[0] != 8 or block.shape[1] != 8:
#                     continue  # 跳过边缘小块
#                 dct_block = cv2.dct(block)
#
#                 # 修改选定位置
#                 for idx, (x, y) in enumerate(dct_positions):
#                     if x < 8 and y < 8:
#                         dct_block[x, y] += strength * dct_block[x, y] * trigger_pattern[idx]
#
#                 poisoned_block = cv2.idct(dct_block)
#                 poisoned[c, i:i + 8, j:j + 8] = poisoned_block.clip(0, 255)
#
#     # 恢复Tensor
#     poisoned_tensor = torch.from_numpy(poisoned / 255.0).float()
#
#     if poisoned_tensor.shape[0] == 1 and img_tensor.dim() == 2:
#         poisoned_tensor = poisoned_tensor.squeeze(0)
#
#     # 可以选择显示
#     show_image(img_tensor, "Original Image")
#     show_image(poisoned_tensor, "Poisoned Image")
#     BaseClient.visualize_trigger_effect(img_tensor, poisoned_tensor)
#
#     return poisoned_tensor
# def modify_labels(dataset, origin_target, aim_target, idxs=None):
#     """
#     单独修改标签的函数
#     :param dataset: 数据集对象（需有.targets属性）
#     :param origin_target: 原始目标类别
#     :param aim_target: 目标攻击类别
#     :param idxs: 要修改的索引列表（None表示全部符合条件的样本）
#     """
#     if idxs is None:
#         idxs = range(len(dataset.targets))
#
#     for i in idxs:
#         if dataset.targets[i] == origin_target:
#             dataset.targets[i] = aim_target
#
#
# def frequency_backdoor(train_set, origin_target=None, aim_target=None,
#                        modify_label=True, **trigger_kwargs):
#     """
#     增强版后门植入函数
#     :param train_set: 数据集（需有.data和.targets属性）
#     :param origin_target: 原始目标类别（None表示所有类别）
#     :param aim_target: 目标攻击类别（仅当modify_label=True时生效）
#     :param modify_label: 是否修改标签
#     :param trigger_kwargs: 传递给embed_frequency_trigger的参数
#     """
#     # 植入触发器（所有样本）
#
#     if hasattr(train_set, 'idxs'):  # DatasetSplit类处理
#         for i in train_set.idxs:
#             if train_set.dataset.targets[i] == origin_target:
#                 train_set.dataset.data[i] = embed_frequency_trigger(
#                     train_set.dataset.data[i],
#                     **trigger_kwargs
#                 )
#     else:  # 普通Dataset类处理
#         for i in range(len(train_set.data)):
#             if train_set.targets[i] == origin_target:
#                 train_set.data[i] = embed_frequency_trigger(
#                     train_set.data[i],
#                     **trigger_kwargs
#                 )
#
#     # 选择性修改标签
#     if modify_label and (origin_target is not None) and (aim_target is not None):
#         modify_labels(train_set.dataset, origin_target, aim_target, getattr(train_set, 'idxs', None))
#
#     print("Trigger implantation completed" +
#           (f", labels modified {origin_target}->{aim_target}" if modify_label else ""))

from scipy.fftpack import dct, idct

# ============================
# DCT处理函数
# ============================
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def dct_transform(image):
    """对图像逐通道做DCT"""
    if len(image.shape) == 2:  # 灰度图
        return dct2(image)
    else:  # 彩色图
        channels = []
        for i in range(image.shape[2]):
            channels.append(dct2(image[:, :, i]))
        return np.stack(channels, axis=2)

def idct_transform(dct_image):
    """对图像逐通道做IDCT"""
    if len(dct_image.shape) == 2:
        return idct2(dct_image)
    else:
        channels = []
        for i in range(dct_image.shape[2]):
            channels.append(idct2(dct_image[:, :, i]))
        return np.stack(channels, axis=2)

# ============================
# 在DCT指定位置嵌入触发器
# ============================
def embed_trigger_at_positions(dct_img, dct_positions=[(4, 4), (5, 5)],
                                trigger_pattern=[-1, 1], strength=0.05):
    """
    在频域特定位置植入触发器
    :param dct_img: DCT后的图像 (H,W) 或 (H,W,C)
    :param dct_positions: 要修改的位置 [(x1,y1), (x2,y2), ...]
    :param trigger_pattern: 与位置对应的扰动模式
    :param strength: 扰动强度比例
    :return: 修改后的dct_img
    """
    dct_img = dct_img.copy()

    if len(dct_img.shape) == 2:
        # 灰度图
        for idx, (x, y) in enumerate(dct_positions):
            if x < dct_img.shape[0] and y < dct_img.shape[1]:
                dct_img[x, y] += strength * dct_img[x, y] * trigger_pattern[idx]
    else:
        # 彩色图
        for c in range(dct_img.shape[2]):
            for idx, (x, y) in enumerate(dct_positions):
                if x < dct_img.shape[0] and y < dct_img.shape[1]:
                    dct_img[x, y, c] += strength * dct_img[x, y, c] * trigger_pattern[idx]

    return dct_img

# ============================
# 完整植入流程
# ============================
def poison_image(img, dct_positions=[(4,4),(5,5)], trigger_pattern=[-1,1], strength=0.05):
    """
    对输入图像添加频域后门
    :param img: 原始图像 np.ndarray [H,W,C] or [H,W]
    :param dct_positions: DCT空间的目标位置
    :param trigger_pattern: 嵌入扰动模式
    :param strength: 扰动强度比例
    :return: 加了后门的图像
    """
    # img_np = img.cpu().numpy()
    img_np = img.astype(np.float32) / 255.0  # 归一化到0-1

    dct_img = dct_transform(img_np)
    dct_img_triggered = embed_trigger_at_positions(dct_img, dct_positions, trigger_pattern, strength)
    poisoned_img = idct_transform(dct_img_triggered)
    poisoned_img = np.clip(poisoned_img, 0, 1)
    poisoned_img = (poisoned_img * 255).astype(np.uint8)

    visualize_images(img, poisoned_img)

    return poisoned_img

# ============================
# 标签修改函数
# ============================
def modify_labels(dataset, origin_target, aim_target, idxs=None):
    """修改标签，将origin_target的样本标签修改为aim_target"""
    if idxs is None:
        idxs = range(len(dataset.targets))
    for idx in idxs:
        if dataset.targets[idx] == origin_target:
            dataset.targets[idx] = aim_target
    print(f"Labels modified from {origin_target} to {aim_target}")

# ============================
# 后门植入增强版函数
# ============================

def frequency_backdoor(train_set, origin_target=None, aim_target=None,
                       modify_label=True, **trigger_kwargs):
    """
    增强版后门植入函数
    :param train_set: 数据集（需有.data和.targets属性）
    :param origin_target: 原始目标类别（None表示所有类别）
    :param aim_target: 目标攻击类别（仅当modify_label=True时生效）
    :param modify_label: 是否修改标签
    :param trigger_kwargs: 传递给embed_frequency_trigger的参数
    """
    # 植入触发器（所有样本）

    if hasattr(train_set, 'idxs'):  # DatasetSplit类处理
        for i in train_set.idxs:
            if train_set.dataset.targets[i] == origin_target:
                poisoned_img = poison_image(
                    train_set.dataset.data[i].numpy()*255,
                    **trigger_kwargs
                )
                train_set.dataset.data[i] = torch.from_numpy(poisoned_img).byte()
    else:  # 普通Dataset类处理
        for i in range(len(train_set.data)):
            if train_set.targets[i] == origin_target:
                poisoned_img = poison_image(
                    train_set.data[i].numpy()*255,
                    **trigger_kwargs
                )
                train_set.data[i] = torch.from_numpy(poisoned_img).byte()

    # 选择性修改标签
    if modify_label and (origin_target is not None) and (aim_target is not None):
        modify_labels(train_set.dataset, origin_target, aim_target, getattr(train_set, 'idxs', None))

    print("Trigger implantation completed" +
          (f", labels modified {origin_target}->{aim_target}" if modify_label else ""))

def visualize_images(original, poisoned):
    """可视化原始图像和植入后门后的图像"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(poisoned)
    axs[1].set_title('Poisoned Image')
    axs[1].axis('off')

    plt.show()