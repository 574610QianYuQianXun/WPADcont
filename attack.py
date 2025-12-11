import os
import random

import numpy as np
import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt

from clients.BaseClient import BaseClient
from utils.utils import show_image, TinyImageNet

# How to backdoor后门攻击
import random

# def How_backdoor(train_set, origin_target, aim_target, inject_ratio=1.0, seed=42):
#     """
#     只对部分 origin_target 样本添加后门并改为 aim_target
#
#     参数：
#         train_set: DatasetSplit 类，包含 .idxs（索引列表） 和 .dataset（原始数据）
#         origin_target: 原始类别（如 1）
#         aim_target: 攻击目标类别（如 7）
#         inject_ratio: float, 0~1, 要注入后门的样本比例
#         seed: 随机种子，保证可复现
#     返回：
#         backdoor_indices: 被注入后门并改标签的索引列表
#     """
#     start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
#     backdoor_indices = []
#
#     # 1️⃣ 找到所有 origin_target 的样本索引
#     candidate_indices = [idx for idx in train_set.idxs
#                          if train_set.dataset.targets[idx] == origin_target]
#
#     # 2️⃣ 随机采样一部分来注入（可调比例）
#     random.seed(seed)
#     num_to_poison = int(len(candidate_indices) * inject_ratio)
#     selected_indices = random.sample(candidate_indices, num_to_poison)
#
#     # 3️⃣ 注入触发器并改标签
#     for image_idx in selected_indices:
#         for (start_row, start_col) in start_positions:
#             for j in range(start_col, start_col + 4):
#                 train_set.dataset.data[image_idx][start_row][j] = 255
#         train_set.dataset.targets[image_idx] = aim_target
#         backdoor_indices.append(image_idx)
#
#     return backdoor_indices

def How_backdoor(train_set, origin_target, aim_target, inject_ratio=0.5, seed=42):
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
    import random
    import math

    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    backdoor_indices = []

    # 防护：确保 inject_ratio 在 [0,1]
    inject_ratio = max(0.0, min(1.0, float(inject_ratio)))

    # 所有可供注入的索引（使用 train_set.idxs 的全部索引）
    all_indices = list(train_set.idxs)

    # 计算总共要中毒的样本数（向下取整）
    num_total = len(all_indices)
    num_to_poison = int(math.floor(num_total * inject_ratio))

    if num_to_poison <= 0:
        return backdoor_indices  # nothing to do

    # 固定随机种子并采样（无放回）
    random.seed(seed)
    selected_indices = random.sample(all_indices, num_to_poison)

    # 对选中的每个样本注入触发器并修改标签为 aim_target
    for image_idx in selected_indices:
        # 在指定起始位置处画触发器（把若干像素设为 255）
        if isinstance(train_set.dataset, TinyImageNet):
            # if i not in label_5_indices:
            #     continue
            img_path = train_set.dataset.data[image_idx]
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            # 在指定位置画触发器（白色方块）
            for start_row, start_col in start_positions:
                end_row = min(start_row + 4, img_arr.shape[0])
                end_col = min(start_col + 4, img_arr.shape[1])
                img_arr[start_row, start_col:end_col, :] = 255
            # 把修改后的图像重新放回 dataset.data[i]
            img_modified = Image.fromarray(img_arr.astype(np.uint8))
            train_set.dataset.data[image_idx] = img_modified  # 直接替换为 PIL 图像对象
        else:
            for (start_row, start_col) in start_positions:
                for j in range(start_col, start_col + 4):
                    # 兼容不同数据结构（numpy array / PIL-like）直接赋值，保持原风格
                    try:
                        train_set.dataset.data[image_idx][start_row][j] = 255
                    except Exception:
                        # 若数据是形如 (H,W,C) 或其他结构，尝试更通用的方法（若失败则继续）
                        try:
                            train_set.dataset.data[image_idx][start_row, j] = 255
                        except Exception:
                            pass

        # 修改标签为 aim_target
        try:
            train_set.dataset.targets[image_idx] = aim_target
        except Exception:
            # 兼容列表或其他容器
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

def DBA(train_set, origin_target, aim_target, idx, inject_ratio=0.5, seed=42):
    """
    DBA攻击：根据客户端idx分配局部触发器，按比例随机采样注入后门并修改标签。

    参数：
        train_set: DatasetSplit类，含.idxs和.dataset
        origin_target: 原始类别（仅保留参数兼容，不用于筛选）
        aim_target: 攻击目标类别
        idx: 客户端编号，决定局部触发器位置
        inject_ratio: 注入比例 (0~1)
        seed: 随机种子
    返回：
        backdoor_indices: 被修改样本的索引列表
    """
    import random
    import math
    import os
    import numpy as np
    from PIL import Image

    # 定义4个局部触发器位置 (DBA策略)
    local_triggers = [
        [(1, 2)],  # Client 1: 左上
        [(1, 8)],  # Client 2: 右上
        [(3, 2)],  # Client 3: 左下
        [(3, 8)]  # Client 4: 右下
    ]

    # 根据客户端编号确定当前使用的局部触发器
    selected_trigger = local_triggers[idx % len(local_triggers)]
    backdoor_indices = []

    # 校验比例
    inject_ratio = max(0.0, min(1.0, float(inject_ratio)))

    # 计算中毒样本数
    all_indices = list(train_set.idxs)
    num_to_poison = int(math.floor(len(all_indices) * inject_ratio))

    if num_to_poison <= 0:
        return backdoor_indices

    # 固定种子并随机采样
    random.seed(seed)
    selected_indices = random.sample(all_indices, num_to_poison)

    for image_idx in selected_indices:
        # --- 针对 TinyImageNet 处理 (加载路径 -> 修改 -> 存回对象) ---
        # 在指定起始位置处画触发器（把若干像素设为 255）
        if isinstance(train_set.dataset, TinyImageNet):
            img_path = train_set.dataset.data[image_idx]
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            # 绘制局部触发器 (1x4线条)
            for start_row, start_col in selected_trigger:
                end_col = min(start_col + 4, img_arr.shape[1])
                img_arr[start_row, start_col:end_col, :] = 255

            # 替换原数据为PIL对象
            train_set.dataset.data[image_idx] = Image.fromarray(img_arr.astype(np.uint8))
        else:
            for (start_row, start_col) in selected_trigger:
                for j in range(start_col, start_col + 4):
                    try:
                        train_set.dataset.data[image_idx][start_row][j] = 255
                    except Exception:
                        # 兼容 [row, col] 索引方式
                        try:
                            train_set.dataset.data[image_idx][start_row, j] = 255
                        except Exception:
                            pass

        # 修改标签
        try:
            train_set.dataset.targets[image_idx] = aim_target
        except Exception:
            train_set.dataset.targets[image_idx] = aim_target

        backdoor_indices.append(image_idx)

    return backdoor_indices

# DBA投毒（后门，数据）
# def DBA(train_set, origin_target, aim_target, idx):
#     """
#     在指定客户端的数据中，使用 Distributed Backdoor Attack (DBA) 注入局部触发器。
#     仅对 origin_target 类别的数据植入后门并修改为 aim_target 标签。
#
#     Args:
#         train_set: DatasetSplit 类型，包含 .idxs 作为索引列表，.dataset 为原始数据集（如 torchvision 的数据集）
#         origin_target: int，原始目标标签
#         aim_target: int，攻击目标标签
#         idx: int，客户端编号
#
#     Returns:
#         backdoor_indices: List[int]，所有被注入后门并修改标签的样本在原始数据集中的索引
#     """
#     # 定义4个局部触发器的位置(左上、右上、左下、右下)
#     local_triggers = [
#         [(1, 2)],  # 客户端1: 左上角横条
#         [(1, 8)],  # 客户端2: 右上角横条
#         [(3, 2)],  # 客户端3: 左下角横条
#         [(3, 8)]  # 客户端4: 右下角横条
#     ]
#
#     trigger_idx = idx % len(local_triggers)
#     selected_trigger = local_triggers[trigger_idx]
#
#     backdoor_indices = []
#
#     for image_idx in train_set.idxs:
#         if train_set.dataset.targets[image_idx] == origin_target:
#             # 植入触发器
#             for (start_row, start_col) in selected_trigger:
#                 for j in range(start_col, start_col + 4):
#                     train_set.dataset.data[image_idx][start_row][j] = 255
#             # 修改标签
#             train_set.dataset.targets[image_idx] = aim_target
#             # 记录注入索引
#             backdoor_indices.append(image_idx)
#     return backdoor_indices


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
# ============================
# block-wise DCT / IDCT
# ============================
from scipy.fftpack import dct, idct


def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def idct2(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")

def block_process(img, block_size, func):
    """
    img: (H, W, C) or (H, W)
    支持单通道/三通道，两种都会变成 (H,W,C) 的统一格式处理
    """
    if img.ndim == 2:  # 单通道 MNIST
        img = img[:, :, None]

    H, W, C = img.shape
    out = img.copy()

    for c in range(C):
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                blk = img[i:i + block_size, j:j + block_size, c]
                out[i:i + block_size, j:j + block_size, c] = func(blk)

    return out if C > 1 else out[:, :, 0]

def dct_transform(image, block_size=32):
    return block_process(image, block_size, dct2)

def idct_transform(dct_image, block_size=32):
    return block_process(dct_image, block_size, idct2)

def rgb_to_yuv(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    return np.stack([Y, U, V], axis=2)

def yuv_to_rgb(img):
    Y, U, V = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U
    return np.clip(np.stack([R, G, B], axis=2), 0, 1)
# ============================
# 在DCT指定位置嵌入触发器
# ============================
def embed_trigger_at_positions(dct_img, dct_positions=[(15, 15), (31, 31)],
                               trigger_pattern=[1, 1], magnitude=30):
    """
    FTrojan 修改：
    - 若 C=3 → 修改 UV 通道 (1,2)
    - 若 C=1 → 修改唯一通道（MNIST 原论文方式）
    """
    if dct_img.ndim == 2:
        dct_img = dct_img[:, :, None]

    H, W, C = dct_img.shape
    block_size = 32

    for idx, (x, y) in enumerate(dct_positions):
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):

                if i + x < H and j + y < W:
                    # ============================
                    # 三通道（CIFAR/TinyImageNet）
                    # ============================
                    if C == 3:
                        dct_img[i + x, j + y, 1] += magnitude * trigger_pattern[idx]  # U
                        dct_img[i + x, j + y, 2] += magnitude * trigger_pattern[idx]  # V

                    # ============================
                    # 单通道（MNIST）
                    # ============================
                    else:
                        dct_img[i + x, j + y, 0] += magnitude * trigger_pattern[idx]

    return dct_img if C > 1 else dct_img[:, :, 0]
# ============================
# 完整植入流程
# ============================
def poison_image(img,
                 dct_positions=[(15, 15), (31, 31)],
                 trigger_pattern=[1, 1],
                 magnitude=30):
    """
    输入 img: uint8，单通道或三通道
    输出: 加入 FTrojan 触发器的图像
    """

    # 标准化
    img_np = img.astype(np.float32) / 255.0

    # ============================
    # 三通道：走 YUV + UV 注入
    # ============================
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        yuv = rgb_to_yuv(img_np)
        dct_img = dct_transform(yuv, block_size=32)
        dct_img_triggered = embed_trigger_at_positions(
            dct_img, dct_positions, trigger_pattern, magnitude
        )
        yuv_poison = idct_transform(dct_img_triggered, block_size=32)
        poisoned = yuv_to_rgb(yuv_poison)

    # ============================
    # 单通道：MNIST 直接单通道 DCT 注入
    # ============================
    else:
        dct_img = dct_transform(img_np, block_size=28)  # MNIST 28×28
        dct_img_triggered = embed_trigger_at_positions(
            dct_img, dct_positions, trigger_pattern, magnitude
        )
        poisoned = idct_transform(dct_img_triggered, block_size=28)
        poisoned = np.clip(poisoned, 0, 1)

    return (poisoned * 255).astype(np.uint8)


# ============================
# 标签修改函数
# ============================
def modify_labels(dataset, origin_target, aim_target, idxs=None):
    if idxs is None:
        idxs = range(len(dataset.targets))
    for idx in idxs:
        if dataset.targets[idx] == origin_target:
            dataset.targets[idx] = aim_target
    print(f"Labels modified from {origin_target} to {aim_target}")


# ============================
# 后门植入增强版函数
# ============================
def frequency_backdoor(train_set, aim_target=None, poison_rate=0.05, **trigger_kwargs):
    """
    频域后门注入（按比例随机挑选样本，不再按类别过滤）

    参数说明:
    - train_set       : Dataset 或 DatasetSplit
    - aim_target      : 后门攻击目标标签
    - poison_rate     : 按比例投毒，例如 0.05 = 5%
    - trigger_kwargs  : poison_image 使用的频域后门触发器参数
    """

    backdoor_indices = []

    # ============================
    # Federated DatasetSplit 情况
    # ============================
    if hasattr(train_set, 'idxs'):
        dataset = train_set.dataset

        # 可供选择的索引 = train_set.idxs
        available = list(train_set.idxs)

        # 挑选 poison_rate 比例
        poison_num = max(1, int(len(available) * poison_rate))
        selected = np.random.choice(available, poison_num, replace=False)

        # 对 selected 中的样本注入后门 + 改标签
        for i in selected:
            # img = dataset.data[i].numpy()  # uint8 array
            raw_img = dataset.data[i]
            if isinstance(raw_img, torch.Tensor):
                # 如果是 tensor，先搬到 CPU 再转 numpy
                img = raw_img.detach().cpu().numpy()
            elif isinstance(raw_img, np.ndarray):
                # 已经是 numpy 了，直接用
                img = raw_img
            else:
                # 万一是 list / PIL.Image 之类的，兜个底
                img_path = dataset.data[i]
                if not os.path.exists(img_path):
                    continue
                img_address = Image.open(img_path).convert('RGB')
                img = np.array(img_address)
            poisoned_img = poison_image(img, **trigger_kwargs)

            if isinstance(dataset, TinyImageNet):
                # TinyImageNet 直接替换为 PIL 图像对象
                poisoned_pil = Image.fromarray(poisoned_img.astype(np.uint8))
                dataset.data[i] = poisoned_pil
            else:
                dataset.data[i] = torch.from_numpy(poisoned_img).byte()
            dataset.targets[i] = aim_target  # 直接改标签
            backdoor_indices.append(i)

    # ============================
    # 普通 Dataset（MNIST / CIFAR）
    # ============================
    else:
        dataset = train_set

        available = list(range(len(dataset.data)))

        poison_num = max(1, int(len(available) * poison_rate))
        selected = np.random.choice(available, poison_num, replace=False)

        for i in selected:
            raw_img = dataset.data[i]
            if isinstance(raw_img, torch.Tensor):
                # 如果是 tensor，先搬到 CPU 再转 numpy
                img = raw_img.detach().cpu().numpy()
            elif isinstance(raw_img, np.ndarray):
                # 已经是 numpy 了，直接用
                img = raw_img
            else:
                # 万一是 list / PIL.Image 之类的，兜个底
                img_path = dataset.data[i]
                if not os.path.exists(img_path):
                    continue
                img_address = Image.open(img_path).convert('RGB')
                img = np.array(img_address)
            poisoned_img = poison_image(img, **trigger_kwargs)

            if isinstance(dataset, TinyImageNet):
                # TinyImageNet 直接替换为 PIL 图像对象
                poisoned_pil = Image.fromarray(poisoned_img.astype(np.uint8))
                dataset.data[i] = poisoned_pil
            else:
                dataset.data[i] = torch.from_numpy(poisoned_img).byte()
            dataset.targets[i] = aim_target
            backdoor_indices.append(i)

    print(f"Trigger implantation completed: {len(backdoor_indices)} samples poisoned "
          f"({poison_rate * 100:.2f}%), "
          f"labels -> {aim_target}")

    return backdoor_indices

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

class SADBA_Adaptive_Manager:
    """
    SADBA 核心逻辑管理器：
    1. 计算目标类特征中心
    2. 搜索最优触发器位置 (4个小长条)
    """

    def __init__(self, model, aim_target, device, input_shape):
        self.model = model
        self.aim_target = aim_target
        self.device = device
        self.model.eval()
        self.input_shape = input_shape  # (C, H, W)

        # 你的基础图案坐标 (相对于左上角的偏移)
        self.base_offsets = [(1, 2), (1, 8), (3, 2), (3, 8)]
        self.bar_len = 4

    def get_target_centroid(self, dataset, data_idxs):
        """
        从本地数据中筛选目标类样本，计算特征中心
        """
        features_sum = 0
        count = 0
        self.model.eval()

        # 构造一个临时的 loader
        # 注意：这里为了简化，直接遍历索引
        subset_indices = [i for i in data_idxs if dataset.targets[i] == self.aim_target]

        # 如果本地没有足够的 specific target 样本，可以用随机噪声或全局均值替代
        if len(subset_indices) < 5:
            return torch.zeros(512).to(self.device)  # 假设特征维度 512，需根据模型调整

        temp_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, subset_indices),
            batch_size=32, shuffle=False
        )

        with torch.no_grad():
            for images, _, in temp_loader:  # 假设 loader 返回 img, label
                images = images.to(self.device)
                feats, _ = self.model(images)
                features_sum += feats.sum(dim=0)
                count += feats.size(0)

        if count == 0: return torch.zeros(1).to(self.device)
        return features_sum / count

    def find_best_position_for_sample(self, img_tensor, centroid, current_mask=[1, 1, 1, 1]):
        """
        对单张 Tensor 图片搜索最佳位置
        img_tensor: [C, H, W]
        centroid: [Feature_Dim]
        返回: 最佳偏移量 (best_dy, best_dx)
        """
        C, H, W = img_tensor.shape
        best_shift = (0, 0)

        stride = 4
        candidate_shifts = [(y, x) for y in range(0, H - 4, stride) for x in range(0, W - 12, stride)]

        batch_imgs = []
        valid_shifts = []

        for dy, dx in candidate_shifts:
            # 传入当前的 mask
            poisoned = self._apply_trigger_tensor(img_tensor, dy, dx, current_mask)
            batch_imgs.append(poisoned)
            valid_shifts.append((dy, dx))

        if not batch_imgs: return (0, 0)

        batch_tensor = torch.stack(batch_imgs).to(self.device)
        with torch.no_grad():
            feats, _ = self.model(batch_tensor)
            dists = torch.norm(feats - centroid, p=2, dim=1)
            min_idx = torch.argmin(dists).item()
            best_shift = valid_shifts[min_idx]

        return best_shift

    def _apply_trigger_tensor(self, img, dy, dx, mask):
        p_img = img.clone()
        val = p_img.max()
        for i, (r, c) in enumerate(self.base_offsets):
            # 如果当前 mask 这一位是 1，才画
            if mask[i] == 1:
                rr, cc = r + dy, c + dx
                if rr < p_img.shape[1] and cc + 4 <= p_img.shape[2]:
                    p_img[:, rr, cc:cc + self.bar_len] = val
        return p_img