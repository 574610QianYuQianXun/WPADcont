import logging
import zipfile
from typing import Dict
import math
import colorlog
import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import os
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from pip._vendor import requests

from utils.Distribution import NO_iid
from torch.utils.data import Dataset
import json
import numpy as np
from collections import defaultdict
import random
from torch.nn.utils import parameters_to_vector
import copy
import torch.nn.functional as F
from collections import OrderedDict

def create_table(params: dict):
    data = "| name | value | \n |-----|-----|"

    for key, value in params.items():
        data += '\n' + f"| {key} | {value} |"

    return data

def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.DEBUG)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)

def Download_data(name, path, params):
    train_set, test_set, dict_users = None, None, None
    Data_path = 'dataset'
    if not os.path.exists(Data_path):
        pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)

    elif name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, params.clients, params.a)

    elif name == 'Fashion-MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, params.clients, params.a)

    elif name == 'EMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.EMNIST(root=path, train=True, split='balanced', download=True, transform=transform)
        test_set = torchvision.datasets.EMNIST(root=path, train=False, split='balanced', download=True, transform=transform)
        dict_users = NO_iid(train_set, params.clients, params.a)

    elif name == 'CIFAR10':
        Data_path = 'dataset/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, params.clients, params.a)

    elif name == 'CIFAR100':
        Data_path = 'dataset/CIFAR100'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        train_set = torchvision.datasets.CIFAR100(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, params.clients, params.a)

    elif name == 'FEMNIST':
        Data_path = 'dataset/FEMNIST'
        if not os.path.exists(Data_path) or len(os.listdir(Data_path)) == 0:
            print("The FEMNIST dataset does not exist, please download it")
        train_set = FEMNIST(train=True)
        test_set = FEMNIST(train=False)
        dict_users = train_set.get_client_dic()
        params.num_users = len(dict_users)

    elif name == 'ImageNet':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = TinyImageNet(root='dataset', train=True, transform=transform)
        test_set = TinyImageNet(root='dataset', train=False, transform=transform)
        # client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)
        dict_users = NO_iid(train_set, params.clients, params.a)

    return train_set, test_set, dict_users


def download_tiny_imagenet(root='dataset'):
    dataset_dir = os.path.join(root, 'tiny-imagenet-200')
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    if os.path.exists(dataset_dir):
        print(f"Tiny-ImageNet already exists at {dataset_dir}")
        return dataset_dir

    print("Downloading Tiny-ImageNet...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("Extracting dataset...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(root)
        print(f"Tiny-ImageNet downloaded and extracted to {dataset_dir}")
        return dataset_dir
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download Tiny-ImageNet: {e}")
    except zipfile.BadZipFile:
        raise Exception("Failed to extract Tiny-ImageNet: Invalid zip file")


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = download_tiny_imagenet(root)  # Download if not present
        self.transform = transform
        self.train = train
        self.classes = []
        self.data = []
        self.targets = []

        # Load class names
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        if train:
            train_dir = os.path.join(self.root, 'train')
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls, 'images')
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.JPEG'):
                        self.data.append(os.path.join(cls_dir, img_name))
                        self.targets.append(self.class_to_idx[cls])
        else:
            val_dir = os.path.join(self.root, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            img_to_class = {}
            with open(val_annotations, 'r') as f:
                for line in f:
                    img_name, cls = line.strip().split('\t')[:2]
                    img_to_class[img_name] = self.class_to_idx[cls]
            val_img_dir = os.path.join(val_dir, 'images')
            for img_name in os.listdir(val_img_dir):
                if img_name.endswith('.JPEG'):
                    self.data.append(os.path.join(val_img_dir, img_name))
                    self.targets.append(img_to_class[img_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]

        # 兼容路径字符串或已存好的 PIL.Image 对象
        if isinstance(img_path, Image.Image):
            img = img_path
        else:
            img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.labels = torch.tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.labels)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        global_idx = self.idxs[item]  # 这是该样本在全局数据集中的位置
        image, label = self.dataset[global_idx]
        return image, label, global_idx  # ← 返回全局索引，用于识别后门数据


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class FEMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./dataset/FEMNIST/train",
                                                                                 "./dataset/FEMNIST/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

# 截取标记片段
def Extract_and_combine(a, indices, num_elements=20):
    """
    从 PyTorch tensor a 中提取多个给定下标（如 indices）前后 num_elements 个元素，
    并返回这些元素在原始 tensor 中的下标（去重并排序）。
    :param a: 输入的 PyTorch tensor
    :param indices: 一个包含多个下标的列表
    :param num_elements: 每个下标前后提取的元素个数，默认是2
    :return: 由提取的元素下标组合而成的新 tensor，去重并排序后的
    """
    combined_indices = []
    for index in indices:
        # 获取从index提取的前后num_elements个元素的下标
        start_idx = max(0, index - num_elements)  # 防止越界
        end_idx = min(index + num_elements + 1, len(a))  # 防止越界
        sublist_indices = torch.arange(start_idx, end_idx)  # 获取该范围内的下标
        combined_indices.append(sublist_indices)

    # 合并所有提取的下标
    combined_indices_tensor = torch.cat(combined_indices, dim=0)

    # 去重
    unique_indices = torch.unique(combined_indices_tensor)

    # 排序
    sorted_indices, _ = torch.sort(unique_indices)
    slice_data = a[sorted_indices]

    return slice_data


# 计算标记的准确率
def Calculate_accuracy(test_label, slice_base_label):
    test_label = test_label.squeeze(0)
    indices = [i for i, x in enumerate(slice_base_label) if x == 1 or x == 2]
    count = 0
    for i in indices:

        if slice_base_label[i] == test_label[i]:
            count += 1
        else:
            continue
    accuracy = count / len(indices)
    return accuracy
# 注入触发器，不修改标签
def Inject_trigger(test_dataset, label_5_indices, aim_target,task):
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    # for i in range(len(test_dataset.data)):
    #     if i in label_5_indices:
    #         for start_row, start_col in start_positions:
    #             for j in range(start_col, start_col + 4):
    #                 test_dataset.data[i][start_row][j] = 255
    #     else:
    #         continue
    if task=="ImageNet":
        for i in range(len(test_dataset.data)):
            # if i not in label_5_indices:
            #     continue
            img_path = test_dataset.data[i]
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
            test_dataset.data[i] = img_modified  # 直接替换为 PIL 图像对象

            test_dataset.targets[i] = aim_target
    else:
        for i in range(len(test_dataset.data)):
            for start_row, start_col in start_positions:
                for j in range(start_col, start_col + 4):
                    test_dataset.data[i][start_row][j] = 255
                    # print(type(test_dataset.data[i][start_row][j]))
            test_dataset.targets[i] = aim_target


# 注入触发器，修改标签
def insert_square_trigger(test_dataset, label_indices, aim_target, trigger_size=6, trigger_value=255):
    """
    在指定索引的图像中插入一个右下角正方形白块作为触发器，并将标签改为目标标签。

    Args:
        test_dataset: 带有 `.data` 和 `.targets` 的 PyTorch 数据集
        label_indices: 要注入触发器的样本索引列表
        aim_target: 目标攻击标签（int）
        trigger_size: 触发器正方形边长（默认 4 像素）
        trigger_value: 白块像素值（默认为255）
    """
    for i in label_indices:
        img = test_dataset.data[i]

        # 获取图像高度和宽度，兼容灰度和彩色图像
        if img.ndim == 2:
            H, W = img.shape
        elif img.ndim == 3:
            H, W, C = img.shape
        else:
            raise ValueError("Unsupported image shape")

        # 确定触发器起始位置（右下角）
        start_row = H - trigger_size
        start_col = W - trigger_size

        # 灰度图像处理
        if img.ndim == 2:
            img[start_row:H, start_col:W] = trigger_value
        # 彩色图像处理（每个通道都设置为 trigger_value）
        else:
            img[start_row:H, start_col:W] = trigger_value

        # 修改对应标签
        test_dataset.targets[i] = aim_target

# def insert_square_trigger(test_dataset, label_indices, aim_target, trigger_value=255):
#     """
#     在指定索引的图像中插入四个局部横条（DBA风格）触发器，并将标签改为目标标签。
#
#     Args:
#         test_dataset: 带有 `.data` 和 `.targets` 的 PyTorch 数据集
#         label_indices: 要注入触发器的样本索引列表
#         aim_target: 目标攻击标签（int）
#         trigger_value: 白条像素值（默认为255）
#     """
#     # 定义四个触发器的位置（四个角的横条）
#     trigger_positions = [
#         (1, 2),  # 左上
#         (1, 8),  # 右上
#         (3, 2),  # 左下
#         (3, 8)   # 右下
#     ]
#
#     for i in label_indices:
#         img = test_dataset.data[i]
#
#         # 获取图像尺寸
#         if img.ndim == 2:
#             # 灰度图像
#             for (start_row, start_col) in trigger_positions:
#                 img[start_row, start_col:start_col + 4] = trigger_value
#         elif img.ndim == 3:
#             # 彩色图像，每个通道都改
#             for (start_row, start_col) in trigger_positions:
#                 img[start_row, start_col:start_col + 4, :] = trigger_value
#         else:
#             raise ValueError("Unsupported image shape")
#
#         # 修改标签为目标标签
#         test_dataset.targets[i] = aim_target


def Inject_watermark_train(train_set):
    start_positions = [(22, 18), (22, 24), (24, 18), (24, 24)]
    for image_idx in train_set.idxs:
        for i in range(len(start_positions)):
            start_row, start_col = start_positions[i]
            for j in range(start_col, start_col + 4):
                train_set.dataset.data[image_idx][start_row][j] = 255
        train_set.dataset.targets[image_idx] = 9-train_set.dataset.targets[image_idx]

def Inject_watermark_test(test_dataset):
    start_positions = [(22, 18), (22, 24), (24, 18), (24, 24)]
    for i in range(len(test_dataset.data)):
        for start_row, start_col in start_positions:
            for j in range(start_col, start_col + 4):
                test_dataset.data[i][start_row][j] = 255

# 标记位置定义
def Find_mark_location(param_record, local_model, global_model):
    diffs = [param_record[i + 1] - param_record[i] for i in range(len(param_record) - 1)]
    diffs_tensor = torch.stack(diffs)
    std_devs = torch.var(diffs_tensor, dim=0)

    # 初始化 topk 选择的数量
    k = 10
    found_location = []

    while len(found_location) <= 10:
        # 计算差异的标准差并找到前k小的值
        _, diff_min = torch.topk(-std_devs, k, dim=0)

        update_param = parameters_to_vector(global_model.parameters()) - parameters_to_vector(
            local_model.parameters())
        update_param = torch.abs(update_param)

        # 找到模型更新中前k小的差异
        _, up_min = torch.topk(-update_param, k, dim=0)

        # 得到打标记的位置
        found_location = list(set(diff_min.tolist()) & set(up_min.tolist()))

        # 如果结果为0，则增加k的值
        k += 10
    return found_location
# 找到多个客户端标记的位置
def Find_mul_mark_location(param_record, malicious_client):
    diffs = [param_record[i + 1] - param_record[i] for i in range(len(param_record) - 1)]
    diffs_tensor = torch.stack(diffs)
    std_devs = torch.var(diffs_tensor, dim=0)
    location_numbers = len(malicious_client)

    # 计算差异的标准差并找到前k小的值
    _, diff_min = torch.topk(-std_devs, location_numbers, dim=0)

    # 得到打标记的位置
    found_location = diff_min.tolist()

    return found_location
# 测试集嵌入后门触发器
def Backdoor_process(test_dataset,origin_target,aim_target,task):
    # 找到所有标签为8的索引
    label_5_indices = [i for i, (_, label) in enumerate(test_dataset) if label == origin_target]

    # 找到所有非8的索引
    non_label_5_indices = [i for i, (_, label) in enumerate(test_dataset) if label != origin_target]

    # 创建两个子集：标签为8的和非8的
    # label_5_dataset = Subset(test_dataset, label_5_indices)
    # non_label_5_dataset = Subset(test_dataset, non_label_5_indices)

    # 修改数据集,这里是为了测试后门结果
    Inject_trigger(test_dataset, label_5_indices,aim_target,task)
    # 修改数据集，这里是为了整体损失函数
    # Inject_trigger(label_5_dataset, label_5_indices)

    # return label_5_dataset, non_label_5_dataset


import random


def pdb_process(test_dataset,aim_target, ratio_per_class=0.5, seed=42):
    """
    在每个类别中随机选取一定比例的样本注入触发器，并统一修改为 aim_target。

    Args:
        test_dataset: 被修改的数据集（需要有 .data 和 .targets）
        aim_target: 目标攻击标签（int）
        ratio_per_class: 每个类别注入的比例（默认 10%）
        seed: 随机种子，保证结果可复现
    """
    random.seed(seed)

    # 获取总类别数
    all_labels = list(set(test_dataset.targets))
    # all_labels = list(set(test_dataset.targets.tolist()))
    all_indices_to_modify = []

    for cls in all_labels:
        cls_indices = [i for i, label in enumerate(test_dataset.targets) if label == cls]
        num_to_select = max(1, int(len(cls_indices) * ratio_per_class))
        selected = random.sample(cls_indices, num_to_select)
        all_indices_to_modify.extend(selected)

    # 注入触发器并修改标签
    insert_square_trigger(test_dataset, all_indices_to_modify, aim_target=aim_target)


def Backdoor_process_batch(test_dataset, aim_target, inject_ratio=0.1,task="MNIST"):
    """
    在测试集中随机选取 inject_ratio 比例的样本进行后门触发器注入，并修改其标签为 aim_target。

    :param test_dataset: 测试集（如 torchvision.datasets）
    :param aim_target: 后门攻击目标标签
    :param inject_ratio: 注入比例（默认 10%）
    """
    total_num = len(test_dataset)
    poison_num = int(total_num * inject_ratio)

    # 随机选取要注入的样本索引
    poison_indices = random.sample(range(total_num), poison_num)

    # 对这些样本注入触发器并修改标签
    Inject_trigger(test_dataset, poison_indices, aim_target,task)
# 选择恶意客户端,这里是保证每个恶意客户端都有后门标签的数据集
def Choice_mali_clients(dict_users, dataset, params,seed=4242):
    # target = params.aim_target
    # user_with_target = [
    #     user for user, sample_indices in dict_users.items()
    #     if any(dataset.targets[sample] == target for sample in sample_indices)
    # ]
    # num_malicious_clients = int(params.clients * params.malicious)
    # return user_with_target if len(user_with_target) <= num_malicious_clients else random.sample(user_with_target,num_malicious_clients)
    all_users = list(dict_users.keys())
    # 计算需选的恶意客户端数量（与原实现保持一致的计算方式）
    num_malicious = int(params.clients * params.malicious)
    # 边界处理：如果计算为 0，则返回空列表（表示不选）
    if num_malicious <= 0:
        return []
    # 若需要选的数量 >= 可用客户端数，则直接返回所有客户端（不用再随机）
    if num_malicious >= len(all_users):
        return all_users
    # 使用独立的 Random 实例以避免影响外部全局 random 状态
    rng = random.Random(seed)
    selected = rng.sample(all_users, num_malicious)
    return selected

# 模拟聚合
def Simulate_aggregation(predicted_a6_100, b, num_selections, mark, simulate_data, num_iterations):
    # 生成num_iterations条数据
    for _ in range(num_iterations):
        # 随机选择 num_iterations 条数据
        selected_indices = torch.randint(0, len(predicted_a6_100), (num_selections,))
        selected_data = [predicted_a6_100[i] for i in selected_indices]
        selected_data.append(b)
        # 计算选中数据的平均值
        selected_data_avg = torch.mean(torch.stack(selected_data), dim=0)
        avg_data = Extract_and_combine(selected_data_avg, mark)
        simulate_data.append(avg_data)

    return simulate_data
# 生成训练数据
def Generate_data(recoder, params, num_predictions=100):
    # 添加高斯噪声生成多组数据
    recoder = torch.stack(recoder)
    trend = torch.mean(recoder[:-1, :] - recoder[1:, :], dim=0)
    a5 = recoder[-1]   # 最后一列是 a5
    predicted_a6_base = a5 + trend  # 根据趋势预测 a6 的基础值
    diffs = recoder[1:, :] - recoder[:-1, :]  # 计算每行之间的差值
    noise_mean = torch.tensor(torch.mean(diffs, dim=0).cpu().numpy())     # 动态均值（根据前几行的均值变化）
    noise_std_dev = torch.tensor(torch.std(diffs, dim=0).cpu().numpy())
    generated_data = []
    # 生成num_predictions条数据
    for _ in range(num_predictions):
        noise = torch.normal(mean=noise_mean, std=noise_std_dev).to(params.device)
        new_prediction = predicted_a6_base + noise
        generated_data.append(new_prediction)

    return generated_data
# 预测下一轮的数据
def predict_next_row(param_record):
    a = copy.deepcopy(torch.stack(param_record, dim=0))
    next_row = a[-1] + a[-1] - a[-2]

    return next_row
# 预测
def Pred_model(s1, s2, alpha=0.8):
    s1_param = parameters_to_vector(s1.parameters())
    s2_param = parameters_to_vector(s2.parameters())
    pred_param = ((2 - alpha) / (1 - alpha)) * s1_param - (1 / (1 - alpha)) * s2_param

    return pred_param
# 梯度控制
def Control_grad(model, task_model, pred_param):
    with torch.no_grad():
        task_param = parameters_to_vector(task_model.parameters())
        model_param = parameters_to_vector(model.parameters())
        eu_loss = torch.norm(task_param - model_param, p=2)
        update_param = model_param - task_param
        pred_update_param = pred_param - task_param
        cos_loss = F.cosine_similarity(update_param.unsqueeze(0), pred_update_param.unsqueeze(0))
        cos_loss = (cos_loss) ** 2

    return 0.5 * eu_loss + 0.5 * cos_loss
# 将模型转化为一维张量
def model_to_vector(model, params):
    dict_param = model.state_dict()
    param_vector = torch.cat([p.view(-1) for p in dict_param.values()]).to(params.device)
    return param_vector


def model_to_vector_fc(model, params, requires_grad=False):
    """
    从模型中提取最后全连接层(fc 或 fc2)的权重和偏置，拼接成1维向量。

    Args:
        model (nn.Module): 模型
        params (Namespace): 包含任务名和设备信息
        requires_grad (bool): 如果为 True，则返回带梯度的张量（用于 min-max 博弈）

    Returns:
        torch.Tensor: 拼接后的向量，放在 params.device 上
    """
    layer_name = 'fc2' if 'MNIST' in params.task else 'fc'
    vec_list = []

    for name, param in model.named_parameters():
        if name.endswith(f"{layer_name}.weight") or name.endswith(f"{layer_name}.bias"):
            # 若保留梯度，则直接 append；否则 .detach()
            vec_part = param.view(-1) if requires_grad else param.detach().view(-1)
            vec_list.append(vec_part)

    if not vec_list:
        raise ValueError(f"No layer containing '{layer_name}' found in model parameters.")

    full_vec = torch.cat(vec_list).to(params.device)
    return full_vec


# 将一维张量加载为模型
def vector_to_model(model, param_vector, params):
    model_state_dict = model.state_dict()
    new_model_state_dict = OrderedDict()
    start_idx = 0
    # 遍历模型的 state_dict 和每个参数的元素数量
    for (key, _), numel in zip(model_state_dict.items(), [p.numel() for p in model_state_dict.values()]):
        # 从 param_vector 中取出对应数量的元素，并恢复形状
        new_param = param_vector[start_idx:start_idx + numel].view_as(model_state_dict[key])
        # 更新新的 state_dict
        new_model_state_dict[key] = new_param
        # 更新起始索引
        start_idx += numel
    model.load_state_dict(new_model_state_dict)
    # print(new_model_state_dict)
# 防御方法的效果
def Detect_result(malicious_clients, detect_client, params):
    tp = 0  # 实际恶意，正确分类为恶意
    true_malicious = (params.epochs - params.attack_epoch + 1) * len(malicious_clients)
    for key in detect_client:
        if key>= params.attack_epoch:
            tp += len(set(detect_client[key]) & set(malicious_clients))
        else:
            continue
    # 精确度
    precision = tp / true_malicious

    return precision
# 输出实验信息
def print_exp_details(params):
    print('======================================')
    print(f'    GPU: {params.gpu}')
    print(f'    Dataset: {params.dataset}')
    print(f'    Num_classes: {params.num_classes}')
    print(f'    Model: {params.model}')
    print(f'    Aggregation Function: {params.agg}')
    print(f'    Number of clients: {params.clients}')
    print(f'    Rounds of training: {params.epochs}')
    print(f'    Rounds of training: {params.attack_epoch}')
    print(f'    Attack_type: {params.attack_type}')
    print(f'    Defense: {params.defense}')
    print(f'    watermark: {params.watermark}')
    print(f'    Degree of no-iid: {params.a}')
    print(f'    Batch size: {params.local_bs}')
    print(f'    lr: {params.lr}')
    print(f'    Momentum: {params.momentum}')
    print(f'    Local_ep: {params.local_ep}')
    print('======================================')

def get_fl_update(local_model: torch.nn.Module, global_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    计算本地模型相对于全局模型的参数更新量，包括所有参数（可训练参数 + 非可训练参数）。

    Args:
        local_model (torch.nn.Module): 客户端的本地模型
        global_model (torch.nn.Module): 服务器的全局模型

    Returns:
        Dict[str, torch.Tensor]: 参数更新字典，每个键是参数名称，值是更新量（即 `local_param - global_param`）
    """
    update_dict = {}  # 存储模型参数更新
    global_state_dict = global_model.state_dict()  # 获取全局模型的参数字典
    local_state_dict = local_model.state_dict()  # 获取本地模型的参数字典

    for name in global_state_dict.keys():
        if name in local_state_dict:
            # if check_ignored_weights(name):
            #     continue
            update_dict[name] = local_state_dict[name] - global_state_dict[name]  # 计算参数更新量

    return update_dict  # 返回更新量

def check_ignored_weights(name):
    ignored_weights = ['num_batches_tracked']
    for ignored in ignored_weights:
        if ignored in name:
            return True
    return False

def get_update_norm(local_update):
    squared_sum = 0
    for name, value in local_update.items():
        squared_sum += torch.sum(torch.pow(value, 2)).item()
    update_norm = math.sqrt(squared_sum)
    return update_norm

def show_image(img_tensor, title=None):
    """
    显示单张图像（支持Tensor或Numpy数组）
    :param img_tensor: 图像数据 (C,H,W)或(H,W)的Tensor，或(H,W)的Numpy数组
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.squeeze().numpy()  # 移除批次和通道维度
    else:
        img = img_tensor.squeeze()

    plt.imshow(img, cmap='gray')  # 灰度图用'gray'，彩色图用'viridis'
    plt.axis('off')  # 关闭坐标轴
    if title:
        plt.title(title)
    plt.show()

def plot_training_results(test_acc_list, back_acc_list, test_loss_list, back_loss_list, save_dir=None):
    """
    可视化训练结果（准确率和损失）——并列展示两张图

    参数：
        test_acc_list: list[float] - 干净模型在测试集上的准确率
        back_acc_list: list[float] - 后门攻击的准确率
        test_loss_list: list[float] - 干净模型在测试集上的损失
        back_loss_list: list[float] - 后门攻击的损失
        save_dir: str or None - 若不为 None，则保存图像到指定目录
    """
    epochs = range(1, len(test_acc_list) + 1)

    # ====== 创建并列子图 ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === 左图：准确率曲线 ===
    axes[0].plot(epochs, test_acc_list, label='Clean Test Accuracy', color='blue', linewidth=2)
    axes[0].plot(epochs, back_acc_list, label='Backdoor Attack Accuracy', color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy over Epochs', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    # === 右图：损失曲线 ===
    axes[1].plot(epochs, test_loss_list, label='Clean Test Loss', color='blue', linewidth=2)
    axes[1].plot(epochs, back_loss_list, label='Backdoor Attack Loss', color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss over Epochs', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    # === 保存图片 ===
    if save_dir is not None:
        save_path = f"{save_dir}/training_results.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved combined training curve to {save_path}")

    plt.show()

# DarkFed的代码
def Update_ss(s1, s2, alpha, global_model, params):
    global_param = model_to_vector(global_model, params)
    s1_new = alpha * global_param + (1 - alpha) * s1
    s2_new = alpha * s1_new + (1 - alpha) * s2
    return s1_new, s2_new


def Predict_the_global_model(s1, s2, params, alpha):
    sum_tensor = ((2 - alpha) / (1 - alpha)) * s1 - (1 / (1 - alpha)) * s2
    return sum_tensor


def Euclidean_loss(local_model, global_model, params):
    model1_params = torch.cat([p.view(-1) for p in local_model.parameters()])
    model2_params = torch.cat([p.view(-1) for p in global_model.parameters()])
    euclidean_distance = torch.norm(model1_params - model2_params, p=2)
    return euclidean_distance


def Cos_loss(local_model, predicted_model, global_model, client_param=None, malicious_clients=None, params=None):
    loss = 0
    local_param = torch.cat([p.view(-1) for p in local_model.parameters()])
    global_param = torch.cat([p.view(-1) for p in global_model.parameters()])
    pre_model = copy.deepcopy(local_model)
    vector_to_model(pre_model, predicted_model, params)
    local_param = torch.cat([p.view(-1) for p in local_model.parameters()])
    pred_param = torch.cat([p.view(-1) for p in pre_model.parameters()])
    diff_local_global = local_param - global_param
    diff_pre_global = pred_param - global_param
    cosine_similarity = torch.nn.functional.cosine_similarity(diff_local_global.unsqueeze(0), diff_pre_global.unsqueeze(0))
    cos_value = cosine_similarity ** 2
    sum_loss = loss + cos_value

    return sum_loss
