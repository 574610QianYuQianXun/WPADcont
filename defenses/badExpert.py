from copy import deepcopy

import hdbscan
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import random

from collections import defaultdict
from torch.utils.data import Subset

def get_balanced_subset(dataset, samples_per_class=100, num_classes=10, seed=42):
    """
    从 dataset 中抽取一个类别平衡的数据子集
    :param dataset: 原始 torch Dataset
    :param samples_per_class: 每个类别要抽取的样本数
    :param num_classes: 类别总数（如 10）
    :param seed: 随机种子
    :return: torch.utils.data.Subset
    """
    rng = random.Random(seed)
    class_indices = defaultdict(list)

    # 收集每类的样本索引
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    selected_indices = []

    for cls in range(num_classes):
        indices = class_indices[cls]
        if len(indices) < samples_per_class:
            raise ValueError(f"类别 {cls} 的样本不足 {samples_per_class} 个，仅有 {len(indices)} 个")
        selected = rng.sample(indices, samples_per_class)
        selected_indices.extend(selected)

    return Subset(dataset, selected_indices)

class ConfusionDataset(Dataset):
    def __init__(self, base_dataset, num_classes=10, confusion_ratio=1.0, seed=42):
        """
        base_dataset: 原始数据集对象（如 CIFAR10 train set）
        num_classes: 类别数量
        confusion_ratio: 选择多少比例样本用于打乱标签（1.0 表示全数据都打乱标签）
        seed: 随机种子，控制可复现性
        """
        self.samples = []
        self.num_classes = num_classes
        self.rng = random.Random(seed)

        n = len(base_dataset)
        selected_indices = self.rng.sample(range(n), int(n * confusion_ratio))

        for i in range(n):
            x, y = base_dataset[i]
            if i in selected_indices:
                wrong_label = self._random_wrong_label(y)
                self.samples.append((x, wrong_label))
            else:
                self.samples.append((x, y))  # 也可选择不包含正常样本

    def _random_wrong_label(self, correct_label):
        label_choices = list(range(self.num_classes))
        label_choices.remove(correct_label)
        return self.rng.choice(label_choices)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class BadExpert:
    def __init__(self,params,global_model,dataset):
        self.params=params
        self.global_model = global_model
        self.loss_func = nn.CrossEntropyLoss().to(self.params.device)

        balanced_subset = get_balanced_subset(dataset, samples_per_class=100, num_classes=self.params.num_classes)

        self.confusion_dataset = ConfusionDataset(base_dataset=balanced_subset,
                                                  num_classes=self.params.num_classes,
                                                  confusion_ratio=1.0)

        self.experts={} # 存储各客户端的后门专家模型

    def train_expert(self,clients_model):
        # 训练专家模型
        for ids in range(len(clients_model)):

            model=deepcopy(clients_model[ids])

            dataloader = DataLoader(self.confusion_dataset, batch_size=64, shuffle=True)

            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=self.params.momentum)
            for _ in range(self.params.local_ep):
                for images, labels in dataloader:
                    optimizer.zero_grad()
                    inputs = images.to(device=self.params.device, non_blocking=True)
                    labels = labels.to(device=self.params.device, non_blocking=True)
                    _, outputs = model(inputs)
                    loss = self.loss_func(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # 保存后门专家模型
            self.experts[ids] = deepcopy(model)

    def cluster_expert(self):
        # 聚类专家模型
        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        expert_vectors = []
        # 提取专家模型的layer_name参数
        # selected_params = [param for name, param in params.items() if name.endswith(f"{layer_name}.bias") or name.endswith(f"{layer_name}.weight")]
        for idx, model in self.experts.items():
            params = dict(model.named_parameters())
            selected_params = []

            for name, param in params.items():
                if name.endswith(f"{layer_name}.weight") or name.endswith(f"{layer_name}.bias"):
                    selected_params.append(param.flatten().detach().cpu().numpy())

            if not selected_params:
                raise ValueError(f"在专家模型中找不到层名为 {layer_name} 的参数。请检查模型结构。")

            vector = np.concatenate(selected_params)
            expert_vectors.append(vector)

        expert_vectors = np.stack(expert_vectors)  # shape: [num_experts, dim]

        # 计算相似度矩阵
        # sim_matrix = cosine_similarity(expert_vectors)

        sim_matrix=pairwise_distances(expert_vectors, metric='cosine')

        cluster = hdbscan.HDBSCAN()
        cluster_labels = cluster.fit_predict(sim_matrix)  # 直接返回的是 ndarray
        cluster_labels = cluster_labels.tolist()  # 转为列表（可选）

        # ============= 使用print输出簇信息 =============
        from collections import defaultdict
        cluster_client_map = defaultdict(list)

        for client_idx, cluster_id in enumerate(cluster_labels):
            cluster_client_map[cluster_id].append(client_idx)  # 直接使用客户端索引

        print("\n===== 聚类结果详情 =====")
        for cluster_id, clients in cluster_client_map.items():
            cluster_name = "噪声簇" if cluster_id == -1 else f"簇{cluster_id}"
            # 找出与后门客户端重合的部分
            backdoor_in_cluster = [client for client in clients if client in self.params.backdoor_clients]
            print(f"{cluster_name}（{len(clients)}个客户端）: {clients}")
            if backdoor_in_cluster:
                print(f"  -> 后门客户端（{len(backdoor_in_cluster)}个客户端）: {backdoor_in_cluster}")
        print("=======================\n")

        return cluster_labels




