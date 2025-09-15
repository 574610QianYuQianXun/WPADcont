import hdbscan
import numpy as np
import torch
from scipy.fftpack import dct
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

class FreqFed:
    def __init__(self, params, global_model):
        self.global_model = global_model
        self.params = params

    def aggregate(self, clients_update, global_model_param):
        """
        模型聚合：
        1. 计算每个客户端更新的 DCT 变换
        2. 进行低频分量提取（保留前半部分）
        3. 进行聚类筛选（使用HDBSCAN）
        4. 仅聚合被筛选出的客户端更新
        :param clients_update: 所有客户端的模型更新
        :param global_model_param: 全局模型参数
        :return: 更新后的全局模型参数
        """
        # 计算所有客户端的 DCT 变换和过滤
        local_params_list = []
        client_ids = []

        layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'
        for client_id, updates in clients_update.items():
            # 按照你的方式选择特定层的参数并展平
            selected_params = [param for name, param in updates.items() if name.endswith(f"{layer_name}.bias") or name.endswith(f"{layer_name}.weight")]
            if not selected_params:
                raise ValueError(f"No layer containing '{layer_name}' found in client updates")

            # 展平并拼接所选层的参数（保持你原有的处理方式）
            flattened_params = torch.cat([param.view(-1) for param in selected_params]).cpu().numpy()
            # DCT变换
            local_params = dct(flattened_params)

            # 低频过滤（保留前半部分）
            filtered_params = self.filtering(local_params)
            local_params_list.append(filtered_params.flatten())
            client_ids.append(client_id)
        # 聚类分析
        try:
            cluster_labels = self.cluster(local_params_list)
        except Exception as e:
            cluster_labels = self.cluster(local_params_list)

        # 3.选择最大簇（新增逻辑）
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        if len(unique_labels) == 0:  # 所有客户端都被标记为噪声
            selected_clients = client_ids
        else:
            majority_cluster = unique_labels[np.argmax(counts)]
            selected_clients = [client_ids[i] for i, label in enumerate(cluster_labels)
                                if label == majority_cluster]

        # 打印筛选出的客户端
        print(f"筛选出的客户端: {selected_clients}")
        backdoor_in_cluster = [client for client in selected_clients if client in self.params.backdoor_clients]
        print(backdoor_in_cluster)
        # 4. 聚合最大簇的更新（恢复你最初的实现方式）
        weight_accumulator = torch.zeros_like(global_model_param)
        for client_id in selected_clients:
            update = clients_update[client_id]
            # 按照你最初的实现方式展平所有参数
            loaded_params = torch.cat([param.view(-1) for param in update.values()])
            weight_accumulator.add_(loaded_params)  # 直接累加

        # 5. 计算等权重平均
        global_model_param = global_model_param + weight_accumulator / len(selected_clients)  # 等权重平均

        return global_model_param

    def filtering(self, temp):
        """与原始代码完全一致的低通滤波方法"""
        filtered_length = len(temp) // 2
        F = temp[:filtered_length]
        return F

    def cluster(self, F_list):
        """与原始代码完全一致的聚类方法"""
        # 将列表转换为2D数组
        features = np.array(F_list,dtype=np.float64)

        # 计算余弦距离矩阵
        distances_matrix = pairwise_distances(features, metric='cosine')

        cluster = hdbscan.HDBSCAN()
        cluster_labels = cluster.fit_predict(distances_matrix)  # 直接返回的是 ndarray
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