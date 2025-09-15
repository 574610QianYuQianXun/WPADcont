from collections import defaultdict
from copy import deepcopy

import hdbscan
import numpy as np
import torch
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
from sklearn.covariance import EllipticEnvelope
from sklearn.experimental import enable_iterative_imputer  # Enable the feature
from sklearn.impute import IterativeImputer  # Now import it
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from tqdm import tqdm

from utils import utils


class FLShield:
    def __init__(self, params, global_model, loss_func):
        self.params = params
        self.global_model = global_model
        self.tau = 0.75  # 双射生成中的贡献比例
        self.n_validators = 10  # 验证者数量
        self.n1 = 1  # 每类最小样本数
        self.n2 = 20  # 每类最大样本数
        self.loss_func=loss_func

        self.layer_name = 'fc2' if 'MNIST' in self.params.task else 'fc'

    def aggregate(self, clients, clients_update):
        """主聚合流程"""
        # 步骤1：生成代表模型
        if self.params.flshield_mode == 'bijective':
            repr_models, client_mapping = self.bijective_representatives(clients_update)
        else:  # 'cluster'
            repr_models, client_mapping = self.cluster_representatives(clients_update)

        # 步骤2：模型验证
        validator_reports = self.model_validation(repr_models, clients)

        # 步骤3：过滤异常验证报告
        # filtered_reports = self.filter_reports(validator_reports)
        filtered_reports=validator_reports

        # 步骤4：选择前50%代表模型
        selected_reprs = self.select_top_representatives(filtered_reports, client_mapping, clients_update)

        # 步骤5：剪裁和聚合
        # global_update_params = self.clip_and_aggregate(selected_reprs)
        global_update_params = self.fedavg_aggregate(selected_reprs)
        # global_update_params = self.clip_and_aggregate_pro(selected_reprs)


        return global_update_params

    def bijective_representatives(self, clients_update):
        """双射代表模型生成（算法13）"""
        # 代表性模型
        repr_models = {}
        # 模型映射
        client_mapping = {}
        # 全局模型参数，用于最后聚合
        global_vec = utils.model_to_vector(self.global_model, self.params)

        for client_id, client_update in clients_update.items():
            # 基础更新
            base_vec = torch.cat([param.flatten() for param in client_update.values()])
            # 用于计算相似度的数组
            base_sim_vec = torch.cat([param.flatten() for name, param in client_update.items() if name.endswith(f"{self.layer_name}.bias") or name.endswith(f"{self.layer_name}.weight")])
            # 算法的分子和分母
            sibling_contribution = 0
            total_weight = 0

            # 计算相似度并累积贡献
            for other_id, other_update in clients_update.items():
                # 聚合其他更新
                if other_id == client_id: continue
                other_vec = torch.cat([param.flatten() for param in other_update.values()])
                other_sim_vec = torch.cat([param.flatten() for name, param in other_update.items() if self.layer_name in name])
                sim = self.cosine_sim_with_relu(base_sim_vec, other_sim_vec)
                scaled_vec = sim * other_vec * (base_vec.norm() / other_vec.norm())
                sibling_contribution += scaled_vec
                total_weight += sim * (base_vec.norm() / other_vec.norm())

            # 生成代表模型
            if total_weight > 0:
                sibling_contribution /= total_weight
            repr_vec = global_vec + (1 - self.tau) * base_vec + self.tau * sibling_contribution
            repr_models[client_id] = repr_vec
            client_mapping[client_id] = [client_id]  # 每个代表模型对应原始客户端

        return repr_models, client_mapping

    # def cluster_representatives(self, clients_update):
    #     """聚类代表模型生成（算法12）"""
    #     global_vec = utils.model_to_vector(self.global_model, self.args)
    #
    #     model_vectors = [torch.cat([param.flatten() for param in client_update.values()]).cpu().numpy()
    #                      for client_update in clients_update.values()]
    #
    #     model_sim_vectors = [torch.cat([param.flatten() for name,param in client_update.items()
    #                                     if self.layer_name in name]).cpu().numpy()
    #                      for client_update in clients_update.values()]
    #
    #     k = self.find_optimal_clusters(model_sim_vectors)
    #     # K-means聚类
    #     kmeans = cluster.KMeans(n_clusters=k, random_state=0).fit(model_sim_vectors)
    #     client_mapping = {i: [] for i in range(k)}
    #     # 构建聚类映射
    #     for client_id, label in enumerate(kmeans.labels_):
    #         client_mapping[label].append(client_id)
    #
    #     # 生成聚类代表模型
    #     repr_models = {}
    #     for cluster_id, members in client_mapping.items():
    #         cluster_vectors = [model_vectors[client_id] for client_id in members]
    #         repr_vec = np.mean(cluster_vectors, axis=0)
    #         repr_models[cluster_id] = global_vec + torch.from_numpy(repr_vec).to(self.args.device)
    #
    #     return repr_models, client_mapping
    #
    # def find_optimal_clusters(self, vectors):
    #     """使用轮廓系数寻找最佳聚类数"""
    #     best_score = -1
    #     best_k = 2
    #
    #     max_clusters = min(len(vectors) // 2, 10)  # 限制最大聚类数，避免过拟合
    #
    #     for k in range(2, max_clusters + 1):
    #         try:
    #             kmeans = cluster.KMeans(n_clusters=k, init='k-means++', random_state=0, n_init=10).fit(vectors)
    #             score = silhouette_score(vectors, kmeans.labels_)
    #             if score > best_score:
    #                 best_score = score
    #                 best_k = k
    #         except ValueError as e:
    #             print(f"跳过 k={k}，错误: {e}")
    #             continue
    #
    #     return best_k

    def cluster_representatives(self, clients_update, clustering_method='KMeans'):
        """聚类代表模型生成（算法12）"""
        global_vec = utils.model_to_vector(self.global_model, self.params)
        # 参数拉平
        model_vectors = [np.concatenate([param.cpu().numpy().flatten()
                                         for param in client_update.values()])
                         for client_update in clients_update.values()]
        # 生成用于相似性计算的子网络特征（特定层）
        model_sim_vectors = [np.concatenate([param.cpu().numpy().flatten()
                                             for name, param in client_update.items()
                                             if name.endswith(f"{self.layer_name}.bias") or name.endswith(f"{self.layer_name}.weight")])
                             for client_update in clients_update.values()]
        # 寻找最佳聚类数
        k = FLShield.find_optimal_clusters(model_sim_vectors, clustering_method)
        # k = 2
        # 执行聚类
        cluster_labels = FLShield.perform_clustering(
            model_sim_vectors,
            k,
            clustering_method
        )
        # 构建客户端分组映射
        client_mapping = defaultdict(list)
        for client_id, label in enumerate(cluster_labels):
            client_mapping[label].append(client_id)

        # 处理噪声数据（hdbscan的特殊情况）
        # client_mapping = self.handle_noise_clients(client_mapping)

        # 生成聚类代表模型
        repr_models = {}
        for cluster_id, members in client_mapping.items():
            cluster_vectors = [model_vectors[client_id] for client_id in members]
            repr_vec = np.mean(cluster_vectors, axis=0)
            repr_models[cluster_id] = global_vec + torch.from_numpy(repr_vec).to(self.params.device)

        return repr_models, client_mapping

    @staticmethod
    def find_optimal_clusters(vectors, clustering_method):
        """使用轮廓系数寻找最佳聚类数"""
        best_score = -1
        best_k = 2
        max_k = min(len(vectors), 15)  # 限制最大尝试聚类数
        # 计算相似性矩阵
        coses = cosine_distances(vectors)
        coses = np.array(coses)
        np.fill_diagonal(coses, 0)
        for k in range(2, max_k + 1):
            try:
                # 执行临时聚类
                temp_labels = FLShield.perform_clustering(
                    vectors,
                    k,
                    clustering_method,
                    coses
                )
                # 计算轮廓系数
                if len(set(temp_labels)) < 2:
                    continue  # 无法计算单一簇的分数

                # score = silhouette_score(
                #     coses if clustering_method != 'KMeans' else vectors,
                #     temp_labels,
                #     metric='precomputed' if clustering_method != 'KMeans' else 'cosine'
                # )
                score = silhouette_score(
                    coses,
                    temp_labels,
                    metric='precomputed'
                )
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"Clustering failed for k={k}: {str(e)}")
                continue

        return best_k if best_score > 0 else 2  # 保证最小聚类数为2

    @staticmethod
    def perform_clustering(vectors, k, method, precomputed=None):
        """执行具体聚类算法"""
        # if method == 'KMeans':
        #     return cluster.KMeans(n_clusters=k).fit(vectors).labels_
        if precomputed is None:
            precomputed = cosine_distances(vectors)
            precomputed = np.array(precomputed)
            np.fill_diagonal(precomputed, 0)
        # 效果挺好
        if method == 'KMeans':
            return cluster.KMeans(n_clusters=k).fit(precomputed).labels_
        # 层次聚类 效果不错
        if method == 'Agglomerative':
            return cluster.AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed',
                linkage='complete'
            ).fit(precomputed).labels_
        # 谱聚类 差劲
        if method == 'Spectral':
            return cluster.SpectralClustering(
                n_clusters=k,
                affinity='precomputed'
            ).fit(precomputed).labels_
        # hdbscan聚类 还行可接受
        if method == 'hdbscan':
            # 确保数据类型为 float64
            precomputed = precomputed.astype(np.float64)
            return hdbscan.HDBSCAN(
                min_cluster_size=max(2, len(vectors) // 4),
                min_samples=1,
                allow_single_cluster=True,
                metric='precomputed'
            ).fit_predict(precomputed)
        raise ValueError(f"Unsupported clustering method: {method}")

    def handle_noise_clients(self, client_mapping):
        """处理hdbscan的噪声数据（标签-1）"""
        if -1 in client_mapping:
            noise_clients = client_mapping.pop(-1)
            if client_mapping:
                # 将噪声客户端分配到最近的非空簇
                largest_cluster = max(client_mapping.values(), key=len)
                largest_cluster.extend(noise_clients)
            else:
                # 所有都是噪声时创建新簇
                client_mapping[0] = noise_clients
        return client_mapping

    def model_validation(self, repr_models, clients):
        """模型验证（算法14）
        :param repr_models: dict，代表性模型
        :param clients: list，所有客户端对象
        :return: reports，存储每个代表性模型在不同验证者上的 LIPC 向量
        """
        validator_ids = np.random.choice(list(range(len(clients))), self.n_validators, replace=False)
        print(validator_ids)
        reports = {}

        for repr_id, repr_vec in tqdm(repr_models.items()):
            model_reports = []
            for val_id in validator_ids:
                val_client = clients[val_id]  # 获取 Client 实例
                val_loader = val_client.train_loader  # 获取 DataLoader 作为验证数据

                # 计算 LIPC 分数
                lipc = self.calculate_lipc(repr_vec, val_loader)
                model_reports.append(lipc)

            reports[repr_id] = np.array(model_reports)

        return reports

    def calculate_lipc(self, repr_vec, val_loader):
        """
        LIPC 计算（算法15）
        :param repr_vec: 代表性模型向量
        :param val_loader: DataLoader，提供 (inputs, labels) 作为验证数据
        :return: LIPC 向量（每个类别一个值）
        """
        # 获取全局模型的向量表示
        global_vec = utils.model_to_vector(self.global_model, self.params)
        lipc_vector = []
        num_classes = self.params.num_classes

        # 针对每个类别计算 LIPC
        for class_id in range(num_classes):
            class_samples = []
            # 遍历 DataLoader 获取当前类别的所有样本
            for inputs, labels in val_loader:
                # 生成类别掩码，注意 mask.sum() 为 tensor 时需要 .item()
                mask = (labels == class_id)
                if mask.sum().item() > 0:
                    # zip() 后得到 (input, label) 对，并加入当前类别样本列表
                    class_samples.extend(zip(inputs[mask], labels[mask]))
            # 如果当前类别样本不足阈值 n1，则该类别返回 NaN
            if len(class_samples) < self.n1:
                lipc_vector.append(np.nan)
                continue
            # 取前 n2 个样本（或所有样本，若数量较少）
            sampled = class_samples[:min(self.n2, len(class_samples))]
            # 批量计算全局模型和代表性模型在这些样本上的平均损失
            global_loss = self.compute_loss(global_vec, sampled)
            repr_loss = self.compute_loss(repr_vec, sampled)
            # LIPC 定义为全局模型损失与代表性模型损失的差值
            lipc_vector.append(global_loss - repr_loss)
        return np.array(lipc_vector)

    # def compute_loss(self, model_vec, samples):
    #     """计算模型在样本上的损失"""
    #     temp_model=deepcopy(self.global_model)
    #     utils.vector_to_model(temp_model,model_vec, self.args)
    #     temp_model.eval()
    #     total_loss = 0
    #     for data, target in samples:
    #         data=data.unsqueeze(0)
    #         target=target.unsqueeze(0)
    #         data, target = data.to(self.args.device), target.to(self.args.device)
    #         output = temp_model(data)
    #         # loss = torch.nn.functional.cross_entropy(output, target)
    #         loss= self.loss_func(output, target)
    #         total_loss += loss.item()
    #     return total_loss / len(samples)

    def compute_loss(self, model_vec, samples):
        """
        计算模型在样本上的损失（改进：批量处理）
        :param model_vec: 模型向量表示（global 或 representative）
        :param samples: list，每个元素为 (data, target)
        :return: 平均损失
        """
        # 复制全局模型并更新为给定模型向量
        temp_model = deepcopy(self.global_model)
        utils.vector_to_model(temp_model, model_vec, self.params)
        temp_model.eval()  # 设置为评估模式

        # 将单个样本整合为批量（提升计算效率）
        data_list = []
        target_list = []
        for data, target in samples:
            data_list.append(data.unsqueeze(0))  # 增加 batch 维度
            target_list.append(target.unsqueeze(0))
        # 拼接成批量张量，并转移到指定设备
        data_batch = torch.cat(data_list, dim=0).to(self.params.device)
        target_batch = torch.cat(target_list, dim=0).to(self.params.device)

        with torch.no_grad():
            _,output = temp_model(data_batch)
            # 这里要求 self.loss_func 的 reduction 设置为 'sum'，否则需调整（例如 loss.item() * batch_size）
            loss = self.loss_func(output, target_batch)
            # 如果 loss 不是标量，则对所有样本求和
            total_loss = loss.item() if isinstance(loss.item(), float) else loss.sum().item()
        # 返回平均损失
        return total_loss / data_batch.size(0)

    # def filter_reports(self, reports):
    #     """过滤异常验证报告（基于论文算法）"""
    #     # 重组数据为 (n_validators, n_models * num_classes)
    #     validator_reports = []
    #     n_validators = next(iter(reports.values())).shape[0]
    #     # num_classes = next(iter(reports.values())).shape[1]
    #
    #     for val_id in range(n_validators):
    #         val_data = []
    #         for repr_id in reports.keys():
    #             val_data.extend(reports[repr_id][val_id, :].tolist())
    #         validator_reports.append(val_data)
    #
    #     validator_reports = np.array(validator_reports)  # 形状: (n_validators, n_models*num_classes)
    #
    #     # 步骤1：填充缺失值
    #     imp = IterativeImputer(max_iter=20, random_state=0)
    #     # imp = IterativeImputer()
    #     filled = imp.fit_transform(validator_reports)
    #
    #     # 步骤2：检测恶意验证者（论文中的椭圆包络检测）
    #     # detector = EllipticEnvelope(
    #     #     contamination=0.5
    #     # )
    #     detector = EllipticEnvelope(
    #         contamination=0.5,
    #         support_fraction=0.9
    #     )
    #     is_inlier = detector.fit_predict(filled)  # 1=正常，-1=异常
    #
    #     # 步骤3：过滤异常验证者的所有报告
    #     filtered = {}
    #     for repr_id in reports.keys():
    #         # 保留正常验证者的数据（按列过滤）
    #         filtered[repr_id] = reports[repr_id][is_inlier == 1, :]
    #     return filtered

    def filter_reports(self, reports):
        """过滤异常验证报告（基于论文算法）

        输入的 reports 为一个字典，键为代表模型 id，值为形状 (n_validators, num_classes) 的验证报告矩阵。
        函数首先将所有报告拼接成一个 (n_validators, n_models*num_classes) 的数组，
        对其中缺失值进行填充，然后利用椭圆包络检测过滤异常验证者，
        最后利用填充后的数据重构各代表模型的报告矩阵返回。
        """
        # 重组数据为 (n_validators, n_models * num_classes)
        validator_reports = []
        keys = list(reports.keys())  # 保持代表模型顺序
        n_validators = next(iter(reports.values())).shape[0]
        num_classes = next(iter(reports.values())).shape[1]

        for val_id in range(n_validators):
            val_data = []
            for repr_id in keys:
                # 将每个代表模型对应的该验证者报告拼接
                val_data.extend(reports[repr_id][val_id, :].tolist())
            validator_reports.append(val_data)

        validator_reports = np.array(validator_reports)  # 形状: (n_validators, n_models*num_classes)

        # 步骤1：填充缺失值（使用 IterativeImputer）
        imp = IterativeImputer(max_iter=20, random_state=0)
        filled = imp.fit_transform(validator_reports)  # filled 为 (n_validators, n_models*num_classes)

        # 步骤2：检测异常验证者（使用椭圆包络检测）
        detector = EllipticEnvelope(
            contamination=0.5,
            support_fraction=0.9
        )
        is_inlier = detector.fit_predict(filled)  # 1=正常，-1=异常

        # 步骤3：根据 is_inlier 筛选出正常验证者对应的填充数据，
        # 并根据原来的代表模型列数还原成字典形式
        filtered = {}
        # 对于每个代表模型，其报告对应 filled 数组的连续一段列
        for i, repr_id in enumerate(keys):
            start_col = i * num_classes
            end_col = (i + 1) * num_classes
            # 提取所有正常验证者（is_inlier == 1）的数据对应当前代表模型的部分
            filtered[repr_id] = filled[is_inlier == 1, start_col:end_col]

        return filtered

    def filter_representatives_by_lipc(self, lipc_scores, clustering_method='KMeans'):
        """
        对代表性模型的 LIPC 分数进行聚类，剔除后门模型（LIPC 分数较低的模型）。
        :param lipc_scores: dict，键为代表性模型 id，值为对应的 LIPC 分数
        :param clustering_method: 聚类方法（目前仅支持 'KMeans'）
        :return: filtered_ids，过滤后的代表性模型 id 列表（剔除了后门模型）
        """
        # 将 LIPC 分数转换为二维数组
        ids = list(lipc_scores.keys())
        scores = np.array([lipc_scores[rid] for rid in ids]).reshape(-1, 1)
        if clustering_method == 'KMeans':
            kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            # 后门模型所在聚类的中心更低
            low_cluster = np.argmin(centers)
            filtered_ids = [ids[i] for i in range(len(ids)) if labels[i] != low_cluster]
            # logger.info(f"Filtered representative model ids (removed backdoor): {filtered_ids}")
            return filtered_ids
        else:
            raise ValueError("Unsupported clustering method for filtering representatives")

    def select_top_representatives(self, filtered_reports, client_mapping, clients_update):
        """选择前50%的代表模型"""
        scores = {}
        for repr_id, reports in filtered_reports.items():
            avg_scores = np.nanmean(reports, axis=0)
            scores[repr_id] = np.min(avg_scores)

        # 利用 LIPC 分数对代表模型进行聚类过滤，剔除后门模型
        filtered_ids = self.filter_representatives_by_lipc(scores, clustering_method='KMeans')

        # 按得分排序
        # sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        # selected_ids = sorted_ids[:len(sorted_ids) // 2]

        # 收集对应的原始客户端更新
        selected_updates = []
        for repr_id in filtered_ids:
            original_clients = client_mapping[repr_id]
            for c in original_clients:
                # 将客户端 c 的模型参数展开成一个一维向量
                # update_vector = torch.cat([param.flatten() for param in clients_update[c].values()])
                # selected_updates.append(update_vector)
                update_dict = {name: param.flatten()
                               for name, param in clients_update[c].items()}
                selected_updates.append(update_dict)
        return selected_updates

    def fedavg_aggregate(self, updates):
        """使用 FedAvg 进行聚合，不进行范数剪裁"""
        # 获取全局模型参数向量
        global_model_param = utils.model_to_vector(self.global_model, self.params).detach()
        # 初始化累加器（形状与更新向量相同）
        weight_accumulator = torch.zeros_like(global_model_param)
        # 遍历所有更新并累加
        for update_params in updates:
            loaded_params = torch.cat([param.flatten() for param in update_params.values()])
            weight_accumulator.add_(loaded_params)
        # 等权重平均并更新全局模型参数
        global_model_param = global_model_param + weight_accumulator / len(updates)
        return global_model_param

    def clip_and_aggregate(self, updates):
        """范数剪裁和聚合"""
        global_model_param = utils.model_to_vector(self.global_model, self.params).detach()
        # 初始化累加器（注意：累加器的形状与更新向量相同）
        weight_accumulator = torch.zeros_like(global_model_param)
        # 使用所有更新的范数中位数作为剪裁阈值
        threshold = np.median([utils.get_update_norm(updates[i]) for i in range(len(updates))])
        # 遍历所有更新
        for i in range(len(updates)):
            update_params=updates[i]
            norm_val = utils.get_update_norm(update_params)
            loaded_params = torch.cat([param.flatten() for param in update_params.values()])
            if norm_val > threshold:
                clip_coef = threshold / norm_val
                # in-place 范数缩放
                loaded_params.mul_(clip_coef)
            weight_accumulator.add_(loaded_params)


        global_model_param = global_model_param + weight_accumulator / len(updates)  # 等权重平均
        return global_model_param

    def clip_and_aggregate_pro(self, updates):
        """范数剪裁和聚合（动态阈值 + 加权聚合 + 可调超参数）

        改进点：
        1. 动态阈值：利用 IQR 筛选有效更新范数，并使用 clip_relax_factor 放宽阈值，
           防止过于严格的剪裁导致有效更新信息丢失。
        2. 加权聚合：在计算权重时引入 weight_exponent，对有效范数做非线性平滑，
           降低因范数差异过大引起的权重不均。
        3. 防止除零错误：引入 epsilon。
        """
        global_model_param = utils.model_to_vector(self.global_model, self.params).detach()
        weight_accumulator = torch.zeros_like(global_model_param)

        # --------------------- 动态阈值计算 ---------------------
        # 计算所有更新的范数
        norms = [utils.get_update_norm(update) for update in updates]

        # 利用 IQR 过滤异常值（避免恶意客户端干扰阈值）
        q75 = np.percentile(norms, 75)  # 第75百分位数
        q25 = np.percentile(norms, 25)  # 第25百分位数
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr  # 异常值上限
        valid_norms = [n for n in norms if n <= upper_bound]

        # 使用有效范数的均值作为基础阈值，并使用放宽因子 clip_relax_factor 调整
        clip_relax_factor = 1.0  # 可调参数，设置为大于1则阈值放宽，减少剪裁强度
        threshold = clip_relax_factor * (np.mean(valid_norms) if len(valid_norms) > 0 else np.median(norms))

        # --------------------- 加权聚合 ---------------------
        total_weight = 0.0
        weight_exponent = 0.5  # 可调参数，取值在 (0, 1]，1 表示不平滑，<1 可平滑权重差异
        epsilon = 1e-6  # 防止除零

        for i, update_params in enumerate(updates):
            norm_val = norms[i]
            loaded_params = torch.cat([param.flatten() for param in update_params.values()])

            # 裁剪逻辑：如果更新范数超过阈值，则缩放到阈值
            if norm_val > threshold:
                clip_coef = threshold / (norm_val + epsilon)
                loaded_params.mul_(clip_coef)
                effective_norm = threshold  # 裁剪后视作阈值
            else:
                effective_norm = norm_val

            # 计算权重：这里使用有效范数的非线性变换（例如平方根），使得权重差异不至于过大
            weight = effective_norm ** weight_exponent
            weight_accumulator.add_(loaded_params * weight)
            total_weight += weight

        # --------------------- 避免除零错误 ---------------------
        if total_weight < 1e-10:
            return global_model_param  # 防御性返回

        # 加权平均更新
        global_model_param = global_model_param + weight_accumulator / total_weight
        return global_model_param

    def cosine_sim_with_relu(self, v1, v2):
        """带ReLU的余弦相似度"""
        cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0)
        return torch.relu(cos_sim)