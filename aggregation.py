import copy
import random
from copy import deepcopy

import hdbscan
import math
import numpy as np
import sklearn.metrics.pairwise as smp
import logging
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from defenses.badExpert import BadExpert
from defenses.flshield import FLShield
from defenses.fndShield import FndShield
from defenses.freqfed import FreqFed
from utils import utils
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

def Aggregation(params, helper, global_model, clients_model, clients_update, clients_his_update, clients, loss_func):
    update_model_params = None
    clients_param = {}
    clients_dict_param = []
    global_model_param = utils.model_to_vector(global_model, params).detach()
    for client_id, model_param in clients_model.items():
        # if client_id in params.backdoor_clients:continue
        # if client_id in [3]:continue
        # else:
        clients_dict_param.append(clients_model[client_id].state_dict())
        clients_param[client_id]=utils.model_to_vector(model_param, params)
    if params.agg == 'FedAvg':

        # badExpert=BadExpert(params,global_model,helper.test_dataset)
        # badExpert.train_expert(clients_model)
        # badExpert.cluster_expert()
        # bad_params={}
        # for client_id, model_param in badExpert.experts.items():
        #     bad_params[client_id] = utils.model_to_vector(model_param, params)
        # update_model_params = Agg_avg(bad_params)

        # 正常聚合
        update_model_params = Agg_avg(clients_param)

    elif params.agg == 'FoolsGold':
        update_model_params = Agg_foolsgold(params, global_model_param, clients_param, clients_update, clients_his_update)
    elif params.agg == 'DeepSight':
        update_model_params = Agg_deepsight(params, global_model, global_model_param, clients_model, clients_update)
    elif params.agg == 'FLShield':
        aggregator = FLShield(params, global_model, loss_func)
        update_model_params = aggregator.aggregate(clients, clients_update)
        # update_model_params = aggregator.clip_and_aggregate(clients_update)
    elif params.agg == 'FreqFed':
        aggregator = FreqFed(params, global_model)
        update_model_params = aggregator.aggregate(clients_update,global_model_param)

    elif params.agg == 'FndShield':
        # 模拟检测到的恶意客户端
        # 从 params.backdoor_clients 中随机选择十个恶意客户端
        malicious_id = random.sample(params.backdoor_clients, min(1, len(params.backdoor_clients)))

        # 剩余的客户端为 benign_id
        benign_id = [client_id for client_id in clients_param.keys() if client_id not in malicious_id]

        # 从 clients_param 中获取正常客户端（benign）的聚合模型
        benign_params = Agg_avg_pro(clients_param, benign_id)

        # 从 clients_param 中获取恶意客户端（malicious）的聚合模型
        malicious_params = Agg_avg_pro(clients_param, malicious_id)

        # 为 FndShield 初始化所需的参数
        # 这里需要将 benign_params 和 malicious_params 转换为模型（例如通过 utils.vector_to_model）
        # 请确保 benign_model 和 malicious_model 被正确创建
        benign_model=deepcopy(global_model)
        malicious_model=deepcopy(global_model)
        utils.vector_to_model(benign_model,benign_params, params)
        utils.vector_to_model(malicious_model,malicious_params, params)

        # 执行聚合
        # 创建测试数据加载器
        test_loader = DataLoader(helper.test_dataset, batch_size=params.bs, shuffle=False)
        fndshield = FndShield(params, benign_model, malicious_model, test_loader)
        fndshield.train()
        helper.global_model=fndshield.global_model
        return

    elif params.agg == 'Rflbat':
        m_client = Rflbat(clients_dict_param, params)
        print(sorted(m_client))

        # 使用列表推导式找到在m_clients中的元素
        in_m_clients = [element for element in m_client if element in params.backdoor_clients]
        # 使用列表推导式找到不在m_clients中的元素
        not_in_m_clients = [element for element in m_client if element not in params.backdoor_clients]
        # 打印结果
        print("防御检测到的恶意客户端中属于m_clients的有：", in_m_clients)
        print("防御检测到的恶意客户端中不属于m_clients的有：", not_in_m_clients)
        # 打印数量
        print("属于m_clients的数量：", len(in_m_clients))
        print("不属于m_clients的数量：", len(not_in_m_clients))

        clients_dict_param = Remove_clients(clients_dict_param, m_client)
        # 展平每个客户端模型
        flatten_clients_param = {}
        for cid, param_dict in enumerate(clients_dict_param):
            vector = torch.cat([p.view(-1) for p in param_dict.values()])
            flatten_clients_param[cid] = vector
        # 聚合展平向量
        update_model_params = Agg_avg(flatten_clients_param)
    elif params.agg == 'multi_krum':
        m_client = Multi_krum(clients_dict_param,global_model,params.backdoor_clients, params)
        print(sorted(m_client))
        # 使用列表推导式找到在m_clients中的元素
        in_m_clients = [element for element in m_client if element in params.backdoor_clients]
        # 使用列表推导式找到不在m_clients中的元素
        not_in_m_clients = [element for element in m_client if element not in params.backdoor_clients]
        # 打印结果
        print("防御检测到的恶意客户端中属于m_clients的有：", in_m_clients)
        print("防御检测到的恶意客户端中不属于m_clients的有：", not_in_m_clients)
        # 打印数量
        print("属于m_clients的数量：", len(in_m_clients))
        print("不属于m_clients的数量：", len(not_in_m_clients))

        clients_dict_param = Remove_clients(clients_dict_param, m_client)
        # 展平每个客户端模型
        flatten_clients_param = {}
        for cid, param_dict in enumerate(clients_dict_param):
            vector = torch.cat([p.view(-1) for p in param_dict.values()])
            flatten_clients_param[cid] = vector
        # 聚合展平向量
        update_model_params = Agg_avg(flatten_clients_param)

    utils.vector_to_model(global_model, update_model_params, params)

# FedAvg聚合方式
def Agg_avg(clients_param):
    sum_model = 0
    for _id, model_param in clients_param.items():
        sum_model += model_param
    # print(sum_model / len(clients_param))
    return sum_model / len(clients_param)

def Agg_avg_pro(clients_param, selected_clients):
    """
    聚合指定客户端模型的平均参数
    Args:
        clients_param: 所有客户端模型的参数字典
        selected_clients: 需要聚合的客户端编号列表
    """
    sum_model = 0
    num_selected_clients = 0

    for client_id, model_param in clients_param.items():
        if client_id in selected_clients:
            sum_model += model_param
            num_selected_clients += 1

    if num_selected_clients > 0:
        return sum_model / num_selected_clients
    else:
        raise ValueError("No selected clients available for aggregation.")

# FoolsGold聚合方法
def Agg_foolsgold(params, global_model_param, clients_param, clients_update, clients_his_update):
    """
    基于 PyTorch 的 FoolsGold 聚合算法
    Args:
        clients_param (dict): 客户端的梯度更新参数，每个客户端为一个键值对 {client_id: model_param}.
    Returns:
        torch.Tensor: 聚合后的全局更新参数.
    """
    print("我已进入'FoolsGold'")
    layer_name = 'fc2' if 'MNIST' in params.task else 'fc'
    epsilon = 1e-5

    foolsgold_his_update={}
    foolsgold_update={}
    for client_id, updates in clients_his_update.items():
        # 将所有参数展开后合并成一个一维向量
        foolsgold_his_update[client_id] = torch.cat([param.flatten() for name, param in updates.items() if layer_name in name])

    for client_id, updates in clients_update.items():
        # 将所有参数展开后合并成一个一维向量
        foolsgold_update[client_id] = torch.cat([param.flatten() for param in updates.values()])

    foolsgold_his_update = {k: v.cpu().numpy() for k, v in foolsgold_his_update.items()}
    his_ids = list(foolsgold_his_update.keys())  # 获取所有客户端 ID
    his_vectors = np.array([foolsgold_his_update[client_id] for client_id in his_ids])  # 直接转换为矩阵
    # 计算余弦相似度矩阵
    cs = smp.cosine_similarity(his_vectors)-np.eye(len(his_vectors))
    maxcs = np.max(cs, axis=1) + epsilon
    for i in range(len(his_vectors)):
        for j in range(len(his_vectors)):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    # Pardoning
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    # print(cs)
    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    # logger.info(f"FoolsGold: Accumulation with lr {wv}")
    # logger.info(f"FoolsGold: update cs is {cs}")
    # print(cs)

    # 初始化累加器
    weight_accumulator = torch.zeros_like(global_model_param)

    # 遍历所有客户端，进行加权累加
    for client_id, update in foolsgold_update.items():
        # Ensure that both wv[client_id] and update are PyTorch tensors
        if isinstance(update, np.ndarray):
            update = torch.from_numpy(update).to(params.device)
        if isinstance(wv[client_id], np.ndarray):
            wv[client_id] = torch.from_numpy(wv[client_id]).to(params.device)
        weight_accumulator.add_((wv[client_id] * update))

    # print(f"wv:{wv}")

    global_model_param = global_model_param + weight_accumulator / len(foolsgold_update)  # 等权重平均
    # global_model_param = global_model_param + weight_accumulator / sum(wv)  # 等权重平均
    return global_model_param


    # print(weight_accumulator)

    # # 获取所有客户端的梯度,并且转成一个temsor张量
    # client_ids = list(clients_param.keys())
    # client_grads = torch.stack([clients_param[client_id].flatten() for client_id in client_ids])  # 维度: (n_clients, gradient_size)
    # # print("Client 梯度:\n", client_grads)
    #
    # # client_grads=clients_param
    #
    # n_clients = client_grads.size(0)
    #
    # # 计算每行的 L2 范数
    # row_norms = torch.norm(client_grads, p=2, dim=1, keepdim=True)  # (n x 1)
    # row_norms=torch.clamp(row_norms,min=1e-5)
    #
    # # 对矩阵的每一行进行归一化
    # normalized_ = client_grads / row_norms  # (n x n)
    # cosine_similarity_matrix = torch.mm(normalized_, normalized_.T)  # 计算余弦相似度矩阵
    # cosine_similarity_matrix.fill_diagonal_(0)  # 清除对角线上的值，设置为0
    # # print("Client 相似度:\n", cosine_similarity_matrix)
    #
    # # 计算每个客户端的最大余弦相似度
    # maxcs = torch.max(cosine_similarity_matrix, dim=1).values + epsilon
    # # print("Max cosine similarity per client:", maxcs)
    #
    # # Pardoning 步骤
    # for i in range(n_clients):
    #     for j in range(n_clients):
    #         if i == j:
    #             continue
    #         if maxcs[i] < maxcs[j]:
    #             cosine_similarity_matrix[i, j] *= (maxcs[i] / maxcs[j]) ** 2
    #
    # # 计算权重向量
    # wv = 1 - torch.max(cosine_similarity_matrix, dim=1).values
    # # wv = torch.clamp(wv, min=0, max=1)  # 限制 wv 在 [0, 1] 范围内
    # wv = torch.clamp(wv, min=epsilon, max=1 - epsilon)  # 将值限制在 [0, 1] 范围内
    #
    # wv = wv / torch.max(wv)  # 归一化
    # wv[wv == 1] = .99  # 防止权重为1的情况
    # # 对权重做进一步平滑处理，比如通过非线性函数（如softmax）进行归一化，以减少恶意客户端的显著权重：
    # # wv = torch.softmax(wv, dim=0)
    #
    # # Logit 函数
    # wv = (torch.log(wv / (1 - wv) + epsilon ) + 0.5)
    # wv = torch.where(torch.isinf(wv), torch.tensor(1.0, device=wv.device), wv)# 将 inf 设置为 1
    # wv = torch.clamp(wv, min=epsilon, max=1-epsilon)  # 将值限制在 [0, 1] 范围内
    # # wv = torch.clamp(wv, min=0, max=1)  # 限制 wv 在 [0, 1] 范围内
    # # wv=1-wv
    # # print("wv:\n", wv)
    #
    # # 根据权重对梯度加权
    # aggregated_grad = torch.zeros_like(client_grads[0])
    #
    # for _id, model_param in clients_param.items():
    #     aggregated_grad += wv[_id] * model_param
    #
    # # aggregated_grad += global_param
    # aggregated_grad = aggregated_grad / torch.sum(wv)
    # # print("aggregated:\n",aggregated_grad)
    # return aggregated_grad
    # return wv

# DeepSight聚合方法
def Agg_deepsight(params, global_model,global_model_param,clients_model, clients_update):
    num_seeds: int = 1
    num_samples: int = 5000
    tau: float = 1 / 3
    num_channel = 3
    if 'MNIST' in params.task:
        dim = 28
        num_channel = 1
    elif 'CIFAR10' in params.task:
        dim = 32
    else:
        dim = 224
    layer_name = 'fc2' if 'MNIST' in params.task else 'fc'
    num_classes = 200 if 'Imagenet' in params.task else 10

    # Threshold exceedings and NEUPs
    TEs, NEUPs, ed = [], [], []
    for i in range(params.clients):
        # file_name = f'{params.folder_path}/saved_updates/update_{i}.pth'
        loaded_params = clients_update[i]
        ed = np.append(ed, utils.get_update_norm(loaded_params))
        UPs = (abs(loaded_params[f'{layer_name}.bias'].cpu().numpy()) +
               np.sum(abs(loaded_params[f'{layer_name}.weight'].cpu().numpy()), axis=1))
        NEUP = UPs ** 2 / np.sum(UPs ** 2)
        TE = 0
        for j in NEUP:
            if j >= (1 / num_classes) * np.max(NEUP):
                TE += 1
        NEUPs.append(NEUP)  # 每个NEUP是长度num_classes的数组
        TEs.append(TE)
    NEUPs = np.array(NEUPs)  # 形状 (N, num_classes)

    print(f'Deepsight: Threshold Exceedings {TEs}')

    labels = []
    for i in TEs:
        if i >= np.median(TEs) / 2:
            labels.append(False)
        else:
            labels.append(True)

    print(f'Deepsight: labels {labels}')
    # ddif
    DDifs = []
    for i, seed in tqdm(enumerate(range(num_seeds))):
        torch.manual_seed(seed)
        dataset = NoiseDataset([num_channel, dim, dim], num_samples)
        loader = torch.utils.data.DataLoader(dataset, params.local_bs, shuffle=False)

        for j in tqdm(range(params.clients)):

            # 直接获取客户端模型计算
            local_model = clients_model[j]
            local_model.eval()
            global_model.eval()
            DDif = torch.zeros(num_classes).to(params.device)
            for x in loader:
                x = x.to(params.device)
                with torch.no_grad():
                    _,output_local = local_model(x)
                    _,output_global = global_model(x)
                    if 'MNIST' not in params.task:
                        output_local = torch.softmax(output_local, dim=1)
                        output_global = torch.softmax(output_global, dim=1)
                temp = torch.div(output_local, output_global + 1e-30)  # avoid zero-value
                temp = torch.sum(temp, dim=0)
                DDif.add_(temp)

            DDif /= num_samples
            DDifs = np.append(DDifs, DDif.cpu().numpy())
    DDifs = np.reshape(DDifs, (num_seeds, params.clients, -1))

    # cosine distance
    #  计算余弦距离，改为使用传入的客户端更新参数进行计算
    loaded_update_params = {
        client_id: torch.cat([param.flatten() for param in updates.values()]).cpu().numpy()
        for client_id, updates in clients_update.items()
    }
    # 获取所有客户端ID，并构建更新向量矩阵
    update_ids = list(loaded_update_params.keys())
    update_vectors = np.array([loaded_update_params[client_id] for client_id in update_ids], dtype=np.float64)
    # 计算余弦相似度矩阵，并将对角线（自相似）置零
    # cs = smp.cosine_similarity(update_vectors) - np.eye(len(update_vectors))
    cd = smp.cosine_distances(update_vectors)

    # classification
    # 余弦距离分类
    cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(cd)
    cosine_cluster_dists = dists_from_clust(cosine_clusters, params.clients)

    # 神经元更新能量分类
    neup_clusters = hdbscan.HDBSCAN().fit_predict(NEUPs)
    neup_cluster_dists = dists_from_clust(neup_clusters, params.clients)

    # 标签输出差异分类
    ddif_clusters, ddif_cluster_dists = [], []
    for i in range(num_seeds):
        ddif_cluster_i = hdbscan.HDBSCAN().fit_predict(DDifs[i])
        # ddif_clusters = np.append(ddif_clusters, ddif_cluster_i)
        ddif_cluster_dists = np.append(ddif_cluster_dists,
                                       dists_from_clust(ddif_cluster_i, params.clients))
    merged_ddif_cluster_dists = np.mean(np.reshape(ddif_cluster_dists,
                                                   (num_seeds, params.clients,
                                                    params.clients)),
                                        axis=0)
    merged_distances = np.mean([merged_ddif_cluster_dists,
                                neup_cluster_dists,
                                cosine_cluster_dists], axis=0)
    clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(merged_distances)
    # 可疑的数量
    positive_counts = {}
    total_counts = {}

    for i, c in enumerate(clusters):
        if c == -1:
            continue
        if c in positive_counts:
            positive_counts[c] += 1 if labels[i] else 0
            total_counts[c] += 1
        else:
            positive_counts[c] = 1 if labels[i] else 0
            total_counts[c] = 1

    # Aggregate and norm-clipping
    weight_accumulator = torch.zeros_like(global_model_param)
    st = np.median(ed)
    print(f"Deepsight: clipping bound {st}")
    # adv_clip = []
    discard_name = []
    for i, c in enumerate(clusters):
        # if i < params.clients*params.malicious:
        #     adv_clip.append(st / ed[i])
        if c != -1 and positive_counts[c] / total_counts[c] < tau:
            # 将所有参数展开后合并成一个一维向量
            loaded_params = clients_update[i]
            loaded_params = torch.cat([param.flatten() for param in loaded_params.values()])
            # 检查范数，进行裁剪
            if 1 > st / ed[i]:
                loaded_params.mul_(st / ed[i])

            # 更新全局模型
            weight_accumulator.add_(loaded_params)  # 直接累加
        else:
            discard_name.append(i)
    # logger.warning(f"Deepsight: clip for adv {adv_clip}")
    # return weight_accumulator
    print(f"恶意客户端： {discard_name}")
    global_model_param = global_model_param + weight_accumulator / len(clients_update)  # 等权重平均
    # global_model_param = global_model_param + weight_accumulator / sum(wv)  # 等权重平均
    return global_model_param

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        return noise


def dists_from_clust(clusters, N):
    pairwise_dists = np.ones((N, N))  # 创建一个全1的N×N矩阵
    for i in range(N):
        for j in range(N):
            if clusters[i] == clusters[j] and clusters[i] != -1 and clusters[j] !=-1:
                pairwise_dists[i][j] = 0  # 如果两个样本属于同一个聚类，距离设为0
    return pairwise_dists

def Rflbat(clients_param, args, eps1=2, eps2=1.5, depth=0, max_depth=3):
    num_clients = len(clients_param)
    clients = list(range(num_clients))
    flattened_params = []
    layer_name = 'fc2' if 'MNIST' in args.task else 'fc'
    # fc_layers = ['fc', 'fc1', 'fc2']
    for client_idx in range(num_clients):
        selected_params = []
        for name, param in clients_param[client_idx].items():
            if name.endswith(f"{layer_name}.bias") or name.endswith(f"{layer_name}.weight"):
            # if not name.endswith(f"{layer_name}.bias") and not name.endswith(f"{layer_name}.weight"):
            # if not any(name.endswith(f"{layer}.weight") or name.endswith(f"{layer}.bias") for layer in fc_layers):
                selected_params.append(param.view(-1))
        client_vector = torch.cat(selected_params).cpu().numpy()
        flattened_params.append(client_vector)

    # Step 1: PCA 降维
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(flattened_params)

    # Step 2: 欧氏距离聚合
    euclidean_scores = []
    for i in range(num_clients):
        dist_sum = sum(np.linalg.norm(reduced_vectors[i] - reduced_vectors[j])
                       for j in range(num_clients) if i != j)
        euclidean_scores.append(dist_sum)

    median_dist = np.median(euclidean_scores)
    prelim_accept = [i for i, score in enumerate(euclidean_scores) if score < eps1 * median_dist]
    reduced_accepted = np.stack([reduced_vectors[i] for i in prelim_accept])

    # Step 3: Gap Statistics 聚类
    num_clusters = gap_statistics(reduced_accepted, num_sampling=2, K_max=10)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++').fit(reduced_accepted)
    cluster_labels = kmeans.labels_

    # Step 4: 每个聚类计算余弦相似度中位数
    cluster_similarities = []
    for cluster_id in range(num_clusters):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if len(indices) <= 1:
            cluster_similarities.append(1.0)
            continue
        cluster_vectors = [flattened_params[prelim_accept[i]] for i in indices]
        cos_matrix = cosine_similarity(cluster_vectors)
        cluster_similarities.append(np.median(np.mean(cos_matrix, axis=1)))

    # Step 5: 保留最相似聚类
    best_cluster_id = int(np.argmax(cluster_similarities))
    final_accept = [prelim_accept[i] for i, label in enumerate(cluster_labels) if label == best_cluster_id]

    # Step 6: 进一步基于欧氏距离筛选
    reduced_final = np.stack([reduced_vectors[i] for i in final_accept])
    refined_scores = []
    for i in range(len(reduced_final)):
        dist_sum = sum(np.linalg.norm(reduced_final[i] - reduced_final[j])
                       for j in range(len(reduced_final)) if i != j)
        refined_scores.append(dist_sum)

    refined_median = np.median(refined_scores)
    final_accept = [final_accept[i] for i, score in enumerate(refined_scores) if score < eps2 * refined_median]

    # Step 7: 最终判断
    malicious = [i for i in clients if i not in final_accept]

    # Step 8: 防御过强重试
    if depth > max_depth or len(malicious) == num_clients:
        if depth > max_depth:
            print("[Rflbat] Reach max depth. Returning all clients as malicious.")
        else:
            print("[Rflbat] All clients filtered out. Relaxing thresholds.")
        return Rflbat(clients_param, args, max(1, eps1 / 2), max(1, eps2 / 2), depth + 1, max_depth)

    return malicious


def gap_statistics(data, num_sampling=2, K_max=10):
    data = np.reshape(data, (data.shape[0], -1))

    # Feature normalization [0,1]
    norm_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        col_min = np.min(data[:, i])
        col_max = np.max(data[:, i])
        if col_max - col_min == 0:
            norm_data[:, i] = 0
        else:
            norm_data[:, i] = (data[:, i] - col_min) / (col_max - col_min)

    gap = []
    sd_list = []

    for k in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++').fit(norm_data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 真实数据损失
        intra_dist = sum(
            np.linalg.norm(centers[label] - norm_data[i])
            for i, label in enumerate(labels)
        )

        # 随机采样的参考损失
        ref_intra_dists = []
        for _ in range(num_sampling):
            fake_data = np.random.uniform(0, 1, size=norm_data.shape)
            fake_kmeans = KMeans(n_clusters=k, init='k-means++').fit(fake_data)
            fake_labels = fake_kmeans.labels_
            fake_centers = fake_kmeans.cluster_centers_

            ref_dist = sum(
                np.linalg.norm(fake_centers[label] - fake_data[i])
                for i, label in enumerate(fake_labels)
            )
            ref_intra_dists.append(ref_dist)

        log_ref = np.log(ref_intra_dists)
        gap_k = np.mean(log_ref) - np.log(intra_dist)
        sd_k = np.sqrt(np.mean((log_ref - np.mean(log_ref)) ** 2)) * np.sqrt((1 + 1 / num_sampling))
        gap.append(gap_k)
        sd_list.append(sd_k)

    # Gap value 选择最优 K
    for k in range(1, K_max):
        if gap[k - 1] - gap[k] + sd_list[k - 1] > 0:
            return k
    return K_max


def Remove_clients(clients_param, m_client):
    clients_param = [clients_param[i] for i in range(len(clients_param)) if i not in m_client]
    return clients_param

def Multi_krum(clients_param, global_model, malicious_clients, args, multi_k=True):
    layer_name = 'fc2' if args.dataset == 'FashionMNIST' else 'fc'
    dict_global_model = global_model.state_dict()
    update_params = []
    for key in range(len(clients_param)):
        selected_param = []
        # for name, param in clients_param[key].items():
        #     # if args.dataset == 'FashionMNIST' or 'fc' in name or 'layer3.1.conv1' in name:
        #     if layer_name in name:
        #         selected_param.append(param.view(-1) - dict_global_model[name].view(-1))
        # combined_tensor = torch.cat(selected_param)
        update_params.append(torch.cat([p.view(-1) for p in clients_param[key].values()]) - torch.cat([p.view(-1) for p in global_model.state_dict().values()]))

    all_indices = np.arange(len(update_params))

    num = len(update_params) - int(len(malicious_clients) * 2)

    torch.cuda.empty_cache()
    distances = []
    for update in update_params:
        distance = []
        for update_ in update_params:
            distance.append(torch.norm(update - update_))
        distance = torch.Tensor(distance).float()
        distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

    # distances = torch.sort(distances, dim=1)[0]         # 对二维列表中的每一行进行升序排序
    distances, sorted_indices = torch.sort(distances, dim=1)

    k = len(update_params) - 5 - len(malicious_clients)
    # k = 80
    scores = torch.sum(distances[:, :k], dim=1)
    indices = torch.argsort(scores)[:num]

    candidate_indices = (all_indices[indices.cpu().numpy()])
    all_client = list(range(len(update_params)))
    malicious_c = [element for element in all_client if element not in candidate_indices]
    malicious_c = torch.tensor(malicious_c)
    # intersection = torch.tensor([x for x in malicious_clients if x in malicious_c])

    return malicious_c.tolist()


