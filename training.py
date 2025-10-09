import logging
import random
from copy import deepcopy

import hdbscan
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from attacks.ReplaceAttack import ReplaceAttack
from demo.get_gpu.free_gpu import get_max_free_memory_gpu
from demo.pca.client_updates_pca import visualize_client_updates_pca
from demo.trigger.TriggerGen import TriggerGenerator
from utils.Test import Backdoor_Evaluate, Evaluate
from aggregation import Aggregation
from helper import Helper
from utils.diff_2_models import ModelTransformDiffAnalyzer
from utils.parameters import Params
from utils import utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from purification.ModelPurifier import ModelPurifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
logger = logging.getLogger('logger')

def load_params_from_yaml(yaml_file: str) -> Params:
    with open(yaml_file) as f:
        yaml_data = yaml.load(f,Loader=yaml.FullLoader)  # 读取 YAML 配置
    return yaml_data  # 直接返回字典

def PDBA(global_model, test_dataset, params):
    pdb_dataset = deepcopy(test_dataset)
    # utils.pdb_process(pdb_dataset, params.origin_target, params.aim_target)
    utils.pdb_process(pdb_dataset, 5)
    # visualize_pdb_samples(pdb_dataset, target_label=1, num_samples=10)
    global_model.train()
    global_model.to(params.device)
    loader = DataLoader(pdb_dataset, batch_size=params.local_bs, shuffle=False)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=params.lr, momentum=params.momentum)
    last_loss = None
    loss_func = nn.CrossEntropyLoss().to(params.device)
    for _ in range(1):
        for images, labels in loader:
            optimizer.zero_grad()
            inputs = images.to(device=params.device, non_blocking=True)
            labels = labels.to(device=params.device, non_blocking=True)
            _, outputs = global_model(inputs)
            loss =  loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
    return global_model, last_loss

def pdb_test(clients_model, pdb_dataset, params, loss_mode="max"):
    """
    评估各客户端模型在注入触发器数据集上的攻击成功率（ASR）和指定方式的批次损失

    Args:
        clients_model: Dict[int, torch.nn.Module]，多个客户端模型
        pdb_dataset: 已注入触发器的数据集，目标标签统一为 aim_target
        params: 参数对象，包含 device, bs（batch size）, aim_target 等
        loss_mode: str，指定返回损失类型，支持 'max'、'min'、'avg'

    Returns:
        asr_list: List[float]，每个模型的攻击成功率（0~1）
        loss_list: List[float]，每个模型的指定损失值
    """
    assert loss_mode in {"max", "min", "avg"}, f"Unsupported loss_mode: {loss_mode}"

    pdb_dataset_ = deepcopy(pdb_dataset)
    utils.pdb_process(pdb_dataset_, 5, 1.0)  # 插入触发器，统一目标为类别 8

    loader = DataLoader(pdb_dataset_, batch_size=params.bs, shuffle=False)

    asr_list = []
    loss_list = []
    loss_func = nn.CrossEntropyLoss().to(params.device)

    for idx, model in clients_model.items():
        model.eval()
        model.to(params.device)

        total = 0
        success = 0
        batch_losses = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(params.device, non_blocking=True)
                labels = labels.to(params.device, non_blocking=True)

                _, outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                success += (preds == labels ).sum().item()
                total += labels.size(0)

                loss = loss_func(outputs, labels)
                batch_losses.append(loss.item())

        asr = success / total if total > 0 else 0.0

        if batch_losses:
            max_loss = max(batch_losses)
            min_loss = min(batch_losses)
            avg_loss = sum(batch_losses) / len(batch_losses)
        else:
            max_loss = min_loss = avg_loss = 0.0

        # 根据用户指定的返回模式选择损失
        if loss_mode == "max":
            final_loss = max_loss
        elif loss_mode == "min":
            final_loss = min_loss
        else:
            final_loss = avg_loss

        loss_list.append(final_loss)
        asr_list.append(asr)

        print(f"[PDB-Test] Client {idx}: ASR = {asr:.4f}, Max Loss = {max_loss:.4f}, "
              f"Min Loss = {min_loss:.4f}, Avg Loss = {avg_loss:.4f}")

    return asr_list, loss_list

def visualize_pdb_samples(pdb_dataset, target_label=None, num_samples=5):
    """
    可视化被注入触发器的样本（例如右下角小白块）。
    兼容 MNIST (灰度图) 和 CIFAR-10 (RGB图)。
    """
    shown = 0
    for i in range(len(pdb_dataset.data)):
        label = pdb_dataset.targets[i]

        if target_label is not None and label != target_label:
            continue

        img = pdb_dataset.data[i]

        # 若为 tensor，先转换为 numpy
        if isinstance(img, torch.Tensor):
            img = img.numpy()

        # 如果图像是 (C, H, W)，需转为 (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        plt.subplot(1, num_samples, shown + 1)

        # 判断是否为灰度图
        if img.ndim == 2 or img.shape[2] == 1:
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)

        plt.axis('off')
        plt.title(f"Label: {label}")
        shown += 1

        if shown >= num_samples:
            break

    plt.suptitle("PDBA Trigger Visualization", fontsize=14)
    plt.show()

def cluster_loss(loss_list, min_cluster_size=2,  n_clusters=2):
    """
    使用 HDBSCAN 对 loss_list 进行无监督聚类。

    Args:
        loss_list: List[float]，每个客户端的 loss 值
        min_cluster_size: 最小簇大小，越小越敏感

    Returns:
        labels: 每个客户端的聚类标签（-1 表示异常点）
    """
    loss_array = np.array(loss_list).reshape(-1, 1)

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    # cluster_labels = clusterer.fit_predict(loss_array)
    #
    # # 输出聚类标签
    # for label in set(cluster_labels):
    #     members = [i for i, l in enumerate(cluster_labels) if l == label]
    #     if label == -1:
    #         print(f"\n异常客户端（HDBSCAN认为是离群点）：{members}")
    #     else:
    #         print(f"簇 {label} 的客户端：{members}")
    #
    # return cluster_labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(loss_array)

    labels = kmeans.labels_  # 每个客户端属于哪个簇
    # 输出每个簇的客户端
    for i in range(n_clusters):
        cluster_members = [idx for idx, label in enumerate(labels) if label == i]
        print(f"\n簇 {i} 包含客户端：{cluster_members}")

    return labels

def run_fl_round(helper: Helper, epoch, generator):

    """
    开始注入防御性后门pdb
    """
    # if epoch == 1:
    #     helper.global_model,pdb_loss = PDBA(helper.global_model,helper.test_dataset,helper.params)
    #     print(f"[PDBA] pdb_loss: {pdb_loss:.4f}")
    # 准备工作，包括模型初始化，
    global_model = helper.global_model
    helper.client_loss=[]
    # PDBA(global_model, helper.test_dataset, helper.params)

    # 进度条定义
    tqdm_bar = tqdm(
        enumerate(helper.clients),
        total=len(helper.clients),
        desc=f"Epoch {epoch} - Training Clients",
        bar_format="{l_bar}{bar}| Client {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        leave=True,
        colour="cyan"  # 保持青色
    )
    benign_ = []
    malicious_ = []

    benign_loss=[]
    malicious_loss=[]
    attack = ReplaceAttack(helper.params)  # 初始化攻击器
    watermark_history_before = []
    watermark_history_after = []
    for idx, client in tqdm_bar:
        # 客户端训练 无论恶意还是非恶意都是通过local_train训练，因为名字一样
        mark_alloc = {}
        # 更新client的global_model
        client.global_model = global_model
        if epoch > 1:
            helper.teacher_model=helper.clients_model[0]
        # if idx in helper.malicious_clients:
        #     print(idx)
        # local_model, loss_val = client.local_train(helper.loss_func,epoch,teacher_model=helper.teacher_model,mask=helper.mask,pattern=helper.pattern,delta_z=helper.delta_z)
        local_model, loss_val = client.local_train(helper.loss_func,epoch,teacher_model=helper.teacher_model,mask=helper.mask,pattern=helper.pattern,delta_z=None)
        # if idx in helper.malicious_clients:
        #     client.match_rate_before_agg = client.extract_watermark(local_model, client.train_loader)
        #     watermark_history_before.append({
        #         'epoch': epoch,
        #         'id':idx,
        #         'match_rate': client.match_rate_before_agg,
        #         'loss': loss_val
        #     })
        """
        恶意模型更改
        """
        # attack.perform_attack(idx, local_model, global_model.state_dict())

        # 返回的客户端参数是多维张量的形式,转成一维张量
        # 这里统一存放模型参数，谁要转一维谁就去自己的模块转
        # with torch.no_grad():
        #     local_param = utils.model_to_vector(local_model, helper.params).detach()
        # 存储每个模型的参数
        helper.clients_model[idx] = local_model
        # import matplotlib.pyplot as plt
        #
        # gradients = []
        # param_names = []
        # for name, param in local_model.named_parameters():
        #     if param.grad is not None:
        #         param_names.append(name)
        #         gradients.append(param.grad.abs().mean().item())
        #
        # plt.figure(figsize=(12, 6))  # 调整图形大小以适应长名称
        # plt.bar(param_names, gradients)
        # plt.title("Layer-wise Gradient Magnitude")
        # plt.xlabel("Parameter Name")
        # plt.ylabel("Avg Gradient")
        # plt.xticks(rotation=90)  # 旋转x轴标签90度避免重叠
        # # plt.tight_layout()  # 自动调整布局
        # plt.show()
        """
        可视化客户端之间的差异
        """
        # if idx not in helper.malicious_clients:
        #     benign_.append(idx)
        # if idx in helper.malicious_clients:
        #     malicious_.append(idx)
        #
        # if len(benign_) >= 2 and len(malicious_) >= 2:
        #     # 随机选择两个良性客户端
        #     benign_sample = random.sample(benign_, 2)
        #     # 随机选择两个恶意客户端
        #     malicious_sample = random.sample(malicious_, 2)
        #     # 随机选择一个良性和一个恶意客户端
        #     mixed_sample = [random.choice(benign_), random.choice(malicious_)]
        #
        #     # 获取模型参数
        #     benign_models = [helper.clients_model[idx] for idx in benign_sample]
        #     # malicious_models = [helper.clients_model[idx] for idx in malicious_sample]
        #     # mixed_models = [helper.clients_model[idx] for idx in mixed_sample]
        #
        #     analyzer = ModelTransformDiffAnalyzer(benign_models[0],benign_models[1])
        #     analyzer.plot_spatial_diff()
        #     analyzer.plot_fft_diff()
        #     analyzer.plot_dct_diff()
        #     print("ok")

        # 存储每个模型的损失
        helper.client_loss.append(loss_val)
        if idx in helper.malicious_clients:
            malicious_loss.append(loss_val)
        else:
            benign_loss.append(loss_val)

        # 计算参数变化量并保存，计算这次训练的模型的变化量，在赋值给每一个模型的变化量字典
        client_update=utils.get_fl_update(local_model, global_model)
        helper.clients_update[idx]=client_update
        # 初始化字典项，避免 KeyError
        if helper.params.agg == 'FoolsGold':
            if idx not in helper.clients_his_update:
                helper.clients_his_update[idx] = client_update   # 直接存入初始值
            else:
                # 若已存在，进行累加
                for key, value in client_update.items():
                    helper.clients_his_update[idx][key] += value  # 逐个参数累加

    # print("\n",sum(malicious_loss)/len(malicious_loss))
    # print(sum(benign_loss)/len(benign_loss))
    print(sum(helper.client_loss)/len(helper.clients))

    # PCA降维可视化
    # visualize_client_updates_pca(helper.clients_update, helper.params.backdoor_clients)
    # 预留位置，对恶意模型进行预处理，增强隐蔽性

    '''
    pdb检测开始
    '''
    # asr_list, loss_list = pdb_test(helper.clients_model, helper.test_dataset, helper.params)
    # # labels = cluster_loss(loss_list)
    # # 排序：从大到小，保留原始索引
    # sorted_loss_indices = sorted(enumerate(loss_list), key=lambda x: x[1], reverse=True)
    # # 取前20个
    # top20 = sorted_loss_indices[:20]
    # # 提取索引
    # top20_indices = [idx for idx, _ in top20]
    # # 先对 top20 的客户端索引从小到大排序
    # top20_sorted = sorted(top20_indices)
    # print("Top 20 clients with highest loss (from high to low):", top20_sorted)
    # # 准备列表保存后门 / 非后门客户端
    # backdoor_clients = []
    # benign_clients = []
    # # 遍历排序后的索引，分类判断
    # for cid in top20_sorted:
    #     if cid in helper.params.backdoor_clients:
    #         print(f"客户端 {cid} 是后门客户端 ✅")
    #         backdoor_clients.append(cid)
    #     else:
    #         print(f"客户端 {cid} 是正常客户端 ❌")
    #         benign_clients.append(cid)
    # # 最终输出汇总
    # print("\n属于后门客户端的ID列表：", backdoor_clients)
    # print("属于正常客户端的ID列表：", benign_clients)
    # # 所有保留的客户端 idx
    # keep_indices = set(helper.clients_model.keys()) - set(top20_indices)
    # # 构建一个新的字典，只保留没被剔除的客户端
    # new_clients_model = {
    #     idx: model for idx, model in helper.clients_model.items() if idx in keep_indices
    # }
    '''
    pdb检测结束
    '''

    '''
    寻找目标标签开始
    '''
    # # 提取全连接层每个类别参数
    # layer_name = 'fc2' if 'MNIST' in helper.params.task else 'fc'
    # per_class_params = {}  # 每个客户端 -> 每个类别 -> 参数向量
    # for client_id, updates in helper.clients_update.items():
    #     weight_name = f"{layer_name}.weight"
    #     bias_name = f"{layer_name}.bias"
    #
    #     if weight_name not in updates or bias_name not in updates:
    #         raise ValueError(f"Missing {layer_name} parameters in client {client_id}")
    #
    #     weight = updates[weight_name]  # [num_classes, D]
    #     bias = updates[bias_name]  # [num_classes]
    #
    #     # 提取每一类对应的向量
    #     num_classes = weight.shape[0]
    #     client_class_params = {}
    #
    #     for c in range(num_classes):
    #         weight_c = weight[c].view(-1).cpu()  # [D]
    #         bias_c = bias[c].view(-1).cpu()  # [1]
    #         class_param = torch.cat([weight_c, bias_c])  # [D+1]
    #         client_class_params[c] = class_param.numpy()
    #     per_class_params[client_id] = client_class_params
    #
    # for c in range(helper.params.num_classes):
    #     class_vectors = [per_class_params[client_id][c] for client_id in per_class_params]
    #     norms = [np.linalg.norm(v) for v in class_vectors]
    #     print(f"Class {c}: mean norm = {np.mean(norms):.4f}, std = {np.std(norms):.4f}")
    #
    # # 构造数据
    # X = []  # 所有参数向量
    # y = []  # 对应的类别标签
    # for client_id in per_class_params:
    #     for c in per_class_params[client_id]:
    #         X.append(per_class_params[client_id][c])
    #         y.append(c)
    #
    # X = np.array(X)
    # y = np.array(y)
    #
    # # PCA 降维
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    #
    # # 可视化
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=40, alpha=0.8)
    #
    # # 添加图例
    # legend = plt.legend(*scatter.legend_elements(), title="Class Label")
    # plt.gca().add_artist(legend)
    #
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.title("Per-class Parameter Distribution (PCA)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    '''
    寻找目标标签结束
    '''

    # 聚合模型之前，分析客户端更新以寻找目标标签和可疑客户端
    # print("\n" + "=" * 20 + " 开始分析客户端更新 " + "=" * 20)
    # potential_target_label, suspicious_clients = find_target_label_and_suspicious_clients(
    #     helper,  # <--- 传递 helper 对象
    #     helper.clients_update,
    #     helper.params
    # )
    # if potential_target_label is not None:
    #     print(f"分析完成。推断的目标标签: {potential_target_label}, 可疑客户端: {suspicious_clients}")
    # print("=" * 20 + " 客户端更新分析结束 " + "=" * 20 + "\n")

    # # ====== 在聚合前对所有客户端模型进行触发器逆向分析 ======
    # print("\n" + "=" * 20 + " 开始对所有客户端模型进行触发器逆向 " + "=" * 20)
    # client_triggers_info = []
    # suspicious_clients_by_trigger = set()
    #
    # for client_id, local_model in helper.clients_model.items():
    #     print(f"[分析] 对客户端 {client_id} 的模型进行触发器逆向...")
    #     mask, pattern, delta_z = generator.generate(
    #         model=local_model,
    #         tri_dataset=helper.test_dataset,
    #
    #     )
    #     # 计算并存储触发器的大小 (L2范数)
    #     trigger_size = 0.0
    #     if delta_z is not None:
    #         trigger_size = torch.norm(delta_z).item()
    #     elif mask is not None:  # 如果是基于mask的触发器，可以计算其L0范数
    #         trigger_size = torch.sum(mask).item()
    #
    #     client_triggers_info.append({'id': client_id, 'size': trigger_size})
    #     print(f"[分析结果] 客户端 {client_id}: 逆向触发器大小 = {trigger_size:.4f}")
    # # 根据触发器大小对客户端进行排序
    # client_triggers_info.sort(key=lambda x: x['size'], reverse=True)
    #
    # print("\n--- 根据逆向触发器大小排序的客户端 ---")
    # for info in client_triggers_info:
    #     is_malicious = " (恶意)" if info['id'] in helper.malicious_clients else ""
    #     print(f"客户端: {info['id']}\t 触发器大小: {info['size']:.4f}{is_malicious}")
    # print("=" * 20 + " 触发器逆向分析结束 " + "=" * 20 + "\n")
    # 聚合模型，计算聚合后的模型参数更新量
    Aggregation(helper.params,helper,global_model, helper.clients_model, helper.clients_update,helper.clients_his_update,helper.clients, helper.loss_func)

    # ====== 使用ModelPurifier对全局模型进行净化，消除后门 ======
    if epoch > 0:
        # 逆向生成触发器
        mask, pattern, delta_z = generator.generate(
            model=helper.global_model,
            tri_dataset = helper.test_dataset,
        )
        # 可视化 delta_z
        # if delta_z.shape[0]==512:
        #     visualize_delta_feature(delta_z, dataset_type="cifar10")
        # else:
        #     visualize_delta_feature(delta_z, dataset_type="mnist")

        helper.mask=mask
        helper.pattern=pattern
        helper.delta_z=delta_z

        # 使用ModelPurifier进行净化
        if delta_z is not None:
        # if delta_z is None:
            # 初始化净化器（如果还没有的话）
            if not hasattr(helper, 'model_purifier'):
                helper.model_purifier = ModelPurifier(device=helper.params.device)

            # ====== 新增：选择净化策略 ======
            # 可以通过参数控制使用哪种净化方法
            purification_strategy = getattr(helper.params, 'purification_strategy', 'reverse_expert')

            if purification_strategy == 'reverse_expert':
                print("\n[净化策略] 使用策略B：反向专家微调进行净化")
                # 执行策略B：反向专家微调净化
                purify_result = helper.model_purifier.reverse_expert_fine_tuning_purification(
                    model=helper.global_model,
                    delta_z=delta_z,
                    target_label=helper.params.aim_target,
                    test_dataset=helper.test_dataset,
                    params=helper.params,
                    epoch=epoch
                )
            elif purification_strategy == 'feature_unlearning':
                print("\n[净化策略] 使用特征解毒方法进行净化")
                # 执行特征解毒净化
                purify_result = helper.model_purifier.feature_unlearning_purification(
                    model=helper.global_model,
                    delta_z=delta_z,
                    target_label=helper.params.aim_target,
                    test_dataset=helper.test_dataset,
                    params=helper.params,
                    epoch=epoch
                )
            elif purification_strategy == 'traditional':
                print("\n[净化策略] 使用传统投影/神经元定位方法进行净化")
                # 执行传统净化方法
                purify_result = helper.model_purifier.purify_model(
                    model=helper.global_model,
                    delta_z=delta_z,
                    target_label=helper.params.aim_target,
                    clients_update=helper.clients_update,
                    test_dataset=helper.test_dataset,
                    params=helper.params,
                    epoch=epoch
                )
            else:  # 'auto' - 智能选择净化策略
                print("\n[净化策略] 自动选择最佳净化策略")

                # 先快速评估触发器强度来决定使用哪种方法
                trigger_strength = helper.model_purifier._validate_feature_trigger(
                    helper.global_model, delta_z, helper.params.aim_target
                )

                print(f"[策略选择] 触发器强度评估: {trigger_strength:.3f}")

                if trigger_strength >= 0.8:
                    # 触发器很强，使用特征解毒方法（更精准，副作用小）
                    print("[策略选择] 触发器强度高，选择特征解毒方法")
                    purify_result = helper.model_purifier.feature_unlearning_purification(
                        model=helper.global_model,
                        delta_z=delta_z,
                        target_label=helper.params.aim_target,
                        test_dataset=helper.test_dataset,
                        params=helper.params,
                        epoch=epoch
                    )

                    # 如果特征解毒失败，回退到传统方法
                    if purify_result is None or not purify_result.get('success', False):
                        print("[策略回退] 特征解毒失败，回退到传统方法")
                        purify_result = helper.model_purifier.purify_model(
                            model=helper.global_model,
                            delta_z=delta_z,
                            target_label=helper.params.aim_target,
                            clients_update=helper.clients_update,
                            test_dataset=helper.test_dataset,
                            params=helper.params,
                            epoch=epoch
                        )
                else:
                    # 触发器较弱或不明确，使用传统方法（更鲁棒）
                    print("[策略选择] 触发器强度较低，选择传统投影/神经元方法")
                    purify_result = helper.model_purifier.purify_model(
                        model=helper.global_model,
                        delta_z=delta_z,
                        target_label=helper.params.aim_target,
                        clients_update=helper.clients_update,
                        test_dataset=helper.test_dataset,
                        params=helper.params,
                        epoch=epoch
                    )

            # ====== 净化结果处理 ======
            if purify_result is not None:
                # 安全地获取净化强度信息，因为不同净化模式返回的键可能不同
                attack_intensity = purify_result.get('attack_intensity', 0.0)
                purify_method = purify_result.get('purify_method', 'unknown')

                if purify_method == 'reverse_expert_finetuning':
                    # 策略B：反向专家微调模式的输出信息
                    asr_after = purify_result.get('attack_success_rate', 0.0)
                    main_acc = purify_result.get('main_accuracy', 0.0)
                    unlearning_steps = purify_result.get('unlearning_steps', 0)
                    success = purify_result.get('success', False)
                    rollback = purify_result.get('rollback', False)
                    initial_loss = purify_result.get('initial_loss', 0.0)
                    final_loss = purify_result.get('final_loss', 0.0)

                    print(f"\n[策略B净化完成] 成功: {success}")
                    print(f"  解毒步数: {unlearning_steps}")
                    print(f"  初始损失: {initial_loss:.4f} → 最终损失: {final_loss:.4f}")
                    print(f"  净化后ASR: {asr_after:.3f}")
                    print(f"  主任务准确率: {main_acc:.3f}")
                    print(f"  是否回滚: {rollback}")

                elif purify_method == 'feature_unlearning':
                    # 特征解毒模式的输出信息
                    asr_after = purify_result.get('attack_success_rate', 0.0)
                    main_acc = purify_result.get('main_accuracy', 0.0)
                    unlearning_steps = purify_result.get('unlearning_steps', 0)
                    trigger_effectiveness = purify_result.get('trigger_effectiveness', 0.0)
                    stage_completed = purify_result.get('stage_completed', 0)
                    success = purify_result.get('success', False)
                    rollback = purify_result.get('rollback', False)

                    print(f"[特征解毒完成] 成功: {success}, 阶段: {stage_completed}/5")
                    print(f"  触发器有效性: {trigger_effectiveness:.3f}")
                    print(f"  解毒步数: {unlearning_steps}")
                    print(f"  净化后ASR: {asr_after:.3f}")
                    print(f"  主任务准确率: {main_acc:.3f}")
                    print(f"  是否回滚: {rollback}")

                elif purify_method == 'neuron_targeting':
                    # 神经元定位模式的输出信息
                    backdoor_neurons = purify_result.get('backdoor_neurons', [])
                    asr_after_pruning = purify_result.get('asr_after_pruning', 0.0)
                    print(f"[净化完成] 攻击强度: {attack_intensity:.3f}, "
                          f"净化方法: 神经元定位, 剪枝神经元: {backdoor_neurons}, "
                          f"剪枝后ASR: {asr_after_pruning:.3f}")
                elif purify_method == 'projection':
                    # 投影净化模式的输出信息
                    purify_ratio = purify_result.get('purify_ratio', 0.0)
                    print(f"[净化完成] 攻击强度: {attack_intensity:.3f}, "
                          f"净化方法: 投影净化, 净化强度: {purify_ratio:.3f}")
                else:
                    # 兜底情况
                    print(f"[净化完成] 攻击强度: {attack_intensity:.3f}, "
                          f"净化方法: {purify_method}")
        else:
            print("No feature trigger generated for purification.")


def find_target_label_and_suspicious_clients(helper, clients_update, params, k=2.0):
    """
    【基于权重异常度的新方法】分析后门攻击在目标类别权重上留下的痕迹
    """
    layer_name = 'fc2' if 'MNIST' in params.task else 'fc'
    num_classes = params.num_classes

    print("\n--- 基于权重异常度的目标标签推断 ---")

    # 1. 收集所有客户端在每个类别上的权重更新
    class_weight_updates = {c: [] for c in range(num_classes)}
    class_bias_updates = {c: [] for c in range(num_classes)}
    client_ids = []

    for client_id, updates in clients_update.items():
        weight_name = f"{layer_name}.weight"
        bias_name = f"{layer_name}.bias"

        if weight_name not in updates or bias_name not in updates:
            continue

        client_ids.append(client_id)
        weight_update = updates[weight_name].cpu()  # [num_classes, feature_dim]
        bias_update = updates[bias_name].cpu()  # [num_classes]

        for c in range(num_classes):
            class_weight_updates[c].append(weight_update[c].numpy())
            class_bias_updates[c].append(bias_update[c].item())

    if len(client_ids) < 4:
        return None, []

    # 2. 计算每个类别的异常度指标
    class_anomaly_scores = {}

    for c in range(num_classes):
        weight_vectors = np.array(class_weight_updates[c])  # [num_clients, feature_dim]
        bias_values = np.array(class_bias_updates[c])  # [num_clients]

        # 指标1: 权重范数的方差（后门会导致某些客户端权重范数异常大）
        weight_norms = np.linalg.norm(weight_vectors, axis=1)
        norm_variance = np.var(weight_norms)

        # 指标2: 权重方向的离散度（后门会让恶意客户端的权重方向与良性客户端不同）
        # 计算所有权重向量两两之间的余弦相似度
        similarities = []
        for i in range(len(weight_vectors)):
            for j in range(i + 1, len(weight_vectors)):
                sim = np.dot(weight_vectors[i], weight_vectors[j]) / (
                        np.linalg.norm(weight_vectors[i]) * np.linalg.norm(weight_vectors[j]) + 1e-8
                )
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 1.0
        similarity_variance = np.var(similarities) if similarities else 0.0

        # 指标3: 偏置更新的方差（后门攻击通常会显著调整目标类别的偏置）
        bias_variance = np.var(bias_values)

        # 指标4: 极端权重的占比（后门需要某些权重变得很大以响应触发器特征）
        weight_magnitudes = np.abs(weight_vectors).flatten()
        threshold_95 = np.percentile(weight_magnitudes, 95)
        extreme_ratio = np.mean(weight_magnitudes > threshold_95)

        # 综合异常度得分（加权组合各指标）
        anomaly_score = (
                0.5 * norm_variance +  # 范数方差权重最高
                0.2 * similarity_variance +  # 相似度方差次之
                0.2 * bias_variance +  # 偏置方差
                0.1 * extreme_ratio  # 极端权重占比
        )

        class_anomaly_scores[c] = {
            'total_score': anomaly_score,
            'norm_variance': norm_variance,
            'similarity_variance': similarity_variance,
            'bias_variance': bias_variance,
            'extreme_ratio': extreme_ratio,
            'avg_similarity': avg_similarity
        }

        print(f"Class {c}: 异常度={anomaly_score:.4f} "
              f"(范数方差={norm_variance:.3f}, 相似度方差={similarity_variance:.3f}, "
              f"偏置方差={bias_variance:.3f}, 极端权重占比={extreme_ratio:.3f})")

    # 3. 时间平滑（保留之前的EMA机制）
    if not hasattr(helper, 'target_label_inference_scores'):
        helper.target_label_inference_scores = {c: 0.0 for c in range(num_classes)}
    if not hasattr(helper, 'inference_smoothing_alpha'):
        helper.inference_smoothing_alpha = 0.3  # 稍微降低平滑系数，增加响应性

    alpha = helper.inference_smoothing_alpha
    for c in range(num_classes):
        current_score = class_anomaly_scores[c]['total_score']
        previous_score = helper.target_label_inference_scores[c]
        smoothed_score = alpha * current_score + (1 - alpha) * previous_score
        helper.target_label_inference_scores[c] = smoothed_score

    # 4. 推断目标标签
    potential_target_label = max(helper.target_label_inference_scores,
                                 key=helper.target_label_inference_scores.get)
    max_score = helper.target_label_inference_scores[potential_target_label]

    print(f"\n[推断] 潜在目标标签: {potential_target_label} (平滑异常度: {max_score:.4f})")

    # 5. 基于异常度识别可疑客户端
    suspicious_clients = []
    if max_score > 0.01:  # 一个相对较低的阈值，避免在无攻击时误报
        target_weights = np.array(class_weight_updates[potential_target_label])
        target_norms = np.linalg.norm(target_weights, axis=1)

        # 使用四分位数方法识别离群点，比固定倍数更稳健
        q75 = np.percentile(target_norms, 75)
        q25 = np.percentile(target_norms, 25)
        iqr = q75 - q25
        upper_threshold = q75 + 1.5 * iqr  # 经典的离群点检测阈值

        for i, norm in enumerate(target_norms):
            if norm > upper_threshold:
                suspicious_clients.append(client_ids[i])
                print(f"  - Client {client_ids[i]} 可疑 (权重范数: {norm:.4f} > 阈值: {upper_threshold:.4f})")

    print(f"\n[结果] 识别出 {len(suspicious_clients)} 个可疑客户端: {sorted(suspicious_clients)}")

    return potential_target_label, sorted(suspicious_clients)

def get_trigger_gradient_vector(model, delta_z, target_label, device=None):
    """
    计算特征触发器对应的梯度向量，用于净化
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # 创建一个随机输入来获取梯度
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # 假设是CIFAR-10格式

    # 前向传播获取特征
    features, _ = model(dummy_input)

    # 在特征上添加触发器
    triggered_features = features + delta_z.unsqueeze(0)

    # 计算对目标标签的损失
    _, outputs = model(features=triggered_features)
    target = torch.tensor([target_label]).to(device)
    loss = F.cross_entropy(outputs, target)

    # 计算梯度
    model.zero_grad()
    loss.backward(retain_graph=True)

    # 提取全连接层的梯度
    fc_gradients = []
    fc_names = []
    for name, param in model.named_parameters():
        if "fc" in name and param.grad is not None:
            fc_gradients.append(param.grad.detach().flatten())
            fc_names.append(name)

    if fc_gradients:
        g_trigger = torch.cat(fc_gradients)
        return g_trigger, fc_names
    else:
        return None, []
# 计算客户端更新与触发器梯度的相似度 未启用
def compute_fc_similarity_with_trigger(clients_update, g_trigger, fc_names):
    """
    计算客户端更新与触发器梯度的相似度
    """
    similarities = {}

    # 归一化触发器梯度
    g_trigger_norm = g_trigger / (g_trigger.norm() + 1e-12)

    for client_id, updates in clients_update.items():
        # 提取该客户端的全连接层更新
        client_fc_updates = []
        for name in fc_names:
            if name in updates:
                client_fc_updates.append(updates[name].flatten())

        if client_fc_updates:
            flat_update = torch.cat(client_fc_updates)
            # 归一化客户端更新
            flat_update_norm = flat_update / (flat_update.norm() + 1e-12)

            # 计算余弦相似度
            similarity = torch.dot(g_trigger_norm, flat_update_norm).item()
            similarities[client_id] = similarity
        else:
            similarities[client_id] = 0.0

    return similarities

def evaluate_model(params, global_model, test_dataset, loss_func):
    """
    评估全局模型的性能。
    若为后门攻击，则调用 Backdoor_Evaluate，并返回后门准确率；
    否则调用 Evaluate，back_acc 和 back_loss 返回 None。
    """
    if params.attack_type in ['How_backdoor', 'dct', 'dba']:
        # 对测试集进行后门处理
        # test_dataset_ = deepcopy(test_dataset)
        # utils.Backdoor_process(test_dataset_, params.origin_target, params.aim_target)

        # back_acc, back_loss, test_acc, test_loss = Backdoor_Evaluate(global_model, test_dataset_, loss_func, params,mask=helper.mask,pattern=helper.pattern,delta_z=None)
        back_acc, back_loss, test_acc, test_loss = Backdoor_Evaluate(global_model, test_dataset, loss_func, params,mask=helper.mask,pattern=helper.pattern,delta_z=helper.delta_z)
        print(f'\n[With Trigger] Test_Loss: {test_loss:.3f} | Test_Acc: {test_acc:.3f} | Back_Acc: {back_acc:.3f} | Back_Loss: {back_loss:.3f}')
    else:
        test_acc, test_loss = Evaluate(global_model, test_dataset, loss_func, params)
        back_acc, back_loss = None, None
        print(f'[No Trigger]  Test_Loss: {test_loss:.3f} | Test_Acc: {test_acc:.3f}')

    return test_acc, back_acc, test_loss, back_loss

def testAndSave(back_acc_list, test_acc_list, back_loss_list, test_loss_list, epoch):
    test_acc, back_acc, test_loss, back_loss = evaluate_model(
        helper.params, helper.global_model, helper.test_dataset, helper.loss_func)

    test_acc_list.append(round(test_acc, 3))
    test_loss_list.append(round(test_loss, 3))

    if back_acc is not None and back_loss is not None:
        back_acc_list.append(round(back_acc, 3))
        back_loss_list.append(round(back_loss, 3))

    helper.model_saver.save_model(helper.global_model, epoch=epoch, val_loss=test_loss)

def visualize_delta_feature(delta_z, dataset_type="cifar10"):
    """
    可视化 delta_z 在 flatten 前的特征图形态

    Args:
        delta_z (torch.Tensor): 展平后的扰动向量, shape=(D,)
        dataset_type (str): "mnist" 或 "cifar10"
    """
    dz = delta_z.detach().cpu()

    if dataset_type.lower() == "cifar10":
        # (512,) -> (512,1,1)
        dz = dz.view(512, 1, 1)

    elif dataset_type.lower() == "mnist":
        # (125,) -> 假设是 (5,5,5)
        if dz.numel() != 125:
            raise ValueError(f"MNIST 特征维度应为 125，但得到 {dz.numel()}")
        dz = dz.view(5, 5, 5)  # 这里你可以改成和你的 conv 输出一致的形状

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # 画出前几个 channel
    C = dz.shape[0]
    n_show = min(8, C)
    fig, axes = plt.subplots(1, n_show, figsize=(3*n_show, 3))
    for i in range(n_show):
        axes[i].imshow(dz[i], cmap="viridis")
        axes[i].set_title(f"Channel {i}")
        axes[i].axis("off")
    plt.show()

def run(helper: Helper):
    #for循环进行预定论次训练
    back_acc_list = []  # 存储每轮后门攻击准确率
    test_acc_list = []  # 存储每轮主任务准确率
    back_loss_list = []
    test_loss_list = []

    # 定义进度条
    tqdm_epochs = tqdm(
        range(1, helper.params.epochs + 1),
        desc="Training Progress",
        bar_format="{l_bar}{bar}| Epoch {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        leave=True,
        colour="blue"
    )

    # 创建 TriggerGenerator
    generator = TriggerGenerator(
        params=helper.params,
    )
    torch.autograd.set_detect_anomaly(True)
    for epoch in tqdm_epochs:
        run_fl_round(helper, epoch, generator)


        # 测试集,模型保存
        testAndSave(back_acc_list, test_acc_list,back_loss_list,test_loss_list,epoch)

    print("测试集准确率历史：", test_acc_list)
    print("测试集损失：", test_loss_list)
    print("后门攻击准确率历史：", back_acc_list)
    print("后门损失", back_loss_list)
    params_dict=helper.params.to_dict()
    logger.warning(utils.create_table(params_dict))

if __name__ == '__main__':
    # 读取 YAML 并创建参数对象
    params = load_params_from_yaml('config/cifar10_fed.yaml')
    # params = load_params_from_yaml('config/mnist_fed.yaml')

    # 自动选择显存最多的 GPU
    if torch.cuda.is_available():
        best_gpu = get_max_free_memory_gpu()
        print(f"GPU with the most free memory is: GPU {best_gpu}")
        params['gpu'] = best_gpu
    else:
        params['gpu'] = 'cpu'

    # 初始化
    helper = Helper(params)
    logger.warning(utils.create_table(params))
    print(helper.params.device)
    print(helper.params.backdoor_clients)
    try:
        run(helper)
    except KeyboardInterrupt:
        print('Interrupted')
    # print(params)

