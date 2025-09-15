import logging
import random
from copy import deepcopy

import hdbscan
import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
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
    # 聚合模型，计算聚合后的模型参数更新量
    Aggregation(helper.params,helper,global_model, helper.clients_model, helper.clients_update,helper.clients_his_update,helper.clients, helper.loss_func)


 # 聚合模型，计算聚合后的模型参数更新量
    Aggregation(helper.params,helper,global_model, helper.clients_model, helper.clients_update,helper.clients_his_update,helper.clients, helper.loss_func)

    # ====== 使用 g_trigger 对全局模型进行净化，消除后门 ======
    # 初始化 g_trigger 变量
    g_trigger = None

    if epoch > 0:
        # 逆向生成触发器
        mask, pattern, delta_z = generator.generate(
            model=helper.global_model,
            # model=helper.clients_model[7],
            tri_dataset = helper.test_dataset,
            attack_size = 5000,  # 用5000个噪声样本做优化
            batch_size = 128
        )
        helper.mask=mask
        helper.pattern=pattern
        helper.delta_z=delta_z

        # if delta_z is None:
        if delta_z is not None:
            g_trigger, fc_names = get_trigger_gradient_vector(
                model=helper.global_model,
                delta_z=delta_z,
                target_label=helper.params.aim_target,
                device=helper.params.device
            )
            print("Trigger gradient vector shape:", g_trigger.shape)
            print("Got feature trigger:", delta_z.shape)
            similarities = compute_fc_similarity_with_trigger(helper.clients_update, g_trigger, fc_names)

            sorted_clients = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            print("Suspicious ranking (fc layer only):")
            for cid, sim in sorted_clients:
                print(f"Client {cid}: similarity={sim:.4f}")

            # ====== 增强净化策略 ======
            # 1. 累积触发器历史信息
            if not hasattr(helper, 'trigger_history'):
                helper.trigger_history = []
            helper.trigger_history.append(g_trigger.clone())

            # 只保留最近10轮的触发器
            if len(helper.trigger_history) > 10:
                helper.trigger_history.pop(0)

            # 2. 计算累积触发器方向（加权平均）
            weights = torch.linspace(0.5, 1.0, len(helper.trigger_history)).to(helper.params.device)
            weighted_triggers = []
            for i, trigger in enumerate(helper.trigger_history):
                weighted_triggers.append(weights[i] * trigger)
            accumulated_trigger = torch.stack(weighted_triggers).mean(dim=0)

            # 3. 识别高相似度客户端并动态调整
            high_sim_clients = [cid for cid, sim in similarities.items() if sim > 0.1]
            attack_intensity = len(high_sim_clients) / len(similarities)
            print(f"[攻击强度] {len(high_sim_clients)}/{len(similarities)} 客户端可疑，强度: {attack_intensity:.2f}")

            # 使用 accumulated_trigger 净化全局模型
            if len(helper.trigger_history) > 0:
                # 仅对全连接层参数做净化
                fc_params = [p for n, p in helper.global_model.named_parameters() if "fc" in n]
                if fc_params:
                    flat_fc = torch.cat([p.detach().flatten() for p in fc_params])

                    # 使用累积触发器计算投影系数
                    alpha_acc = torch.dot(flat_fc, accumulated_trigger) / (accumulated_trigger.norm()**2 + 1e-12)
                    alpha_cur = torch.dot(flat_fc, g_trigger) / (g_trigger.norm()**2 + 1e-12)

                    # ====== 优化的净化策略 ======
                    # 1. 大幅降低基础净化强度，采用渐进式净化
                    base_ratio = min(0.15, 0.02 + attack_intensity * 0.1)  # 显著降低净化强度

                    # 2. 基于净化历史的自适应调整（更保守）
                    if hasattr(helper, 'purify_history') and len(helper.purify_history) >= 2:
                        recent_main_trend = helper.purify_history[-1]['main_acc'] - helper.purify_history[-2]['main_acc']
                        recent_bd_trend = helper.purify_history[-1]['backdoor_acc'] - helper.purify_history[-2]['backdoor_acc']

                        # 如果主任务下降但后门没改善，大幅减弱净化
                        if recent_main_trend < -0.03 and recent_bd_trend > -0.05:
                            base_ratio *= 0.5
                            print(f"[净化减弱] 主任务下降但后门未改善，减弱净化强度至 {base_ratio:.3f}")
                        # 如果后门明显上升，适度增强
                        elif recent_bd_trend > 0.08:
                            base_ratio = min(0.25, base_ratio * 1.3)
                            print(f"[净化增强] 后门攻击率上升，增强净化强度至 {base_ratio:.3f}")

                    # 3. 改进净化方向：使用正交投影而非简单减法
                    # 先备份原参数
                    original_params = [p.data.clone() for p in fc_params]

                    # 获取净化前基准性能
                    helper.global_model.eval()
                    with torch.no_grad():
                        test_loader = torch.utils.data.DataLoader(helper.test_dataset, batch_size=128, shuffle=True)
                        batch_x, batch_y = next(iter(test_loader))
                        batch_x, batch_y = batch_x.to(helper.params.device), batch_y.to(helper.params.device)
                        _, baseline_outputs = helper.global_model(batch_x)
                        baseline_acc = (baseline_outputs.argmax(dim=1) == batch_y).float().mean().item()

                    # 4. 温和的正交投影净化
                    # 归一化触发器方向
                    acc_unit = accumulated_trigger / (accumulated_trigger.norm() + 1e-12)
                    cur_unit = g_trigger / (g_trigger.norm() + 1e-12)

                    # 计算需要去除的后门分量（使用更小的系数）
                    backdoor_component_acc = torch.dot(flat_fc, acc_unit) * acc_unit * base_ratio * 0.7
                    backdoor_component_cur = torch.dot(flat_fc, cur_unit) * cur_unit * base_ratio * 0.3

                    # 应用净化
                    flat_fc_clean = flat_fc - backdoor_component_acc - backdoor_component_cur

                    start = 0
                    for p in fc_params:
                        numel = p.numel()
                        p.data.copy_(flat_fc_clean[start:start+numel].view_as(p))
                        start += numel

                    # 5. 立即评估净化效果
                    with torch.no_grad():
                        _, after_outputs = helper.global_model(batch_x)
                        main_acc = (after_outputs.argmax(dim=1) == batch_y).float().mean().item()

                        # 评估后门性能
                        backdoor_acc = 0.0
                        origin_mask = (batch_y == helper.params.origin_target)
                        if origin_mask.sum() > 0 and hasattr(helper, 'delta_z') and helper.delta_z is not None:
                            origin_samples = batch_x[origin_mask]
                            features, _ = helper.global_model(origin_samples)
                            z_adv = features + helper.delta_z.unsqueeze(0).expand_as(features)
                            _, bd_outputs = helper.global_model(features=z_adv)
                            target_labels = torch.full((origin_samples.size(0),), helper.params.aim_target, device=helper.params.device)
                            backdoor_acc = (bd_outputs.argmax(dim=1) == target_labels).float().mean().item()

                    # 6. 改进的回滚机制：相对阈值+绝对保护
                    performance_drop = baseline_acc - main_acc
                    should_rollback = (performance_drop > 0.1) or (main_acc < max(0.3, baseline_acc * 0.8))

                    if should_rollback:
                        print(f"[净化回滚] 性能下降 {performance_drop:.3f} 或绝对精度过低 {main_acc:.3f}，恢复原参数")
                        for p, orig_p in zip(fc_params, original_params):
                            p.data.copy_(orig_p)
                        main_acc = baseline_acc  # 恢复基准精度
                    else:
                        print(f"[净化成功] 投影系数: acc={alpha_acc:.4f}, cur={alpha_cur:.4f}")
                        print(f"[净化效果] 强度={base_ratio:.3f}, 主任务={main_acc:.3f}, 后门={backdoor_acc:.3f}")

                        # 记录净化历史
                        if not hasattr(helper, 'purify_history'):
                            helper.purify_history = []
                        helper.purify_history.append({
                            'epoch': epoch,
                            'alpha_acc': alpha_acc.item(),
                            'alpha_cur': alpha_cur.item(),
                            'purify_ratio': base_ratio,
                            'main_acc': main_acc,
                            'backdoor_acc': backdoor_acc,
                            'attack_intensity': attack_intensity
                        })

                        # 7. 简化的趋势监控（减少复杂的if判断）
                        if len(helper.purify_history) >= 3:
                            recent_bd_accs = [h['backdoor_acc'] for h in helper.purify_history[-3:]]
                            if np.mean(recent_bd_accs) > 0.6:  # 后门持续高企时才考虑强化净化
                                print("[警告] 后门攻击率持续较高，建议检查触发器逆向质量")
        else:
            print("No fully connected layer found for trigger mitigation.")

    # 可选：保存触发器图像用于可视化
    # generator.save_trigger(best_mask, best_pattern, prefix="reverse_trigger")

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
        test_dataset_ = deepcopy(test_dataset)
        utils.Backdoor_process(test_dataset_, params.origin_target, params.aim_target)
        back_acc, back_loss, test_acc, test_loss = Backdoor_Evaluate(global_model, test_dataset_, loss_func, params,mask=helper.mask,pattern=helper.pattern,delta_z=None)
        # back_acc, back_loss, test_acc, test_loss = Backdoor_Evaluate(global_model, test_dataset_, loss_func, params,mask=helper.mask,pattern=helper.pattern,delta_z=helper.delta_z)
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
        attack_succ_threshold=0.9,
        regularization='l1',
        init_cost=1e-3,
        lr=0.1,
        steps=100
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