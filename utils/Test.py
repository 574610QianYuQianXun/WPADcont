from copy import deepcopy
from torch.utils.data import DataLoader
from collections import defaultdict

from attack import frequency_backdoor
from clients.MaliciousClient import visualize_backdoor_samples
from utils import utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def get_attention_map(feature):
    """
    输入: 特征张量 [batch_size, channels, height, width]
    输出: 注意力图 [batch_size, height, width]
    方法: 对 channel 做平方和再开方，得到空间注意力分布
    """
    # Feature: [B, C, H, W]
    return torch.sqrt((feature ** 2).sum(1))  # -> [B, H, W]

def get_attention_map_p(feature, p=2):
    """
    输入:
        feature: 特征张量 [batch_size, channels, height, width]
        p: 缩放幂次（公式中的 p，需满足 p > 1）
    输出:
        注意力图 [batch_size, height, width]
    方法:
        对 channel 维度的激活值取绝对值的 p 次幂，再求均值（公式5）
    """
    if p <= 1:
        raise ValueError(f"p must be > 1, got {p}")

    # 计算绝对值的 p 次幂，并沿通道维度求均值
    att_map = (feature.abs().pow(p)).mean(dim=1)  # [B, H, W]
    return att_map

def visualize_attention(input_image, feature, save_path=None):
    """
    可视化单张图片的注意力图
    参数:
        input_image: 原始输入图像，[1, 1, H, W]（灰度图）
        feature: 中间层特征输出，形状 [1, 5, 5, 5] (你需要 reshape 回 feature map)
        save_path: 可选，保存路径
    """
    if input_image is None or feature is None:
        print("Warning: input_image or feature is None")
        return

    try:
        # Step 1: 计算注意力图
        att_map = get_attention_map_p(feature, p=4)  # 得到 [1, H, W]

        # Step 2: 插值到输入图像尺寸
        att_map = F.interpolate(
            att_map.unsqueeze(1),  # 添加通道维 [1, 1, H, W]
            size=input_image.shape[-2:],  # 目标尺寸（28x28）
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().detach().numpy()  # -> [28, 28]

        # Step 3: 原图处理
        input_img = input_image.squeeze().cpu().detach().numpy()  # -> [H, W]

        # Step 4: 安全的归一化
        def safe_normalize(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min < 1e-8:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        att_map = safe_normalize(att_map)
        input_img = safe_normalize(input_img)

        # Step 5: 可视化
        plt.figure(figsize=(12, 4))

        # 原图
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        plt.imshow(input_img, cmap='gray')
        plt.axis('off')

        # 注意力图
        plt.subplot(1, 3, 2)
        plt.title('Attention Map')
        plt.imshow(att_map, cmap='jet')
        plt.colorbar()
        plt.axis('off')

        # 叠加图
        plt.subplot(1, 3, 3)
        plt.title('Overlay')
        plt.imshow(input_img, cmap='gray')
        plt.imshow(att_map, cmap='jet', alpha=0.5)
        plt.axis('off')

        plt.tight_layout()

        # 保存 or 显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # 释放内存
        else:
            plt.show()

    except Exception as e:
        print(f"Error in visualize_attention: {e}")
        if 'plt' in locals():
            plt.close()

def Evaluate(model, datasets, loss_func, params):
    """
    评估模型在数据集上的性能
    """
    if model is None or datasets is None:
        raise ValueError("Model and datasets cannot be None")

    model.eval()
    model.to(params.device)

    total_loss = 0.0
    correct = 0
    total = 0

    test_loader = DataLoader(datasets, batch_size=params.bs, shuffle=False)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(params.device), target.to(params.device)

            try:
                _, output = model(data)
                loss = loss_func(output, target)

                total_loss += loss.item() * data.size(0)  # 使用data.size(0)更准确
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            except Exception as e:
                print(f"Error during evaluation batch: {e}")
                continue

    if total == 0:
        return 0.0, float('inf')

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss
# def Backdoor_Evaluate(model, dataset, loss_func, params, enable_classwise_loss=False, mask=None, pattern=None, delta_z=None):
#     """
#     评估模型的后门攻击性能
#     """
#     model.eval()
#     model.to(params.device)
#     loader = DataLoader(dataset, batch_size=params.bs, shuffle=False)
#
#     # 初始化统计变量
#     total_correct_all, total_loss_all, total_samples_all = 0, 0.0, 0
#     total_correct_backdoor, total_loss_backdoor, total_samples_backdoor = 0, 0.0, 0
#     total_entropy_sum, total_entropy_count = 0.0, 0
#
#     if enable_classwise_loss:
#         class_loss_sum = defaultdict(float)
#         class_sample_count = defaultdict(int)
#
#     with torch.no_grad():
#         for data, target in loader:
#             data, target = data.to(params.device), target.to(params.device)
#
#             features, outputs = model(data)
#
#             # 计算softmax概率和熵
#             prob = F.softmax(outputs, dim=1)
#             entropy_per_sample = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
#             total_entropy_sum += entropy_per_sample.sum().item()
#             total_entropy_count += entropy_per_sample.size(0)
#
#             # 计算整体性能
#             loss = loss_func(outputs, target)
#             _, predicted = torch.max(outputs, 1)
#
#             total_samples_all += target.size(0)
#             total_correct_all += (predicted == target).sum().item()
#             total_loss_all += loss.item() * target.size(0)
#
#             # 按类别统计损失
#             if enable_classwise_loss:
#                 for i in range(len(target)):
#                     label = target[i].item()
#                     sample_loss = loss_func(outputs[i:i+1], target[i:i+1]).item()
#                     class_loss_sum[label] += sample_loss
#                     class_sample_count[label] += 1
#
#             # 后门攻击评估
#             origin_mask = (target == params.origin_target)
#             if origin_mask.sum() > 0:
#                 data_masked = data[origin_mask]
#                 target_masked = target[origin_mask]
#
#                 # 重新前向传播获取特征
#                 features_masked, output_masked = model(data_masked)
#
#                 # 如果有特征触发器，添加扰动
#                 if delta_z is not None:
#                     # 确保维度匹配
#                     if features_masked.shape[1] != delta_z.shape[0]:
#                         print(f"Warning: Feature dimension mismatch in backdoor eval. "
#                               f"Expected {delta_z.shape[0]}, got {features_masked.shape[1]}")
#                     else:
#                         new_features = features_masked + delta_z.unsqueeze(0).expand_as(features_masked)
#                         _, output_masked = model(features=new_features)
#
#                 # 计算后门攻击成功率
#                 target_aim = torch.full_like(target_masked, fill_value=params.aim_target)
#                 loss_masked = loss_func(output_masked, target_aim)
#
#                 _, predicted_masked = torch.max(output_masked, 1)
#                 total_samples_backdoor += target_masked.size(0)
#                 total_loss_backdoor += loss_masked.item() * target_masked.size(0)
#                 total_correct_backdoor += (predicted_masked == params.aim_target).sum().item()
#
#     # 安全除法函数
#     def safe_div(numerator, denominator):
#         return numerator / denominator if denominator > 0 else 0.0
#
#     # 计算最终指标
#     acc_all = 100.0 * safe_div(total_correct_all, total_samples_all)
#     acc_backdoor = 100.0 * safe_div(total_correct_backdoor, total_samples_backdoor)
#     loss_all = safe_div(total_loss_all, total_samples_all)
#     loss_backdoor = safe_div(total_loss_backdoor, total_samples_backdoor)
#     avg_entropy = safe_div(total_entropy_sum, total_entropy_count)
#
#     # 打印结果
#     print(f"[平均熵] Avg entropy of softmax probabilities: {avg_entropy:.4f}")
#
#     if enable_classwise_loss and class_sample_count:
#         print("\n[每类平均损失统计]")
#         for cls in sorted(class_loss_sum.keys()):
#             avg_loss = safe_div(class_loss_sum[cls], class_sample_count[cls])
#             print(f"类别 {cls:2d} | 样本数: {class_sample_count[cls]:4d} | 平均损失: {avg_loss:.4f}")
#
#     return acc_backdoor, loss_backdoor, acc_all, loss_all
def test_pure_delta_z(model, delta_z, params, loss_func, num_samples=100):
    """
    测试只把 delta_z 输入模型，计算结果是目标标签的概率

    Args:
        model: 模型
        delta_z: 特征触发器
        params: 参数对象（包含device, aim_target等）
        loss_func: 损失函数
        num_samples: 测试样本数量

    Returns:
        acc: 预测为目标标签的准确率
        avg_prob: 预测为目标标签的平均概率
    """
    import torch
    model.eval()
    model.to(params.device)

    if delta_z is None:
        print("[Pure Delta_z 测试] delta_z 为空，跳过测试")
        return 0.0, 0.0

    # 确保 delta_z 是正确的张量
    if torch.is_tensor(delta_z):
        dz = delta_z.to(params.device)
    else:
        dz = torch.tensor(delta_z, dtype=torch.float32, device=params.device)

    if dz.dim() == 2 and dz.size(0) == 1:
        dz = dz.view(-1)
    if dz.dim() != 1:
        dz = dz.view(-1)

    # 复制 delta_z 成 batch
    batch_delta_z = dz.unsqueeze(0).expand(num_samples, -1)  # [num_samples, feat_dim]

    total_correct = 0
    total_prob_sum = 0.0

    with torch.no_grad():
        try:
            # 直接将 delta_z 作为特征输入模型
            _, outputs = model(features=batch_delta_z)

            # 计算 softmax 概率
            probs = torch.softmax(outputs, dim=1)
            target_probs = probs[:, params.aim_target]

            # 统计预测为目标标签的数量
            _, predicted = torch.max(outputs, 1)
            total_correct = (predicted == params.aim_target).sum().item()
            total_prob_sum = target_probs.sum().item()

        except Exception as e:
            print(f"[Pure Delta_z 测试] 模型不支持 features= 参数或其他错误: {e}")
            return 0.0, 0.0

    acc = 100.0 * total_correct / num_samples
    avg_prob = total_prob_sum / num_samples

    return acc, avg_prob

def test_delta_z_on_noise(model, delta_z, params, loss_func, num_samples=100):
    """
    使用噪声数据，在噪声数据获得的特征上加入 delta_z 再输入模型，
    计算结果为目标标签的概率

    Args:
        model: 模型
        delta_z: 特征触发器
        params: 参数对象
        loss_func: 损失函数
        num_samples: 测试样本数量

    Returns:
        acc: 预测为目标标签的准确率
        avg_prob: 预测为目标标签的平均概率
    """
    import torch
    model.eval()
    model.to(params.device)

    if delta_z is None:
        print("[Noise+Delta_z 测试] delta_z 为空，跳过测试")
        return 0.0, 0.0

    # 确保 delta_z 是正确的张量
    if torch.is_tensor(delta_z):
        dz = delta_z.to(params.device)
    else:
        dz = torch.tensor(delta_z, dtype=torch.float32, device=params.device)

    if dz.dim() == 2 and dz.size(0) == 1:
        dz = dz.view(-1)
    if dz.dim() != 1:
        dz = dz.view(-1)

    # 生成噪声数据（假设输入尺寸与数据集匹配）
    # 根据 params 推断输入尺寸
    # 默认使用 MNIST 尺寸 (1, 28, 28) 或 CIFAR (3, 32, 32)
    # 这里需要根据实际情况调整
    # noise_data = torch.randn(num_samples, 1, 28, 28, device=params.device)
    if params.task == 'MNIST':
        noise_data = torch.randn(num_samples, 1, 28, 28, device=params.device)
    elif params.task == 'CIFAR10' or params.task == 'CIFAR100':
        noise_data = torch.randn(num_samples, 3, 32, 32, device=params.device)
    else:
        noise_data = torch.randn(num_samples, 3, 64, 64, device=params.device)

    total_correct = 0
    total_prob_sum = 0.0

    with torch.no_grad():
        try:
            # 1. 从噪声数据提取特征
            features, _ = model(noise_data)

            # 展平特征
            if features.dim() > 2:
                features_flat = features.view(features.size(0), -1)
            else:
                features_flat = features

            # 检查维度匹配
            if features_flat.size(1) != dz.shape[0]:
                print(f"[Noise+Delta_z 测试] 特征维度不匹配: {features_flat.size(1)} vs {dz.shape[0]}")
                return 0.0, 0.0

            # 2. 在特征上加入 delta_z
            new_features = features_flat + dz.unsqueeze(0).expand_as(features_flat)

            # 3. 将新特征输入模型
            _, outputs = model(features=new_features)

            # 计算 softmax 概率
            probs = torch.softmax(outputs, dim=1)
            target_probs = probs[:, params.aim_target]

            # 统计预测为目标标签的数量
            _, predicted = torch.max(outputs, 1)
            total_correct = (predicted == params.aim_target).sum().item()
            total_prob_sum = target_probs.sum().item()

        except Exception as e:
            print(f"[Noise+Delta_z 测试] 发生错误: {e}")
            return 0.0, 0.0

    acc = 100.0 * total_correct / num_samples
    avg_prob = total_prob_sum / num_samples

    return acc, avg_prob

def Backdoor_Evaluate(model, dataset, loss_func, params,
                      enable_classwise_loss=False, mask=None, pattern=None, delta_z=None):
    """
    评估模型的后门攻击性能，同时收集 features 与 labels。

    逻辑顺序（严格）：
      1) 在原始测试集(dataset)上评估主任务精度（不打印，只计算）
      2) 使用原始测试集测试 delta_z 的效果（如果提供 delta_z），调用封装函数并打印结果
      3) 后门测试：deepcopy 一份测试集，调用 utils.Backdoor_process 将所有样本加入触发器并改为目标标签，评估后门精度（不打印，只计算）
      4) 两个新测试：纯delta_z测试 和 噪声+delta_z测试，调用封装函数并打印结果
      5) 相似度计算（已有）
      6) 返回 (acc_backdoor, loss_backdoor, acc_all, loss_all)
    """

    model.eval()
    model.to(params.device)

    # -------------------------
    # 1) 主任务（干净测试集）| 不打印，只计算
    # -------------------------
    loader_clean = DataLoader(dataset, batch_size=params.bs, shuffle=False)

    total_correct_all, total_loss_all, total_samples_all = 0, 0.0, 0
    total_entropy_sum, total_entropy_count = 0.0, 0

    if enable_classwise_loss:
        class_loss_sum = defaultdict(float)
        class_sample_count = defaultdict(int)

    # 收集特征用于可视化
    collected_features = []
    collected_labels = []
    collected_features_triggered = []
    collected_labels_triggered = []
    collected_features_delta_z = []
    collected_labels_delta_z = []

    with torch.no_grad():
        for data, target in loader_clean:
            data, target = data.to(params.device), target.to(params.device)

            features, outputs = model(data)

            # 展平特征
            if features.dim() > 2:
                features_flat = features.view(features.size(0), -1)
            else:
                features_flat = features

            # 收集原始特征（用于后续可视化）
            collected_features.append(features_flat.cpu().numpy())
            collected_labels.append(target.cpu().numpy())

            # 计算 softmax 概率与熵（保留）
            prob = F.softmax(outputs, dim=1)
            entropy_per_sample = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
            total_entropy_sum += float(entropy_per_sample.sum().item())
            total_entropy_count += int(entropy_per_sample.size(0))

            # 计算整体性能（干净）
            loss = loss_func(outputs, target)
            _, predicted = torch.max(outputs, 1)

            total_samples_all += target.size(0)
            total_correct_all += int((predicted == target).sum().item())
            total_loss_all += float(loss.item() * target.size(0))

            # 按类别统计损失（可选）
            if enable_classwise_loss:
                for i in range(len(target)):
                    label = int(target[i].item())
                    sample_loss = float(loss_func(outputs[i:i+1], target[i:i+1]).item())
                    class_loss_sum[label] += sample_loss
                    class_sample_count[label] += 1

    # 计算主任务指标（但不打印）
    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    acc_all = 100.0 * safe_div(total_correct_all, total_samples_all)
    loss_all = safe_div(total_loss_all, total_samples_all)
    avg_entropy = safe_div(total_entropy_sum, total_entropy_count)

    # 打印平均熵（保留原来的信息）
    print(f"[平均熵] Avg entropy of softmax probabilities: {avg_entropy:.4f}")

    # 如果开启 classwise loss，保留打印（这是你之前保留的行为）
    if enable_classwise_loss and class_sample_count:
        print("\n[每类平均损失统计]")
        for cls in sorted(class_loss_sum.keys()):
            avg_loss = safe_div(class_loss_sum[cls], class_sample_count[cls])
            print(f"类别 {cls:2d} | 样本数: {class_sample_count[cls]:4d} | 平均损失: {avg_loss:.4f}")

    # -------------------------
    # 2) 使用干净测试集测试 delta_z 的效果（调用封装函数）
    # -------------------------
    if delta_z is not None:
        loader_clean_for_dz = DataLoader(dataset, batch_size=params.bs, shuffle=False)
        total_correct_dz, total_loss_dz, total_samples_dz = 0, 0.0, 0

        # ensure delta_z is a torch tensor on correct device and 1D
        if torch.is_tensor(delta_z):
            dz = delta_z.to(params.device)
        else:
            dz = torch.tensor(delta_z, dtype=torch.float32, device=params.device)

        if dz.dim() == 2 and dz.size(0) == 1:
            dz = dz.view(-1)
        if dz.dim() != 1:
            print("Warning: delta_z should be 1D vector. Attempting to flatten.")
            dz = dz.view(-1)

        with torch.no_grad():
            for data, target in loader_clean_for_dz:
                data, target = data.to(params.device), target.to(params.device)
                features, outputs = model(data)

                # 展平特征
                if features.dim() > 2:
                    features_flat = features.view(features.size(0), -1)
                else:
                    features_flat = features

                # 检查维度匹配
                if features_flat.size(1) != dz.shape[0]:
                    print(f"Warning: Feature dimension mismatch for delta_z test. "
                          f"Expected {features_flat.size(1)}, got {dz.shape[0]}. Skipping delta_z test.")
                    total_samples_dz = 0
                    break

                # 生成带 delta 的特征并 forward
                new_features = features_flat + dz.unsqueeze(0).expand_as(features_flat)

                # 收集delta_z测试时的特征
                collected_features_delta_z.append(new_features.cpu().numpy())
                collected_labels_delta_z.append(target.cpu().numpy())

                try:
                    _, outputs_dz = model(features=new_features)
                except TypeError:
                    print("Warning: model does not accept 'features=' kwarg. delta_z test skipped for this model.")
                    total_samples_dz = 0
                    break

                # 目标标签为 params.aim_target
                target_aim = torch.full_like(target, fill_value=params.aim_target)
                loss_dz = loss_func(outputs_dz, target_aim)

                _, pred_dz = torch.max(outputs_dz, 1)
                total_samples_dz += target.size(0)
                total_loss_dz += float(loss_dz.item() * target.size(0))
                total_correct_dz += int((pred_dz == params.aim_target).sum().item())

        # 计算并打印 delta_z 结果
        if total_samples_dz > 0:
            acc_dz = 100.0 * safe_div(total_correct_dz, total_samples_dz)
            loss_dz_avg = safe_div(total_loss_dz, total_samples_dz)
            print(f"[delta_z 测试结果] ASR (using delta_z on clean features): {acc_dz:.2f}% | Loss: {loss_dz_avg:.4f}")
        else:
            print("[delta_z 测试] 没有执行（样本数为0或模型不支持 features= 参数）。")

    # -------------------------
    # 3) 后门测试：deepcopy 测试集，调用 utils.Backdoor_process，评估后门精度（不打印，只计算）
    # -------------------------
    test_dataset_bd = deepcopy(dataset)
    try:
        if params.attack_type== 'dct':
            print("dct")
            frequency_backdoor(
                train_set=test_dataset_bd,
                aim_target=params.aim_target,
                poison_rate=1.0,
            )
        elif params.attack_type == 'sadba':
            print(f"SADBA Backdoor Testing...")
            rec_pos = getattr(params, 'sadba_recorded_positions', set())
            # 2. 调用处理函数
            # utils.SADBA_Backdoor_process(
            #     test_dataset=test_dataset_bd,
            #     aim_target=params.aim_target,
            #     recorded_positions=rec_pos,
            #     task=params.task
            # )
            utils.SADBA_Backdoor_process(
                test_dataset=test_dataset_bd,
                aim_target=params.aim_target,
                task=params.task,
                current_model=model,  # 当前轮的全局模型
                device=params.device,
            )
        else:
            print("backdoor")
            utils.Backdoor_process(test_dataset_bd, params.origin_target, params.aim_target,params.task)
    except Exception as e:
        print("Error when calling utils.Backdoor_process:", e)
        total_correct_backdoor, total_loss_backdoor, total_samples_backdoor = 0, 0.0, 0
        acc_backdoor, loss_backdoor = 0.0, 0.0

    # visualize_backdoor_samples(test_dataset_bd, n_samples=16, nrow=4)
    loader_bd = DataLoader(test_dataset_bd, batch_size=params.bs, shuffle=False)
    total_correct_backdoor, total_loss_backdoor, total_samples_backdoor = 0, 0.0, 0

    with torch.no_grad():
        for data, target in loader_bd:
            data, target = data.to(params.device), target.to(params.device)
            features, outputs_bd = model(data)

            # 展平特征并收集触发后的特征
            if features.dim() > 2:
                features_flat = features.view(features.size(0), -1)
            else:
                features_flat = features

            # 收集触发后的特征
            collected_features_triggered.append(features_flat.cpu().numpy())
            collected_labels_triggered.append(target.cpu().numpy())

            loss_bd = loss_func(outputs_bd, target)
            _, pred_bd = torch.max(outputs_bd, 1)

            total_samples_backdoor += target.size(0)
            total_loss_backdoor += float(loss_bd.item() * target.size(0))
            total_correct_backdoor += int((pred_bd == target).sum().item())

    acc_backdoor = 100.0 * safe_div(total_correct_backdoor, total_samples_backdoor)
    loss_backdoor = safe_div(total_loss_backdoor, total_samples_backdoor)

    # -------------------------
    # 4) 两个新测试：纯delta_z测试 和 噪声+delta_z测试
    # -------------------------
    if delta_z is not None:
        # 测试1：纯 delta_z 输入
        acc_pure_dz, prob_pure_dz = test_pure_delta_z(model, delta_z, params, loss_func, num_samples=1000)
        print(f"[纯Delta_z测试] 预测为目标标签的准确率: {acc_pure_dz:.2f}% | 平均概率: {prob_pure_dz:.4f}")

        # 测试2：噪声 + delta_z
        acc_noise_dz, prob_noise_dz = test_delta_z_on_noise(model, delta_z, params, loss_func, num_samples=1000)
        print(f"[噪声+Delta_z测试] 预测为目标标签的准确率: {acc_noise_dz:.2f}% | 平均概率: {prob_noise_dz:.4f}")

    # -------------------------
    # 5) 相似度计算和可视化（保留原有逻辑）
    # -------------------------
    try:
        trigger_features, trigger_names, trigger_similarities = extract_trigger_features(
            model=model,
            params=params,
            delta_z=delta_z
        )

        features_all = np.concatenate(collected_features, axis=0) if collected_features else np.zeros((0, 0))
        labels_all = np.concatenate(collected_labels, axis=0) if collected_labels else np.zeros((0,))
        features_dict = {'features': features_all, 'labels': labels_all}

        collected_features_triggered = []
        collected_labels_triggered = []
        collected_features_delta_z = []
        collected_labels_delta_z = []

        if collected_features_triggered:
            features_triggered = np.concatenate(collected_features_triggered, axis=0)
            labels_triggered = np.concatenate(collected_labels_triggered, axis=0)
            features_dict['features_triggered'] = features_triggered
            features_dict['labels_triggered'] = labels_triggered

        if collected_features_delta_z:
            features_delta_z = np.concatenate(collected_features_delta_z, axis=0)
            labels_delta_z = np.concatenate(collected_labels_delta_z, axis=0)
            features_dict['features_delta_z'] = features_delta_z
            features_dict['labels_delta_z'] = labels_delta_z

        # 添加触发器图像特征到可视化字典
        if trigger_features:
            features_dict['trigger_features'] = np.vstack(trigger_features)
            features_dict['trigger_names'] = trigger_names
            features_dict['trigger_similarities'] = trigger_similarities

        # visualize_features_tsne(features_dict, delta_z=delta_z, save_path="tsne.png")

    except Exception:
        # 可视化失败无需中断流程
        pass

    # 返回后门准确率、后门损失、干净准确率、干净损失
    return acc_backdoor, loss_backdoor, acc_all, loss_all



def extract_trigger_features(model, params, delta_z=None, trigger_paths=None):
    """
    提取触发器图像的特征，并计算与 delta_z 的相似度

    Args:
        model: 已加载的模型，需支持输出中间特征 (features, logits)
        params: 含有 device 信息的参数对象
        delta_z: 可选，用于计算触发器特征与净化方向的相似度
        trigger_paths: [(path, name), ...] 格式的触发器列表

    Returns:
        trigger_features: list[np.ndarray]
        trigger_names: list[str]
        trigger_similarities: list[float]
    """
    if trigger_paths is None:
        if params.task == 'MNIST':
            trigger_paths = [("demo/trigger/mnist_trigger.png", "MNIST Trigger")]
        else:
            trigger_paths = [("demo/trigger/cifar_trigger.png", "CIFAR Trigger")]

    trigger_features = []
    trigger_names = []
    trigger_similarities = []

    for trigger_path, trigger_name in trigger_paths:
        if not os.path.exists(trigger_path):
            print(f"触发器图像文件不存在: {trigger_path}")
            continue

        try:
            # 根据任务类型决定颜色模式
            if "MNIST" in trigger_name:
                img = Image.open(trigger_path).convert("L")
            else:
                img = Image.open(trigger_path).convert("RGB")

            arr = np.array(img)

            # 转 tensor 并调整维度
            if len(arr.shape) == 2:  # 灰度图
                tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
            elif len(arr.shape) == 3:  # 彩色图
                tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"不支持的图像维度: {arr.shape}")

            tensor = tensor / 255.0
            tensor = tensor.to(params.device)

            # 提取特征
            with torch.no_grad():
                features, _ = model(tensor)
                if features.dim() > 2:
                    features_flat = features.view(features.size(0), -1)
                else:
                    features_flat = features

                trigger_features.append(features_flat.cpu().numpy())
                trigger_names.append(trigger_name)

                mag = torch.norm(features_flat, p=2).item()
                print(f"[触发器特征强度] {trigger_name}: {mag:.4f}")

                # 相似度计算
                if delta_z is not None:
                    dz_tensor = delta_z.to(params.device) if torch.is_tensor(delta_z) else torch.tensor(delta_z, device=params.device)
                    if dz_tensor.dim() == 2 and dz_tensor.size(0) == 1:
                        dz_tensor = dz_tensor.view(-1)

                    # 在 extract_trigger_features 函数中，找到这部分：
                    if features_flat.size(1) == dz_tensor.size(0):
                        # ✅ 原有的余弦相似度
                        cos_sim = F.cosine_similarity(features_flat, dz_tensor.unsqueeze(0), dim=1)
                        similarity = float(cos_sim.item())
                        trigger_similarities.append(similarity)
                        print(f"[触发器相似度] {trigger_name} vs delta_z: {similarity:.4f}")

                        # ⭐ 新增：L2距离（绝对差异）
                        l2_dist = torch.norm(features_flat - dz_tensor.unsqueeze(0), p=2, dim=1)
                        print(f"  ├─ L2距离: {l2_dist.item():.4f}")

                        # ⭐ 新增：幅度比（检查是否同一量级）
                        mag_trigger = torch.norm(features_flat, p=2, dim=1)
                        mag_delta_z = torch.norm(dz_tensor, p=2)
                        magnitude_ratio = mag_trigger / mag_delta_z
                        print(f"  ├─ 幅度比 (trigger/delta_z): {magnitude_ratio.item():.4f}")

                        # ⭐ 新增：内积（检查投影长度）
                        inner_product = torch.sum(features_flat * dz_tensor.unsqueeze(0), dim=1)
                        print(f"  └─ 内积 (投影强度): {inner_product.item():.4f}")

                    else:
                        trigger_similarities.append(0.0)
                        print(f"[触发器相似度] {trigger_name}: 维度不匹配，跳过计算")
                else:
                    trigger_similarities.append(0.0)

        except Exception as e:
            print(f"处理触发器图像 {trigger_path} 时出错: {e}")

    return trigger_features, trigger_names, trigger_similarities


def visualize_features_tsne(features_dict, delta_z=None,
                            perplexity=30, n_iter=1500, random_state=42,
                            save_path=None, show_plot=True):
    """
    使用 t-SNE 可视化特征分布（2D）：
      - 主任务特征：各个类别不同颜色
      - 触发后的特征：蓝色x号标记
      - delta_z测试特征：绿色小三角形标记
      - 触发器图像特征：紫色菱形标记
      - delta_z向量：红色星号
    """

    features = features_dict['features']
    labels = features_dict['labels']

    if features.size == 0:
        print("Warning: No features to visualize")
        return

    stack_list = [features]
    tag_list = labels.tolist()

    # 加入触发后的特征 (标记为-2)
    if 'features_triggered' in features_dict and features_dict['features_triggered'].size > 0:
        triggered_features = features_dict['features_triggered']
        triggered_labels = features_dict['labels_triggered']
        stack_list.append(triggered_features)
        # 使用-2标记触发后的特征
        tag_list.extend([-2] * len(triggered_labels))

    # 加入delta_z测试时的特征 (标记为-3)
    if 'features_delta_z' in features_dict and features_dict['features_delta_z'].size > 0:
        delta_z_features = features_dict['features_delta_z']
        delta_z_labels = features_dict['labels_delta_z']
        stack_list.append(delta_z_features)
        # 使用-3标记delta_z测试特征
        tag_list.extend([-3] * len(delta_z_labels))

    # 加入触发器图像特征 (标记为-4)
    if 'trigger_features' in features_dict and features_dict['trigger_features'].size > 0:
        trigger_features = features_dict['trigger_features']
        trigger_names = features_dict.get('trigger_names', [])
        stack_list.append(trigger_features)
        # 使用-4标记触发器图像特征
        tag_list.extend([-4] * len(trigger_features))

    # 加入 delta_z 向量本身 (标记为-1)
    if delta_z is not None:
        dz = delta_z.cpu().numpy() if torch.is_tensor(delta_z) else delta_z
        dz = dz.reshape(1, -1)
        stack_list.append(dz)
        tag_list.append(-1)

    X = np.vstack(stack_list)

    # 检查特征数量
    if X.shape[0] < 2:
        print("Warning: Not enough samples for t-SNE visualization")
        return

    # 调整perplexity以适应样本数量
    actual_perplexity = min(perplexity, (X.shape[0] - 1) // 3)
    if actual_perplexity < 5:
        actual_perplexity = 5

    # 生成2D的t-SNE嵌入
    # tsne_2d = TSNE(n_components=2, perplexity=actual_perplexity,n_iter=n_iter, random_state=random_state, init='pca')
    tsne_2d = TSNE(n_components=2, perplexity=actual_perplexity, random_state=random_state, init='pca')
    X_emb_2d = tsne_2d.fit_transform(X)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 获取正常类别（标签>=0）
    unique_labels = sorted(set(l for l in tag_list if l >= 0))
    palette = sns.color_palette("tab10", len(unique_labels))

    # 画主任务特征（正常类别）
    for i, lab in enumerate(unique_labels):
        idx = [j for j, t in enumerate(tag_list) if t == lab]
        if len(idx) == 0:
            continue
        if lab == 7:  # 第8类（标签=7）固定为黑色
            ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1],
                      label=f"{lab}", s=20, alpha=0.7, color='black')
        else:
            ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1],
                      label=f"{lab}", s=20, alpha=0.7, color=palette[i % len(palette)])

    # 画触发后的特征（蓝色x号，减小大小）
    triggered_idx = [j for j, t in enumerate(tag_list) if t == -2]
    if triggered_idx:
        ax.scatter(X_emb_2d[triggered_idx, 0], X_emb_2d[triggered_idx, 1],
                  label="Backdoor Features", s=40, c='blue', marker='x', alpha=0.8, linewidths=2)

    # 画delta_z测试时的特征（绿色小三角形）
    delta_z_test_idx = [j for j, t in enumerate(tag_list) if t == -3]
    if delta_z_test_idx:
        ax.scatter(X_emb_2d[delta_z_test_idx, 0], X_emb_2d[delta_z_test_idx, 1],
                  label="Delta_z Test Features", s=30, c='green', marker='^', alpha=0.8)

    # 画触发器图像特征（紫色菱形）
    trigger_img_idx = [j for j, t in enumerate(tag_list) if t == -4]
    if trigger_img_idx:
        trigger_names = features_dict.get('trigger_names', [])
        trigger_similarities = features_dict.get('trigger_similarities', [])

        # 为每个触发器图像特征添加标签和相似度信息
        for k, idx in enumerate(trigger_img_idx):
            name = trigger_names[k] if k < len(trigger_names) else f"Trigger {k+1}"
            similarity = trigger_similarities[k] if k < len(trigger_similarities) else 0.0
            label_text = f"{name} (sim: {similarity:.3f})" if similarity != 0.0 else name

            ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1],
                      label=label_text, s=120, c='purple', marker='D',
                      alpha=0.9, edgecolors='black', linewidth=1)

    # 画 delta_z 向量本身（红色星号）
    dz_idx = [j for j, t in enumerate(tag_list) if t == -1]
    if dz_idx:
        ax.scatter(X_emb_2d[dz_idx, 0], X_emb_2d[dz_idx, 1],
                  label="Delta_z Vector", s=150, c='red', marker='*',
                  edgecolors='black', linewidth=1)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title("t-SNE Feature Visualization")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()
