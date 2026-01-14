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
        elif params.attack_type == 'a3fl':
            print("A3FL Backdoor Testing... (inject in eval loop)")
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

    import torchvision.transforms.functional as TF
    from torchvision.utils import make_grid
    vis_a3fl_done = False
    with torch.no_grad():
        for data, target in loader_bd:
            data, target = data.to(params.device), target.to(params.device)
            if params.attack_type == 'a3fl':
                if hasattr(params, "a3fl_patch") and params.a3fl_patch is not None:
                    patch = params.a3fl_patch.to(params.device)  # [C,H,W]，已经带了 mask
                    # 1. 整批加补丁
                    data = torch.clamp(data + patch, 0.0, 1.0)
                    # 2. 整批标签全改成目标类
                    target[:] = params.aim_target
                    # ---- 2) 只在第一批上可视化后门图像 ----
                    if not vis_a3fl_done:
                        n_show = min(16, data.size(0))  # 比如看前 16 张
                        grid = make_grid(
                            data[:n_show].cpu(),  # 搬到 CPU
                            nrow=4,
                            normalize=True,
                            scale_each=True
                        )
                        plt.figure(figsize=(6, 6))
                        plt.imshow(TF.to_pil_image(grid))
                        plt.title("A3FL poisoned test samples")
                        plt.axis("off")
                        plt.show()

                        vis_a3fl_done = True

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
    # 3.5) 新增：计算 (x+trigger) 特征 - x 特征，并收集特征差
    # 不改动旧逻辑，仅新增这一段
    # -------------------------
    collected_feature_diffs = []          # 每个样本的特征差 (triggered - clean)
    collected_feature_diffs_labels = []   # 可选：存一下干净标签，方便后续分析

    # 保证一一对应
    if len(test_dataset_bd) != len(dataset):
        print(f"[Warn] len(test_dataset_bd)={len(test_dataset_bd)} != len(dataset)={len(dataset)}; "
              "feature diff will be computed on the min length by zip().")

    loader_clean = DataLoader(dataset, batch_size=params.bs, shuffle=False)
    loader_poison = DataLoader(test_dataset_bd, batch_size=params.bs, shuffle=False)

    model.eval()
    with torch.no_grad():
        for (data_clean, target_clean), (data_bd_pair, target_bd_pair) in zip(loader_clean, loader_poison):
            data_clean = data_clean.to(params.device)
            data_bd_pair = data_bd_pair.to(params.device)
            target_clean = target_clean.to(params.device)

            # 注意：A3FL 的触发器是在 eval loop 里注入的，不一定真的写进了 test_dataset_bd
            # 所以这里也做同样的注入，保证 data_bd_pair 真的是带触发器的输入
            if params.attack_type == 'a3fl':
                if hasattr(params, "a3fl_patch") and params.a3fl_patch is not None:
                    patch = params.a3fl_patch.to(params.device)
                    data_bd_pair = torch.clamp(data_bd_pair + patch, 0.0, 1.0)

            # 分别提特征
            feat_clean, _ = model(data_clean)
            feat_bd, _ = model(data_bd_pair)

            # 展平特征
            if feat_clean.dim() > 2:
                feat_clean = feat_clean.view(feat_clean.size(0), -1)
            if feat_bd.dim() > 2:
                feat_bd = feat_bd.view(feat_bd.size(0), -1)

            # 计算特征差：triggered - clean
            feat_diff = feat_bd - feat_clean

            # 收集
            collected_feature_diffs.append(feat_diff.detach().cpu())
            collected_feature_diffs_labels.append(target_clean.detach().cpu())

    # 拼成一个整体张量 / numpy，后续你想算均值方向 v_delta、余弦分布都方便
    feature_diffs_tensor = torch.cat(collected_feature_diffs, dim=0)   # [N, D]
    feature_diffs_np = feature_diffs_tensor.numpy()

    # labels_clean_tensor = torch.cat(collected_feature_diffs_labels, dim=0)  # [N]
    # labels_clean_np = labels_clean_tensor.numpy()

    # feature_diffs_tensor: [N, D] torch tensor on CPU
    # mean_diff = feature_diffs_tensor.mean(dim=0)  # [D]
    # 或 numpy:
    # mean_diff = feature_diffs_np.mean(axis=0)    # [D]
    # feature_diffs_tensor: [N, D] torch tensor on CPU
    mean_diff = feature_diffs_tensor.mean(dim=0)  # [D]

    # -------------------------
    # 新增：计算 mean_diff 与 delta_z 的余弦相似度
    # -------------------------
    if delta_z is not None:
        # 统一成 torch 张量并放到 CPU
        if torch.is_tensor(delta_z):
            dz = delta_z.detach().cpu().float().view(-1)
        else:
            dz = torch.tensor(delta_z, dtype=torch.float32).view(-1)

        md = mean_diff.detach().cpu().float().view(-1)

        # 维度检查
        if md.numel() != dz.numel():
            print(f"[Warn] Dimension mismatch: mean_diff dim={md.numel()} vs delta_z dim={dz.numel()}. "
                  "Skip cosine similarity.")
        else:
            md_norm = torch.norm(md, p=2)
            dz_norm = torch.norm(dz, p=2)

            if md_norm.item() < 1e-12 or dz_norm.item() < 1e-12:
                print(f"[Warn] Zero-norm vector encountered: ||mean_diff||={md_norm.item():.3e}, "
                      f"||delta_z||={dz_norm.item():.3e}. Skip cosine similarity.")
            else:
                cos_sim = torch.dot(md, dz) / (md_norm * dz_norm)
                print(f"[Info] Cosine(mean_diff, delta_z) = {cos_sim.item():.6f}")
    else:
        print("[Info] delta_z not found or is None; skip cosine(mean_diff, delta_z).")

    print(f"[Info] Collected feature diffs: {feature_diffs_tensor.shape}")


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

        visualize_features_tsne(features_dict, delta_z=delta_z, mean_diff=mean_diff, save_path="tsne.png")

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


# def visualize_features_tsne(features_dict, delta_z=None,mean_diff=None,
#                             perplexity=30, n_iter=1500, random_state=42,
#                             save_path=None, show_plot=True):
#     """
#     使用 t-SNE 可视化特征分布（2D）：
#       - 主任务特征：各个类别不同颜色
#       - 触发后的特征：蓝色x号标记
#       - delta_z测试特征：绿色小三角形标记
#       - 触发器图像特征：紫色菱形标记
#       - delta_z向量：红色星号
#     """
#
#     features = features_dict['features']
#     labels = features_dict['labels']
#
#     if features.size == 0:
#         print("Warning: No features to visualize")
#         return
#
#     stack_list = [features]
#     tag_list = labels.tolist()
#
#     # 加入触发后的特征 (标记为-2)
#     if 'features_triggered' in features_dict and features_dict['features_triggered'].size > 0:
#         triggered_features = features_dict['features_triggered']
#         triggered_labels = features_dict['labels_triggered']
#         stack_list.append(triggered_features)
#         # 使用-2标记触发后的特征
#         tag_list.extend([-2] * len(triggered_labels))
#
#     # 加入delta_z测试时的特征 (标记为-3)
#     if 'features_delta_z' in features_dict and features_dict['features_delta_z'].size > 0:
#         delta_z_features = features_dict['features_delta_z']
#         delta_z_labels = features_dict['labels_delta_z']
#         stack_list.append(delta_z_features)
#         # 使用-3标记delta_z测试特征
#         tag_list.extend([-3] * len(delta_z_labels))
#
#     # 加入触发器图像特征 (标记为-4)
#     if 'trigger_features' in features_dict and features_dict['trigger_features'].size > 0:
#         trigger_features = features_dict['trigger_features']
#         trigger_names = features_dict.get('trigger_names', [])
#         stack_list.append(trigger_features)
#         # 使用-4标记触发器图像特征
#         tag_list.extend([-4] * len(trigger_features))
#
#     # 加入 mean feature diff 向量 (标记为-6)
#     if mean_diff is not None:
#         md = mean_diff.cpu().numpy() if torch.is_tensor(mean_diff) else mean_diff
#         md = md.reshape(1, -1)
#         stack_list.append(md)
#         tag_list.append(-6)
#
#     # 加入 delta_z 向量本身 (标记为-1)
#     if delta_z is not None:
#         dz = delta_z.cpu().numpy() if torch.is_tensor(delta_z) else delta_z
#         dz = dz.reshape(1, -1)
#         stack_list.append(dz)
#         tag_list.append(-1)
#
#     X = np.vstack(stack_list)
#
#     # 检查特征数量
#     if X.shape[0] < 2:
#         print("Warning: Not enough samples for t-SNE visualization")
#         return
#
#     # 调整perplexity以适应样本数量
#     actual_perplexity = min(perplexity, (X.shape[0] - 1) // 3)
#     if actual_perplexity < 5:
#         actual_perplexity = 5
#
#     # 生成2D的t-SNE嵌入
#     # tsne_2d = TSNE(n_components=2, perplexity=actual_perplexity,n_iter=n_iter, random_state=random_state, init='pca')
#     tsne_2d = TSNE(n_components=2, perplexity=actual_perplexity, random_state=random_state, init='pca')
#     X_emb_2d = tsne_2d.fit_transform(X)
#
#     # 创建图形
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # 获取正常类别（标签>=0）
#     unique_labels = sorted(set(l for l in tag_list if l >= 0))
#     palette = sns.color_palette("tab10", len(unique_labels))
#
#     # 画主任务特征（正常类别）
#     for i, lab in enumerate(unique_labels):
#         idx = [j for j, t in enumerate(tag_list) if t == lab]
#         if len(idx) == 0:
#             continue
#         if lab == 7:  # 第8类（标签=7）固定为黑色
#             ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1],
#                       label=f"{lab}", s=20, alpha=0.7, color='black')
#         else:
#             ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1],
#                       label=f"{lab}", s=20, alpha=0.7, color=palette[i % len(palette)])
#
#     # 画触发后的特征（蓝色x号，减小大小）
#     triggered_idx = [j for j, t in enumerate(tag_list) if t == -2]
#     if triggered_idx:
#         ax.scatter(X_emb_2d[triggered_idx, 0], X_emb_2d[triggered_idx, 1],
#                   label="Backdoor Features", s=40, c='blue', marker='x', alpha=0.8, linewidths=2)
#
#     # 画delta_z测试时的特征（绿色小三角形）
#     delta_z_test_idx = [j for j, t in enumerate(tag_list) if t == -3]
#     if delta_z_test_idx:
#         ax.scatter(X_emb_2d[delta_z_test_idx, 0], X_emb_2d[delta_z_test_idx, 1],
#                   label="Delta_z Test Features", s=30, c='green', marker='^', alpha=0.8)
#
#     # 画 mean feature diff 向量（橙色星号）
#     md_idx = [j for j, t in enumerate(tag_list) if t == -6]
#     if md_idx:
#         print(f"Plotting Mean Feature Diff at coordinates: {X_emb_2d[md_idx]}")
#         ax.scatter(X_emb_2d[md_idx, 0], X_emb_2d[md_idx, 1],
#                    label="Mean Feature Diff (BD - Clean)", s=160,
#                    c='orange', marker='*', edgecolors='black', linewidth=1,zorder=10)
#
#     # 画触发器图像特征（紫色菱形）
#     trigger_img_idx = [j for j, t in enumerate(tag_list) if t == -4]
#     if trigger_img_idx:
#         trigger_names = features_dict.get('trigger_names', [])
#         trigger_similarities = features_dict.get('trigger_similarities', [])
#
#         # 为每个触发器图像特征添加标签和相似度信息
#         for k, idx in enumerate(trigger_img_idx):
#             name = trigger_names[k] if k < len(trigger_names) else f"Trigger {k+1}"
#             similarity = trigger_similarities[k] if k < len(trigger_similarities) else 0.0
#             label_text = f"{name} (sim: {similarity:.3f})" if similarity != 0.0 else name
#
#             ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1],
#                       label=label_text, s=120, c='purple', marker='D',
#                       alpha=0.9, edgecolors='black', linewidth=1)
#
#     # 画 delta_z 向量本身（红色星号）
#     dz_idx = [j for j, t in enumerate(tag_list) if t == -1]
#     if dz_idx:
#         ax.scatter(X_emb_2d[dz_idx, 0], X_emb_2d[dz_idx, 1],
#                   label="Delta_z Vector", s=150, c='red', marker='*',
#                   edgecolors='black', linewidth=1)
#
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.set_title("t-SNE Feature Visualization")
#     ax.set_xlabel("t-SNE Dimension 1")
#     ax.set_ylabel("t-SNE Dimension 2")
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"t-SNE visualization saved to {save_path}")
#
#     if show_plot:
#         plt.show()
#
#     plt.close()

import matplotlib.patches as patches

def visualize_features_tsne(
        features_dict,
        delta_z=None,
        mean_diff=None,
        perplexity=30,
        n_iter=1500,
        random_state=42,
        save_path=None,
        show_plot=True,
        show_class_legend=True,
        legend_fontsize=15,
):
    # --------- 读取数据 ---------
    features = features_dict.get('features', None)
    labels = features_dict.get('labels', None)
    if features is None or labels is None:
        raise ValueError("features_dict must contain 'features' and 'labels'.")

    features = np.asarray(features)
    labels = np.asarray(labels)

    stack_list = [features]
    tag_list = labels.tolist()

    # 可选：触发后的特征
    if 'features_triggered' in features_dict and getattr(features_dict['features_triggered'], "size", 0) > 0:
        triggered_features = np.asarray(features_dict['features_triggered'])
        stack_list.append(triggered_features)
        tag_list.extend([-2] * len(triggered_features))

    # 可选：delta_z 测试特征
    if 'features_delta_z' in features_dict and getattr(features_dict['features_delta_z'], "size", 0) > 0:
        delta_z_features = np.asarray(features_dict['features_delta_z'])
        stack_list.append(delta_z_features)
        tag_list.extend([-3] * len(delta_z_features))

    # mean feature diff 向量 (标记 -6)
    if mean_diff is not None:
        if hasattr(mean_diff, "detach"):
            md = mean_diff.detach().cpu().numpy()
        else:
            md = np.asarray(mean_diff)
        stack_list.append(md.reshape(1, -1))
        tag_list.append(-6)

    # delta_z 向量 (标记 -1)
    if delta_z is not None:
        if hasattr(delta_z, "detach"):
            dz = delta_z.detach().cpu().numpy()
        else:
            dz = np.asarray(delta_z)
        stack_list.append(dz.reshape(1, -1))
        tag_list.append(-1)

    X = np.vstack(stack_list)

    # --------- t-SNE 计算 ---------
    actual_perplexity = min(perplexity, (X.shape[0] - 1) // 3)
    actual_perplexity = max(actual_perplexity, 5)

    try:
        tsne_2d = TSNE(n_components=2, perplexity=actual_perplexity, n_iter=n_iter, random_state=random_state,
                       init='pca')
    except TypeError:
        tsne_2d = TSNE(n_components=2, perplexity=actual_perplexity, max_iter=n_iter, random_state=random_state,
                       init='pca')

    X_emb_2d = tsne_2d.fit_transform(X)

    import os
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    def _set_times_new_roman_linux_ok(font_path=None):
        """
        优先 Times New Roman；Linux 若未安装则 fallback 到接近 Times 的衬线字体。
        font_path: 可选，指向你自己的 Times New Roman .ttf 路径（最稳）。
        """
        # 1) 如果你手动提供了 ttf，就强制加载（最稳，Linux 一定生效）
        if font_path is not None and os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = [name]
        else:
            # 2) 否则：系统已安装则用 Times New Roman；否则用 Linux 常见替代
            candidates = [
                "Times New Roman",  # Windows / 手动安装后
                "TeX Gyre Termes",  # Linux 常见（很像 Times）
                "Nimbus Roman No9 L",  # Linux 常见（很像 Times）
                "Nimbus Roman",
                "Times",  # 兜底
                "DejaVu Serif",  # 再兜底
            ]
            available = {f.name for f in fm.fontManager.ttflist}
            chosen = next((c for c in candidates if c in available), "DejaVu Serif")

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = [chosen]

        # 3) 数学公式字体也尽量贴近 Times
        plt.rcParams["mathtext.fontset"] = "stix"  # STIX 更像 Times 风格
        plt.rcParams["axes.unicode_minus"] = False  # 负号显示正常

    # 在你画图前调用一次即可（例如放在函数里、创建 fig 之前）
    _set_times_new_roman_linux_ok(
        font_path=None  # 如果你有 Times New Roman.ttf，填路径最稳
    )


    # --------- 开始绘图 ---------
    fig, ax = plt.subplots(figsize=(9, 7))

    # 1. 绘制普通样本点
    unique_labels = sorted(set(l for l in tag_list if l >= 0))
    cmap = plt.get_cmap("tab10")
    for lab in unique_labels:
        idx = [j for j, t in enumerate(tag_list) if t == lab]
        if not idx: continue
        ax.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1], s=20, alpha=0.55, color=cmap(lab % 10),
                   label=f"Client{lab}" if show_class_legend else None, zorder=1)

    # 2. 绘制额外点（Triggered / Delta Z Test）
    triggered_idx = [j for j, t in enumerate(tag_list) if t == -2]
    if triggered_idx:
        ax.scatter(X_emb_2d[triggered_idx, 0], X_emb_2d[triggered_idx, 1], s=26, c='blue', marker='x', alpha=0.6,
                   zorder=2)

    delta_z_test_idx = [j for j, t in enumerate(tag_list) if t == -3]
    if delta_z_test_idx:
        ax.scatter(X_emb_2d[delta_z_test_idx, 0], X_emb_2d[delta_z_test_idx, 1], s=18, c='green', marker='^',
                   alpha=0.45, zorder=2)

    # 3. 获取特殊点坐标
    md_idx = [j for j, t in enumerate(tag_list) if t == -6]  # 菱形
    dz_idx = [j for j, t in enumerate(tag_list) if t == -1]  # 星星

    md_xy = X_emb_2d[md_idx[0]] if md_idx else None
    dz_xy = X_emb_2d[dz_idx[0]] if dz_idx else None

    # --------- 绘制红框 (核心修改部分) ---------
    box_half_w, box_half_h = 0, 0  # 初始化用于后面计算箭头偏移

    if md_xy is not None and dz_xy is not None:
        # 3.1 计算全局数据的跨度 (Span)，用于自适应框的大小
        x_span = X_emb_2d[:, 0].max() - X_emb_2d[:, 0].min()
        y_span = X_emb_2d[:, 1].max() - X_emb_2d[:, 1].min()

        # 3.2 定义框的比例 (占总图宽高的 6% 到 8% 左右，视觉效果最像截图)
        scale_ratio = 0.08
        box_w = x_span * scale_ratio
        box_h = y_span * scale_ratio

        # 记录半宽半高，用于后续文字定位
        box_half_w = box_w / 2
        box_half_h = box_h / 2

        # 3.3 计算两个特殊点的中心位置
        center_x = (md_xy[0] + dz_xy[0]) / 2
        center_y = (md_xy[1] + dz_xy[1]) / 2

        # 3.4 绘制红框 (以中心点向四周扩散)
        # xy参数是矩形左下角坐标
        rect = patches.Rectangle(
            (center_x - box_half_w, center_y - box_half_h),
            box_w,
            box_h,
            linewidth=3.5,  # 红色线宽
            edgecolor='red',  # 红框
            facecolor='none',  # 透明内部
            zorder=100  # 放在最上层
        )
        ax.add_patch(rect)

    # 4. 绘制特殊点 (放在框画完之后，确保点在视觉上清晰)
    if md_xy is not None:
        ax.scatter(md_xy[0], md_xy[1], label=r"$\mathbf{v}_{\mathcal{T}}$",
                   s=200, marker='D', c='purple', edgecolors='black', linewidth=1.6, zorder=110)

    if dz_xy is not None:
        ax.scatter(dz_xy[0], dz_xy[1], label=r"FST",
                   s=250, marker='*', c='red', edgecolors='black', linewidth=1.4, zorder=111)

    # 5. 添加文字标注 (根据框的大小自适应偏移)
    # 截图风格：文字在框外，箭头指向点
    offset_scale_x = box_half_w * 1.5  # 文字水平偏移量
    offset_scale_y = box_half_h * 1.8  # 文字垂直偏移量

    if md_xy is not None:
        ax.annotate(
            r"$\mathbf{v}_{\mathcal{T}}$",
            xy=(md_xy[0], md_xy[1]),  # 箭头指向点
            xytext=(md_xy[0] + offset_scale_x, md_xy[1] + offset_scale_y),  # 文字位置：右上
            arrowprops=dict(arrowstyle='-', color='black', lw=1.2),  # 实线箭头
            fontsize=15, fontweight='bold', zorder=120
        )

    if dz_xy is not None:
        ax.annotate(
            r"FST",
            xy=(dz_xy[0], dz_xy[1]),  # 箭头指向点
            xytext=(dz_xy[0] + offset_scale_x, dz_xy[1] - offset_scale_y),  # 文字位置：右下
            arrowprops=dict(arrowstyle='-', color='black', lw=1.2),
            fontsize=15, fontweight='bold', zorder=120
        )

    # --------- 图注与修饰 ---------
    handles, labs = ax.get_legend_handles_labels()
    final_handles, final_labels = [], []

    # 筛选图注
    for h, lab in zip(handles, labs):
        if lab == r"$\mathbf{v}_{\mathcal{T}}$" or lab == r"FST":
            final_handles.append(h)
            final_labels.append(lab)

    if show_class_legend:
        class_items = []
        for h, lab in zip(handles, labs):
            if lab.startswith("Client") and (h, lab) not in zip(final_handles, final_labels):
                class_items.append((h, lab))
        class_items.sort(key=lambda x: int(x[1].replace("Client", "")))
        for h, lab in class_items:
            final_handles.append(h)
            final_labels.append(lab)

    if final_handles:
        ax.legend(final_handles, final_labels, loc='lower right', frameon=True,
                  fontsize=legend_fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)  # 保留底边框线作为边界
    ax.spines['left'].set_visible(True)  # 保留左边框线

    plt.tight_layout()

    if save_path:
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        pdf_dir = os.path.dirname(save_path)
        pdf_path = os.path.join(pdf_dir if pdf_dir else '.', f"{base_name}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved to {save_path} and {pdf_path}")

    if show_plot:
        plt.show()

    plt.close()

