from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


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


def Backdoor_Evaluate(model, dataset, loss_func, params, enable_classwise_loss=False, mask=None, pattern=None, delta_z=None):
    """
    评估模型的后门攻击性能
    """
    model.eval()
    model.to(params.device)
    loader = DataLoader(dataset, batch_size=params.bs, shuffle=False)

    # 初始化统计变量
    total_correct_all, total_loss_all, total_samples_all = 0, 0.0, 0
    total_correct_backdoor, total_loss_backdoor, total_samples_backdoor = 0, 0.0, 0
    total_entropy_sum, total_entropy_count = 0.0, 0

    if enable_classwise_loss:
        class_loss_sum = defaultdict(float)
        class_sample_count = defaultdict(int)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(params.device), target.to(params.device)

            features, outputs = model(data)

            # 计算softmax概率和熵
            prob = F.softmax(outputs, dim=1)
            entropy_per_sample = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
            total_entropy_sum += entropy_per_sample.sum().item()
            total_entropy_count += entropy_per_sample.size(0)

            # 计算整体性能
            loss = loss_func(outputs, target)
            _, predicted = torch.max(outputs, 1)

            total_samples_all += target.size(0)
            total_correct_all += (predicted == target).sum().item()
            total_loss_all += loss.item() * target.size(0)

            # 按类别统计损失
            if enable_classwise_loss:
                for i in range(len(target)):
                    label = target[i].item()
                    sample_loss = loss_func(outputs[i:i+1], target[i:i+1]).item()
                    class_loss_sum[label] += sample_loss
                    class_sample_count[label] += 1

            # 后门攻击评估
            origin_mask = (target == params.origin_target)
            if origin_mask.sum() > 0:
                data_masked = data[origin_mask]
                target_masked = target[origin_mask]

                # 重新前向传播获取特征
                features_masked, output_masked = model(data_masked)

                # 如果有特征触发器，添加扰动
                if delta_z is not None:
                    # 确保维度匹配
                    if features_masked.shape[1] != delta_z.shape[0]:
                        print(f"Warning: Feature dimension mismatch in backdoor eval. "
                              f"Expected {delta_z.shape[0]}, got {features_masked.shape[1]}")
                    else:
                        new_features = features_masked + delta_z.unsqueeze(0).expand_as(features_masked)
                        _, output_masked = model(features=new_features)

                # 计算后门攻击成功率
                target_aim = torch.full_like(target_masked, fill_value=params.aim_target)
                loss_masked = loss_func(output_masked, target_aim)

                _, predicted_masked = torch.max(output_masked, 1)
                total_samples_backdoor += target_masked.size(0)
                total_loss_backdoor += loss_masked.item() * target_masked.size(0)
                total_correct_backdoor += (predicted_masked == params.aim_target).sum().item()

    # 安全除法函数
    def safe_div(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    # 计算最终指标
    acc_all = 100.0 * safe_div(total_correct_all, total_samples_all)
    acc_backdoor = 100.0 * safe_div(total_correct_backdoor, total_samples_backdoor)
    loss_all = safe_div(total_loss_all, total_samples_all)
    loss_backdoor = safe_div(total_loss_backdoor, total_samples_backdoor)
    avg_entropy = safe_div(total_entropy_sum, total_entropy_count)

    # 打印结果
    print(f"[平均熵] Avg entropy of softmax probabilities: {avg_entropy:.4f}")

    if enable_classwise_loss and class_sample_count:
        print("\n[每类平均损失统计]")
        for cls in sorted(class_loss_sum.keys()):
            avg_loss = safe_div(class_loss_sum[cls], class_sample_count[cls])
            print(f"类别 {cls:2d} | 样本数: {class_sample_count[cls]:4d} | 平均损失: {avg_loss:.4f}")

    return acc_backdoor, loss_backdoor, acc_all, loss_all
