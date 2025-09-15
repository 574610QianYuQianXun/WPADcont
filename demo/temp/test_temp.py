from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import transforms


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

    # Step 1: 计算注意力图
    att_map = get_attention_map_p(feature,p=4)  # 得到 [1, H, W]
    # att_map = att_map.squeeze(0).cpu().detach().numpy()  # -> [H, W]
    att_map = F.interpolate(
        att_map.unsqueeze(1),  # 添加通道维 [1, 1, H, W]
        size=input_image.shape[-2:],  # 目标尺寸（28x28）
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().detach().numpy()  # -> [28, 28]
    # Step 2: 原图处理
    input_img = input_image.squeeze(0).squeeze(0).cpu().detach().numpy()  # -> [H, W]

    # Step 3: Normalize
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)  # 归一化到0-1
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

    # Step 4: 可视化
    plt.figure(figsize=(10, 5))

    # 原图
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')

    # 注意力图
    plt.subplot(1, 3, 2)
    plt.title('Attention Map')
    plt.imshow(att_map, cmap='jet')  # 注意力图用 jet colormap
    plt.axis('off')

    # 叠加图
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(input_img, cmap='gray')
    plt.imshow(att_map, cmap='jet', alpha=0.5)  # alpha 控制透明度
    plt.axis('off')

    plt.tight_layout()

    # 保存 or 显示
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def Evaluate(model, datasets, loss_func, params):
    model.eval()
    total_loss = 0
    model.to(params.device)
    correct = 0
    total = 0
    test_loader = DataLoader(datasets, batch_size=params.bs, shuffle=False)
    for data, target in test_loader:
        data, target = data.to(params.device), target.to(params.device)
        _,output = model(data)
        loss = loss_func(output, target)
        total_loss += loss.item() * output.shape[0]
        _, predict = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predict == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(datasets)
    return accuracy, avg_loss

from collections import defaultdict
import torch
from torch.utils.data import DataLoader

def Backdoor_Evaluate(model, dataset, loss_func, params, enable_classwise_loss=False):
    """
    对模型在后门攻击场景和整体数据集上进行评估。
    如果 enable_classwise_loss=True，则统计每类损失。
    """
    input_shape = dataset[0][0].shape

    pattern_tensor = torch.tensor([
        [1., -10., 1.],
        [-10., 1., -10.],
        [-10., -10., -10.],
        [-10., 1., -10.],
        [1., -10., 1.]])
    x_top = 3
    y_top = 23
    mask_value = -10
    model.eval()
    model.to(params.device)
    mask, pattern = Make_pattern(x_top, y_top, mask_value, pattern_tensor, input_shape, params)

    loader = DataLoader(dataset, batch_size=params.bs, shuffle=False)
    total_correct_all, total_loss_all, total_samples_all = 0, 0.0, 0
    total_correct_backdoor, total_loss_backdoor, total_samples_backdoor = 0, 0.0, 0

    # 每类损失统计
    if enable_classwise_loss:
        class_loss_sum = defaultdict(float)
        class_sample_count = defaultdict(int)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(params.device), target.to(params.device)

            _, output = model(data)
            loss = loss_func(output, target)

            # 总体准确率和损失
            _, predicted = torch.max(output, 1)
            total_samples_all += target.size(0)
            total_correct_all += (predicted == target).sum().item()
            total_loss_all += loss.item() * target.size(0)

            # 仅在启用时进行每类损失统计
            if enable_classwise_loss:
                for i in range(len(target)):
                    label = target[i].item()
                    sample_loss = loss_func(output[i].unsqueeze(0), target[i].unsqueeze(0)).item()
                    class_loss_sum[label] += sample_loss
                    class_sample_count[label] += 1

            # 后门样本评估
            if params.poison_type==2:
                index = Implant_trigger(data, target, mask, pattern, params)
            else:
                index = (target == params.origin_target)
            if index.sum() > 0:
                data_masked = data[index]
                target_masked = target[index]

                _, output_masked = model(data_masked)

                # 将目标标签全部改为攻击目标标签（即标签反转）
                target_aim = torch.full_like(target_masked, fill_value=params.aim_target)
                loss_masked = loss_func(output_masked, target_aim)

                _, predicted_masked = torch.max(output_masked, 1)

                total_samples_backdoor += target_masked.size(0)
                total_loss_backdoor += loss_masked.item() * target_masked.size(0)
                total_correct_backdoor += (predicted_masked == params.aim_target).sum().item()

    # 安全除法函数
    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    acc_all = 100 * safe_div(total_correct_all, total_samples_all)
    acc_backdoor = 100 * safe_div(total_correct_backdoor, total_samples_backdoor)
    loss_all = safe_div(total_loss_all, total_samples_all)
    loss_backdoor = safe_div(total_loss_backdoor, total_samples_backdoor)

    # 打印每类平均损失
    if enable_classwise_loss:
        print("\n[每类平均损失统计]")
        for cls in sorted(class_loss_sum.keys()):
            avg_loss = safe_div(class_loss_sum[cls], class_sample_count[cls])
            print(f"类别 {cls:2d} | 样本数: {class_sample_count[cls]:4d} | 平均损失: {avg_loss:.4f}")

    return acc_backdoor, loss_backdoor, acc_all, loss_all

def Implant_trigger(data, label, mask, pattern, params):
    index = []
    for i in range(len(data)):
        if label[i] == params.aim_target:
            continue
        else:
            data[i] = (1 - mask) * data[i] + mask * pattern
            # label[i] = params.aim_target
            index.append(i)

    index = torch.tensor(index).to(params.device)
    return index

def Make_pattern(x_top, y_top, mask_value, pattern_tensor, input_shape, params):

    normalize = None
    if params.task == "MNIST":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    if params.task == "CIFAR10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                              std=[0.2023, 0.1994, 0.2010])
    full_image = torch.zeros(input_shape)
    full_image.fill_(mask_value)
    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    if x_bot >= input_shape[1] or y_bot >= input_shape[2]:
        raise ValueError(...)

    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
    mask = 1 * (full_image != mask_value).to(params.device)
    pattern = normalize(full_image).to(params.device)

    return mask, pattern



