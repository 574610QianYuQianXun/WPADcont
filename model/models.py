from torch import nn
import torch
import torch.nn.functional as F


def get_model(args):
    model = None
    if args.model == 'CNNMnist':
        model = CNNMnist(args=args).to(args.device)
    if args.model == 'ResNet18':
        model = ResNet18(args=args).to(args.device)

    return model


# 其他文章中使用的
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        # 第一层卷积和ReLU
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3, stride=1, padding=1)
        # 第一层最大池化
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        # 第二层卷积和ReLU
        self.conv2 = nn.Conv2d(30, 5, kernel_size=3, stride=1, padding=1)
        # 第二层最大池化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(125, 100)
        # 输出层
        self.fc2 = nn.Linear(100, args.num_classes)

    def forward(self, x=None, features=None):
        """
        支持两种调用方式：
        1. 输入原始图像 x：返回 (features, logits)
        2. 输入中间特征 features：返回 (features, logits)，features 不再经过 CNN
        """
        if features is None:
            # === 从图像计算 features ===
            if x is None:
                raise ValueError("必须提供 x 或 features")
            x = self.pool1(F.relu(self.conv1(x)))
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            features = x.view(x.size(0), -1)  # 展平为 [B, 125]
        else:
            # === 直接用外部传入的 features ===
            # 确保是二维 [batch, D]
            if features.dim() != 2:
                features = features.view(features.size(0), -1)

        # === 分类头 ===
        x = F.relu(self.fc1(features))
        logits = self.fc2(x)

        return features, logits

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#
#         # 卷积层部分
#         self.conv_layers = nn.Sequential(
#             # 第一卷积块
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # 第二卷积块
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         # 全连接分类器部分 (包装成一个模块)
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 7 * 7, 128),  # MNIST经过两次2x2池化后是7x7
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, args.num_classes)
#         )
#
#     def forward(self, x):
#         # 特征提取
#         x = self.conv_layers(x)
#
#         # 展平特征
#         features = x.view(x.size(0), -1)  # 保持特征输出
#
#         # 分类
#         x = self.classifier(features)
#
#         return features, x  # 保持原始返回格式
#
#     def get_classifier(self):
#         """获取分类器模块的引用"""
#         return self.classifier
#
#     def get_feature_extractor(self):
#         """获取特征提取部分的引用"""
#         return self.conv_layers

# 轻量级ResNet18
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 64, stride=1)
        self.layer2 = self.make_layer(64, 128, stride=2)
        self.layer3 = self.make_layer(128, 256, stride=2)
        self.layer4 = self.make_layer(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, args.num_classes)

    def make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x=None, features=None):
        """
        Forward 支持两种调用方式：
        1. 输入原始图像 x：返回 (features, logits)
        2. 输入中间特征 features：返回 (features, logits)，features 不再经过 CNN
        """
        if features is None:
            # === 从图像计算 features ===
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            features = x
        else:
            # === 直接用外部传入的 features ===
            # 确保是二维 [batch, D]
            if features.dim() != 2:
                features = features.view(features.size(0), -1)

        # 分类头
        logits = self.fc(features)
        return features, logits


