import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import cv2
from scipy.fftpack import dct, idct


# ------- 基础DCT工具函数 -------
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')


def idct2(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')


def dct_transform(image):
    if len(image.shape) == 2:
        return dct2(image)
    else:
        return np.stack([dct2(image[:, :, i]) for i in range(image.shape[2])], axis=2)


def idct_transform(dct_image):
    if len(dct_image.shape) == 2:
        return idct2(dct_image)
    else:
        return np.stack([idct2(dct_image[:, :, i]) for i in range(dct_image.shape[2])], axis=2)


def generate_trigger(size=(8, 8)):
    wm = np.zeros(size, dtype=np.uint8)
    wm[2:6, 2:6] = 255
    return wm


def embed_trigger(dct_img, trigger, strength=0.05):
    dct_img = dct_img.copy()
    if len(dct_img.shape) == 2:
        H, W = dct_img.shape
        start_h, start_w = H // 4, W // 4
        h, w = trigger.shape
        if h > (H - start_h) or w > (W - start_w):
            scale = min((H - start_h) / h, (W - start_w) / w)
            trigger = cv2.resize(trigger, (int(w * scale), int(h * scale)))
            h, w = trigger.shape
        dct_img[start_h:start_h + h, start_w:start_w + w] += (trigger > 128) * strength
    else:
        H, W, C = dct_img.shape
        start_h, start_w = H // 4, W // 4
        h, w = trigger.shape
        if h > (H - start_h) or w > (W - start_w):
            scale = min((H - start_h) / h, (W - start_w) / w)
            trigger = cv2.resize(trigger, (int(w * scale), int(h * scale)))
            h, w = trigger.shape
        for c in range(C):
            dct_img[start_h:start_h + h, start_w:start_w + w, c] += (trigger > 128) * strength
    return dct_img, (start_h, start_w, h, w)


# ------- 核心后门函数 -------
def How_backdoor_DCT(train_set, origin_target, aim_target, dataset_name='CIFAR10', strength=0.05):
    trigger = generate_trigger()

    # 遍历训练集中的所有数据
    for idx in train_set.idxs:
        # 判断是否为目标标签
        if train_set.dataset.targets[idx] == origin_target or (
                isinstance(train_set.dataset.targets, torch.Tensor) and train_set.dataset.targets[
            idx].item() == origin_target
        ):
            img = train_set.dataset.data[idx]

            # 如果是MNIST，处理为灰度图
            if dataset_name == 'MNIST':
                # 转换为numpy并移除通道维度
                img_np = img.numpy().squeeze() * 255
                img_np = img_np.astype(np.float32)
            else:  # CIFAR10处理
                img_np = img.astype(np.float32) / 255.0

            # 对图像进行DCT变换
            dct_img = dct_transform(img_np)
            # 嵌入触发器并返回触发器的位置
            dct_img_triggered, trigger_pos = embed_trigger(dct_img, trigger, strength=strength)
            # 逆DCT变换恢复图像
            poisoned_img = idct_transform(dct_img_triggered)
            poisoned_img = np.clip(poisoned_img, 0, 1)

            # 转换为uint8类型
            poisoned_img = (poisoned_img * 255).astype(np.uint8)

            # 如果是MNIST，train_set.dataset.data通常是一个ByteTensor，需要将poisoned_img转换为ByteTensor
            if dataset_name == 'MNIST':
                poisoned_img = torch.from_numpy(poisoned_img).byte()

            # 替换数据和标签
            train_set.dataset.data[idx] = poisoned_img
            train_set.dataset.targets[idx] = aim_target

    return trigger_pos


# ------- 简单的数据集包装 -------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.labels = torch.tensor([self.dataset.targets[idx] for idx in self.idxs])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# ------- 通用测试流程 -------
def test_demo(dataset_name='CIFAR10'):
    if dataset_name == 'CIFAR10':
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.CIFAR10(root='../../dataset/CIFAR10', train=True, download=True, transform=transform)
        origin_target, aim_target = 8, 7
    elif dataset_name == 'MNIST':
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST(root='../../dataset', train=True, download=True, transform=transform)
        origin_target, aim_target = 1, 8
    else:
        raise ValueError('Unsupported dataset!')

    # 包装成DatasetSplit
    all_idxs = list(range(len(dataset)))
    train_set = DatasetSplit(dataset, all_idxs)

    # 选取原target的前5张
    idx_to_show = [i for i in train_set.idxs if train_set.dataset.targets[i] == origin_target][:5]

    fig, axes = plt.subplots(3, len(idx_to_show), figsize=(15, 10))
    for i, idx in enumerate(idx_to_show):
        img = train_set.dataset.data[idx]
        if dataset_name == 'MNIST':
            axes[0, i].imshow(img, cmap='gray')
        else:
            axes[0, i].imshow(img)
        axes[0, i].set_title(f"Before (Label: {train_set.dataset.targets[idx]})")
        axes[0, i].axis('off')


    # 显示DCT后的图像
    for i, idx in enumerate(idx_to_show):
        img = train_set.dataset.data[idx]
        if dataset_name == 'MNIST':
            img_np = img.numpy().squeeze() * 255
            img_np = img_np.astype(np.float32)
        else:
            img_np = img.astype(np.float32) / 255.0

        dct_img = dct_transform(img_np)
        axes[1, i].imshow(np.log(np.abs(dct_img) + 1e-6), cmap='gray')
        axes[1, i].set_title(f"DCT of Image {i + 1}")
        axes[1, i].axis('off')

    # 植入后门并获取触发器的位置
    trigger_pos = How_backdoor_DCT(train_set, origin_target=origin_target, aim_target=aim_target,
                                       dataset_name=dataset_name, strength=0.05)

    # 显示植入触发器后的图像
    for i, idx in enumerate(idx_to_show):
        img = train_set.dataset.data[idx]
        if dataset_name == 'MNIST':
            axes[2, i].imshow(img, cmap='gray')
        else:
            axes[2, i].imshow(img)
        axes[2, i].set_title(f"After (Label: {train_set.dataset.targets[idx]})")
        axes[2, i].axis('off')

    # 可视化触发器和其植入位置
    for i in range(len(idx_to_show)):
        start_h, start_w, h, w = trigger_pos
        axes[2, i].add_patch(plt.Rectangle((start_w, start_h), w, h, edgecolor='red', facecolor='none', linewidth=2))
        axes[2, i].text(start_w, start_h, 'Trigger', color='red', fontsize=12)

    plt.suptitle(f"{dataset_name} - Backdoor Injection Demo", fontsize=16)
    plt.tight_layout()
    plt.show()


# ------- 主程序入口 -------
if __name__ == "__main__":
    print("Testing CIFAR10...")
    test_demo('CIFAR10')

    print("Testing MNIST...")
    # test_demo('MNIST')
