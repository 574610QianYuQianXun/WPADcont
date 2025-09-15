import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# ============================
# DCT处理函数
# ============================
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def dct_transform(image):
    """对图像做逐通道DCT"""
    if len(image.shape) == 2:  # 灰度图
        return dct2(image)
    else:  # 彩色图
        channels = []
        for i in range(image.shape[2]):
            channels.append(dct2(image[:,:,i]))
        return np.stack(channels, axis=2)

def idct_transform(dct_image):
    """对图像做逐通道IDCT"""
    if len(dct_image.shape) == 2:
        return idct2(dct_image)
    else:
        channels = []
        for i in range(dct_image.shape[2]):
            channels.append(idct2(dct_image[:,:,i]))
        return np.stack(channels, axis=2)

# ============================
# 生成触发器
# ============================
def generate_trigger(size=(8, 8)):
    wm = np.zeros(size, dtype=np.uint8)
    # 简单生成个小方块
    wm[2:6, 2:6] = 255
    return wm

# ============================
# 嵌入触发器到中高频
# ============================
import cv2


def embed_trigger(dct_img, trigger, strength=0.05):
    dct_img = dct_img.copy()

    if len(dct_img.shape) == 2:
        # 灰度图
        H, W = dct_img.shape
        start_h = H // 4
        start_w = W // 4

        max_h = H - start_h
        max_w = W - start_w

        h, w = trigger.shape
        # 如果 trigger 太大，自动缩小
        if h > max_h or w > max_w:
            scale_h = max_h / h
            scale_w = max_w / w
            scale = min(scale_h, scale_w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            trigger = cv2.resize(trigger, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = trigger.shape

        end_h = start_h + h
        end_w = start_w + w

        sub_dct = dct_img[start_h:end_h, start_w:end_w]
        trigger_binary = (trigger > 128).astype(np.float32)
        dct_img[start_h:end_h, start_w:end_w] = sub_dct + trigger_binary * strength

    else:
        # 彩色图
        H, W, _ = dct_img.shape
        start_h = H // 4
        start_w = W // 4

        max_h = H - start_h
        max_w = W - start_w

        h, w = trigger.shape
        # 如果 trigger 太大，自动缩小
        if h > max_h or w > max_w:
            scale_h = max_h / h
            scale_w = max_w / w
            scale = min(scale_h, scale_w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            trigger = cv2.resize(trigger, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = trigger.shape

        end_h = start_h + h
        end_w = start_w + w

        for c in range(3):
            sub_dct = dct_img[start_h:end_h, start_w:end_w, c]
            trigger_binary = (trigger > 128).astype(np.float32)
            dct_img[start_h:end_h, start_w:end_w, c] = sub_dct + trigger_binary * strength

    return dct_img


# ============================
# 植入后门
# ============================
def poison_image(img_np, trigger):
    img_np = img_np.astype(np.float32) / 255.0

    dct_img = dct_transform(img_np)
    dct_img_triggered = embed_trigger(dct_img, trigger, strength=0.05)
    poisoned_img = idct_transform(dct_img_triggered)
    poisoned_img = np.clip(poisoned_img, 0, 1)

    return (poisoned_img * 255).astype(np.uint8)

# ============================
# 处理 MNIST
# ============================
def main_mnist(num_images=5):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='../../dataset', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    trigger = generate_trigger()

    clean_imgs = []
    poisoned_imgs = []

    for idx, (img, label) in enumerate(dataloader):
        img_np = img.squeeze(0).numpy() * 255  # [H,W]

        poisoned_img = poison_image(img_np, trigger)

        clean_imgs.append(img_np.astype(np.uint8))
        poisoned_imgs.append(poisoned_img)

        if idx+1 >= num_images:
            break

    # visualize_results(clean_imgs, poisoned_imgs, dataset_name='MNIST')
    visualize_results(clean_imgs, poisoned_imgs, trigger, dataset_name='MNIST')


# ============================
# 处理 CIFAR10
# ============================
def main_cifar10(num_images=5):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root='../../dataset/CIFAR10', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    trigger = generate_trigger()

    clean_imgs = []
    poisoned_imgs = []

    for idx, (img, label) in enumerate(dataloader):
        img_np = img.squeeze(0).permute(1,2,0).numpy() * 255  # [H,W,C]

        poisoned_img = poison_image(img_np, trigger)

        clean_imgs.append(img_np.astype(np.uint8))
        poisoned_imgs.append(poisoned_img)

        if idx+1 >= num_images:
            break

    visualize_results(clean_imgs, poisoned_imgs, trigger, dataset_name='CIFAR10')


# ============================
# 可视化
# ============================
def visualize_results(clean_imgs, poisoned_imgs, trigger, dataset_name='MNIST'):
    num_imgs = len(clean_imgs)
    plt.figure(figsize=(12, 6))

    for i in range(num_imgs):
        # Clean image
        plt.subplot(3, num_imgs, i + 1)
        img = clean_imgs[i]
        if dataset_name == 'MNIST':
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
        plt.title('Clean')
        plt.axis('off')

        # Poisoned image
        plt.subplot(3, num_imgs, i + 1 + num_imgs)
        img = poisoned_imgs[i]
        if dataset_name == 'MNIST':
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
        plt.title('Poisoned')
        plt.axis('off')

        # Trigger (显示 trigger)
        plt.subplot(3, num_imgs, i + 1 + num_imgs * 2)
        if dataset_name == 'MNIST':
            plt.imshow(trigger, cmap='gray')
        else:
            plt.imshow(trigger, cmap='gray')  # trigger一般是灰度的
        plt.title('Trigger')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



# ============================
# 执行
# ============================
if __name__ == '__main__':
    print("处理 MNIST 数据集...")
    # main_mnist(num_images=5)
    print("\n处理 CIFAR10 数据集...")
    main_cifar10(num_images=5)
