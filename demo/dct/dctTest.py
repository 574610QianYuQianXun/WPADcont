import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

def dct_single_image(img):
    """
    对单张图像进行DCT变换。
    img: numpy数组，shape可以是 (H,W) 或 (H,W,C)
    """
    if img.ndim == 2:
        img = np.float32(img) / 255.0
        dct_img = cv2.dct(img)
        return dct_img
    elif img.ndim == 3:
        dct_channels = []
        for c in range(img.shape[2]):
            channel = np.float32(img[:, :, c]) / 255.0
            dct_c = cv2.dct(channel)
            dct_channels.append(dct_c)
        return np.stack(dct_channels, axis=2)
    else:
        raise ValueError(f"不支持的图像维度: {img.ndim}")

def idct_single_image(dct_img):
    """
    对单张DCT图像进行逆DCT变换。
    dct_img: numpy数组，shape可以是 (H,W) 或 (H,W,C)
    """
    if dct_img.ndim == 2:
        img = cv2.idct(dct_img)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
    elif dct_img.ndim == 3:
        idct_channels = []
        for c in range(dct_img.shape[2]):
            idct_c = cv2.idct(dct_img[:, :, c])
            idct_c = np.clip(idct_c * 255.0, 0, 255).astype(np.uint8)
            idct_channels.append(idct_c)
        return np.stack(idct_channels, axis=2)
    else:
        raise ValueError(f"不支持的图像维度: {dct_img.ndim}")

def normalize_for_display(dct_img):
    """
    将DCT结果归一化到0-1区间，方便可视化。
    """
    dct_img = np.abs(dct_img)  # 取绝对值
    dct_img -= dct_img.min()
    dct_img /= (dct_img.max() + 1e-8)  # 防止除0
    return dct_img

def process_dataset(dataset_name, num_images=1):
    """
    dataset_name: 'mnist' 或 'cifar10'
    num_images: 指定要处理多少张图片
    """
    if dataset_name.lower() == 'mnist':
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST(root='../../dataset', train=True, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.CIFAR10(root='../../dataset/CIFAR10', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, (image, label) in enumerate(dataloader):
        image = image.squeeze(0).numpy()  # [C,H,W]

        if dataset_name.lower() == 'mnist':
            img = np.squeeze(image)  # [H,W]
            img = np.uint8(img * 255)
        else:  # CIFAR-10
            img = np.transpose(image, (1,2,0))  # [H,W,C]
            img = np.uint8(img * 255)

        dct_img = dct_single_image(img)
        idct_img = idct_single_image(dct_img)
        dct_img_disp = normalize_for_display(dct_img)

        # 可视化：原图、DCT图、逆DCT还原图
        plt.figure(figsize=(12,4))
        plt.suptitle(f'{dataset_name.upper()} - Image {idx+1}', fontsize=16)

        if img.ndim == 2:  # MNIST
            plt.subplot(1,3,1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(dct_img_disp, cmap='gray')
            plt.title('DCT')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(idct_img, cmap='gray')
            plt.title('Inverse DCT')
            plt.axis('off')
        else:  # CIFAR10
            plt.subplot(1,3,1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            plt.title('Original')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(dct_img_disp)
            plt.title('DCT')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(cv2.cvtColor(idct_img, cv2.COLOR_RGB2BGR))
            plt.title('Inverse DCT')
            plt.axis('off')

        plt.show()

        if idx + 1 >= num_images:
            break

if __name__ == "__main__":
    # 示例：处理 MNIST，显示1张
    # process_dataset('mnist', num_images=1)

    # 示例：处理 CIFAR-10，显示2张
    process_dataset('cifar10', num_images=2)
