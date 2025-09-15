import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

def dct_single_image(img):
    if img.ndim == 2:
        img = np.float32(img) / 255.0
        return cv2.dct(img)
    elif img.ndim == 3:
        return np.stack([cv2.dct(np.float32(img[:,:,c]) / 255.0) for c in range(img.shape[2])], axis=2)
    else:
        raise ValueError("Unsupported image format")

def idct_single_image(dct_img):
    if dct_img.ndim == 2:
        img = cv2.idct(dct_img)
        return np.uint8(np.clip(img * 255.0, 0, 255))
    elif dct_img.ndim == 3:
        return np.stack([
            np.uint8(np.clip(cv2.idct(dct_img[:,:,c]) * 255.0, 0, 255))
            for c in range(dct_img.shape[2])
        ], axis=2)
    else:
        raise ValueError("Unsupported DCT format")


def embed_watermark_in_dct(dct_img, watermark):
    h, w = watermark.shape
    wm_flat = watermark.flatten()
    wm_binary = (wm_flat > 128).astype(np.float32)

    dct_copy = dct_img.copy()

    if dct_copy.ndim == 2:
        # 灰度图
        embed_area = dct_copy[10:10 + h, 10:10 + w]
        embed_area += wm_binary.reshape(h, w) * 0.01
        dct_copy[10:10 + h, 10:10 + w] = embed_area
    elif dct_copy.ndim == 3:
        # 彩色图
        for c in range(3):  # 对每个通道分别嵌入
            embed_area = dct_copy[10:10 + h, 10:10 + w, c]
            embed_area += wm_binary.reshape(h, w) * 0.01
            dct_copy[10:10 + h, 10:10 + w, c] = embed_area
    else:
        raise ValueError("Unsupported image format in DCT")

    return dct_copy


def extract_watermark_from_dct(dct_img, shape=(16, 16)):
    h, w = shape
    if dct_img.ndim == 2:
        extracted = dct_img[10:10 + h, 10:10 + w]
    elif dct_img.ndim == 3:
        extracted = dct_img[10:10 + h, 10:10 + w, 0]  # 只取第一个通道提取
    else:
        raise ValueError("Unsupported DCT format in extraction")

    watermark = (extracted > np.median(extracted)).astype(np.uint8) * 255
    return watermark


def normalize_img(img):
    img = np.abs(img)
    img = (img - img.min()) / (img.max() + 1e-8)
    return img

def get_sample_image(dataset_name='mnist'):
    if dataset_name.lower() == 'mnist':
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST(root='../../dataset', train=True, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.CIFAR10(root='../../dataset/CIFAR10', train=True, download=True, transform=transform)
    else:
        raise ValueError("Only mnist or cifar10 allowed")

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    image, _ = next(iter(loader))
    image = image.squeeze().numpy()

    if dataset_name == 'mnist':
        img = np.uint8(image * 255)
    else:
        img = np.transpose(image, (1,2,0))
        img = np.uint8(img * 255)
    return img

def generate_watermark(size=(16,16)):
    # 可以用文字/图案生成，先做简单图案
    wm = np.zeros(size, dtype=np.uint8)
    cv2.putText(wm, "WM", (1, size[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    return wm

def visualize_all(original, dct_img, watermarked_img, extracted_wm):
    plt.figure(figsize=(14,4))
    plt.subplot(1,4,1)
    plt.imshow(original if original.ndim==2 else cv2.cvtColor(original, cv2.COLOR_RGB2BGR), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(normalize_img(dct_img) if dct_img.ndim==2 else normalize_img(dct_img[:,:,0]), cmap='gray')
    plt.title("DCT (with watermark)")
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(watermarked_img if watermarked_img.ndim==2 else cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2BGR), cmap='gray')
    plt.title("After Inverse DCT")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(extracted_wm, cmap='gray')
    plt.title("Extracted Watermark")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(dataset='mnist'):
    original_img = get_sample_image(dataset)
    watermark = generate_watermark((16,16))

    dct_img = dct_single_image(original_img)
    dct_with_wm = embed_watermark_in_dct(dct_img, watermark)
    wm_img = idct_single_image(dct_with_wm)
    extracted_wm = extract_watermark_from_dct(dct_with_wm, shape=watermark.shape)

    visualize_all(original_img, dct_with_wm, wm_img, extracted_wm)

if __name__ == "__main__":
    # 用 'mnist' 或 'cifar10' 都可以
    main('cifar10')
