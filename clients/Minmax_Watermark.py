import math
import torch
import torch.nn.functional as F
from scipy.fftpack import dct

def to_device(tensor, device):
    """
    Utility: move tensor to target device if not already.
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor

def torch_dct_1d(x, device=None):
    """
    计算输入x的1D DCT-II变换，返回大小不变。
    x: Tensor shape [N]
    device: torch.device，和x保持一致
    """
    N = x.shape[0]
    if device is None:
        device = x.device

    # 构造DCT基矩阵
    n = torch.arange(N, device=device).reshape(1, -1)  # [1, N]
    k = torch.arange(N, device=device).reshape(-1, 1)  # [N, 1]
    dct_basis = torch.cos(math.pi / N * (n + 0.5) * k)  # [N, N]

    scale = torch.sqrt(torch.tensor(2.0 / N, device=device)) * torch.ones(N, device=device)  # [N]

    # 第一行的scale应为 sqrt(1/N)
    scale[0] = torch.sqrt(torch.tensor(1.0 / N, device=device))

    dct_mat = scale.unsqueeze(1) * dct_basis  # [N, N]

    # 计算DCT
    y = torch.matmul(dct_mat, x)  # [N]

    return y


class MinMaxWatermarker:
    """
    Implements the min-max adversarial game for watermark embedding:
      - Maximizes watermark match rate
      - Minimizes anomaly detection score
    """
    def __init__(self, device, lambda_match=1.0, lambda_ano=1.0, dct_ratio=0.5, enable_minmax=True):
        self.device = device
        self.lambda_match = lambda_match
        self.lambda_ano = lambda_ano
        self.dct_ratio = dct_ratio
        self.enable_minmax = enable_minmax

    def compute_watermark_loss(self, logits, watermark_code):
        """
        Compute watermark loss: encourage logits to align with code.
        Args:
            logits: Tensor [B, C]
            watermark_code: Tensor [C] or [B, C]

        Returns:
            wm_loss (Tensor): scalar loss (lower => better alignment)
            match_rate (float): fraction of batch exceeding threshold
        """
        # normalize vectors to compare direction
        logits_norm = F.normalize(logits, dim=1)
        code_norm  = F.normalize(watermark_code, dim=1) if watermark_code.dim()==2 else F.normalize(watermark_code.unsqueeze(0), dim=1)
        cos_sim = (logits_norm * code_norm).sum(dim=1)
        wm_loss = 1.0 - cos_sim.mean()
        match_rate = (cos_sim > 0.5).float().mean().item()
        return wm_loss, match_rate

    def compute_anomaly_score_dct(self, local_vec, clean_vec):
        """
        计算 DCT 低频部分相似度（支持反向传播）
        """
        # 保持在同一个设备上
        local_dct = torch_dct_1d(local_vec)
        clean_dct = torch_dct_1d(clean_vec)

        cutoff = int(self.dct_ratio * local_dct.shape[0])
        local_low = local_dct[:cutoff]
        clean_low = clean_dct[:cutoff]

        # 相似度（反向传播链保留）
        sim = F.cosine_similarity(local_low.unsqueeze(0), clean_low.unsqueeze(0), dim=1)
        ano_score = (1.0 - sim).mean()
        return ano_score

    def compute_anomaly_score(self, watermarked_model_vec, clean_model_vec):
        """
        Computes how much the watermarked model deviates from the clean model.
        Lower cosine similarity => more anomalous due to watermark.
        """
        w_vec = to_device(watermarked_model_vec, self.device)
        c_vec = to_device(clean_model_vec, self.device)
        sim = F.cosine_similarity(w_vec.unsqueeze(0), c_vec.unsqueeze(0), dim=1)
        ano_score = (1.0 - sim).mean()
        return ano_score

    def total_loss(self, ce_loss, logits, watermark_code, local_model_vec, global_model_vec):
        """
        Combine classification, watermark, and anomaly losses into final loss.
        """
        # ensure watermark_code on device
        watermark_code = to_device(watermark_code, self.device)
        wm_loss, match_rate = self.compute_watermark_loss(logits, watermark_code)
        # wm_loss, match_rate = 0.0, 0.0
        ano_score = self.compute_anomaly_score(local_model_vec, global_model_vec)
        # ano_score = 0
        # minimize -match_rate? but match_rate is static, we use wm_loss
        total = ce_loss + self.lambda_match * wm_loss + self.lambda_ano * ano_score
        # return total, {'wm_loss': wm_loss.item(), 'match_rate': match_rate, 'ano_score': ano_score.item()}
        return total, {'wm_loss': wm_loss, 'match_rate': match_rate, 'ano_score': ano_score}

    def minmax_train_step(self, model, clean_vec, images, labels, loss_func, optimizer, watermark_code,
                          model_to_vector_fn):
        """
        一次训练步骤，内部包含 min-max 博弈：
        - 用当前模型做 forward，计算正常 loss
        - 先模拟防御方最大化异常得分，再进行正常训练

        Args:
            model: 当前模型
            clean_vec: 干净模型向量 (作为防御者的参考标准)
            images: 输入图像
            labels: 标签
            loss_func: 交叉熵
            optimizer: 模型优化器
            watermark_code: 水印编码 [C]
            model_to_vector_fn: 用于提取当前模型向量（通常只包含FC层）

        Returns:
            loss值、调试信息字典
        """
        model.train()

        # Step 1: 防御视角 —— 最大化异常检测项（先模拟一次更新）
        if self.enable_minmax:
            optimizer.zero_grad()
            _, _ = model(images)  # dummy forward
            wm_vec = model_to_vector_fn(model,requires_grad=True)
            ano_score = self.compute_anomaly_score(wm_vec, clean_vec)
            (-ano_score).backward()  # 反向最大化异常分数
            optimizer.step()

        # Step 2: 正常训练 —— 最小化分类 + 水印 + 异常
        optimizer.zero_grad()
        _, logits = model(images)
        ce_loss = loss_func(logits, labels)
        wm_vec = model_to_vector_fn(model)
        total_loss, stats = self.total_loss(ce_loss, logits, watermark_code, wm_vec, clean_vec)
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), stats


class Detector:
    """
    Simple anomaly detector to measure anomaly scores on client updates.
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def is_anomalous(self, local_model_vec, global_model_vec):
        """
        Returns True if anomaly_score > threshold
        """
        sim = F.cosine_similarity(local_model_vec.unsqueeze(0), global_model_vec.unsqueeze(0), dim=1)
        ano_score = (1.0 - sim).item()
        return ano_score > self.threshold, ano_score

# Example usage:
# watermarker = MinMaxWatermarker(device=torch.device('cuda'), lambda_match=0.25, lambda_ano=0.1)
# ce = loss_fn(outputs, labels)
# local_vec = model_to_vector(local_model)
# global_vec = model_to_vector(global_model)
# total, stats = watermarker.total_loss(ce, outputs, watermark_code, local_vec, global_vec)
# loss.backward()
