import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import utils


class WatermarkSystem:
    def __init__(self, params, client_id, embed_ratio=0.15, beta=0.5, gamma=0.2):
        """
        独立的水印系统模块
        params: 全局参数
        client_id: 客户端ID
        embed_ratio: DCT低频嵌入比例 (0~1)
        beta: 水印强度系数
        gamma: 异常约束系数
        """
        self.params = params
        self.id = client_id
        self.embed_ratio = embed_ratio
        self.beta = beta
        self.gamma = gamma
        self.watermark_code = self.generate_code()
        self.dct_ratio = embed_ratio

    def generate_code(self):
        """生成实数水印码"""
        min_dim = max(16, self.params.malicious * self.params.clients,
                      self.params.num_classes)
        H_dim = 2 ** int(np.ceil(np.log2(min_dim)))

        # 使用实数Hadamard矩阵
        H = self._hadamard(H_dim)
        row_idx = (self.id * 997 + 1) % H_dim
        code = H[row_idx, :self.params.num_classes]

        # 归一化到[0,1]
        code = (code - np.min(code)) / (np.max(code) - np.min(code) + 1e-8)
        return torch.tensor(code, dtype=torch.float32).to(self.params.device)

    def _hadamard(self, n):
        """生成Hadamard矩阵"""
        if n == 1:
            return np.array([[1]])
        H = self._hadamard(n // 2)
        return np.block([[H, H], [H, -H]])

    def torch_dct_1d(self, x, inverse=False):
        """实数DCT变换"""
        N = x.size(1)
        device = x.device

        if not inverse:
            n = torch.arange(N, device=device).float()[None, :]
            k = torch.arange(N, device=device).float()[:, None]
            dct_mat = torch.cos(torch.pi / N * (n + 0.5) * k)
            dct_mat[:, 0] *= 1 / math.sqrt(N)
            dct_mat[:, 1:] *= math.sqrt(2.0 / N)
            return x @ dct_mat.T
        else:
            k = torch.arange(N, device=device).float()[:, None]
            n = torch.arange(N, device=device).float()[:, None]
            idct_mat = torch.cos(torch.pi / N * k * (n + 0.5))
            idct_mat[0, :] *= 1 / math.sqrt(N)
            idct_mat[1:, :] *= math.sqrt(2.0 / N)
            return x @ idct_mat

    def dct_embed(self, logits, watermark):
        """频域水印嵌入"""
        B, C = logits.shape
        k = max(1, int(C * self.embed_ratio))

        logits_dct = self.torch_dct_1d(logits.float())
        low_freq = logits_dct[:, :k].clone()

        # 幅度调整
        mag_low = torch.norm(low_freq, dim=0, keepdim=True)
        wm_mag = watermark[:k]
        low_freq = (low_freq / (mag_low + 1e-8)) * wm_mag.unsqueeze(0)

        wm_logits_dct = torch.cat([low_freq, logits_dct[:, k:]], dim=1)
        return self.torch_dct_1d(wm_logits_dct, inverse=True)

    def watermark_loss(self, logits):
        """水印损失计算"""
        logits_dct = self.torch_dct_1d(logits.float())
        k = max(1, int(logits_dct.size(1) * self.embed_ratio))
        logits_low = logits_dct[:, :k]

        mag = torch.norm(logits_low, dim=0)
        wm_mag = self.watermark_code[:k]

        mag_loss = 1 - F.cosine_similarity(mag.unsqueeze(0),
                                           wm_mag.unsqueeze(0))
        match_rate = (1 - mag_loss).clamp(0, 1).item()

        return mag_loss, match_rate

    def compute_anomaly_score(self, local_model, clean_model):
        """
        计算模型参数DCT域异常度
        """
        local_vec = utils.model_to_vector_fc(local_model, self.params)
        clean_vec = utils.model_to_vector_fc(clean_model, self.params)

        local_dct = self.torch_dct_1d(local_vec.unsqueeze(0))
        clean_dct = self.torch_dct_1d(clean_vec.unsqueeze(0))

        cutoff = int(self.dct_ratio * local_dct.size(1))
        local_low = local_dct[:, :cutoff]
        clean_low = clean_dct[:, :cutoff]

        sim = F.cosine_similarity(local_low, clean_low)
        return 1 - sim.item()

    def extract_watermark(self, model, test_loader):
        """
        频域水印提取与匹配
        """
        model.eval()
        total_matches = 0
        total_samples = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.params.device)
                _, logits = model(images)

                wm_loss, match_rate = self.watermark_loss(logits)
                total_matches += (match_rate > 0.6) * images.size(0)
                total_samples += images.size(0)

        return total_matches / total_samples