import torch
import torch.nn.functional as F
from sympy.physics.vector.printing import params
from torch import nn, optim


class FndShield:
    def __init__(self,params, global_model, malicious_model, distill_loader):
        """
        初始化负蒸馏器
        Args:
            global_model: 当前全局模型（需要被训练）
            malicious_model: 恶意客户端聚合得到的模型（固定）
            distill_loader: 用于负蒸馏的服务器数据集（DataLoader）
            device: 'cpu' or 'cuda'
            lambda_distill: 负蒸馏强度超参数
            lr: 负蒸馏训练的学习率
        """
        self.params = params
        self.global_model = global_model
        self.malicious_model = malicious_model

        self.distill_loader = distill_loader
        self.device = params.device
        self.lambda_distill = params.lambda_distill
        self.lr = params.lr

        self.global_model.to(self.device)
        self.malicious_model.to(self.device)
        self.malicious_model.eval()  # 恶意模型固定，不更新权重

        self.optimizer = optim.Adam(self.global_model.parameters(), lr=self.lr)

    def compute_negative_distillation_loss(self, outputs_global, outputs_malicious):
        """
        计算负蒸馏损失
        目标：让全局模型的输出分布远离恶意模型的输出分布
        """
        # 使用 KL 散度，方向是全局 -> 恶意
        kl_div = F.kl_div(
            input=F.log_softmax(outputs_global, dim=1),
            target=F.softmax(outputs_malicious, dim=1),
            reduction='batchmean'
        )
        # 负蒸馏：最大化 KL 散度，因此取负
        negative_loss = -kl_div
        return negative_loss

    def train(self):
        """
        在服务器上进行负蒸馏训练
        Args:
            epochs: 负蒸馏的训练轮数
        """
        self.global_model.train()

        for epoch in range(self.params.epochs):
            total_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(self.distill_loader):
                inputs = inputs.to(self.device)

                # 前向传播
                outputs_global = self.global_model(inputs)
                with torch.no_grad():
                    outputs_malicious = self.malicious_model(inputs)

                # 计算负蒸馏损失
                loss = self.lambda_distill * self.compute_negative_distillation_loss(outputs_global, outputs_malicious)

                # 反向传播 + 参数更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.distill_loader)
            print(f"[Negative Distillation] Epoch {epoch + 1}/{self.params.epochs}, Avg Loss: {avg_loss:.6f}")

