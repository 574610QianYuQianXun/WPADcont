import torch

from utils import utils


class BaseClient:
    def __init__(self, client_id, params, global_model, train_dataset=None, data_idxs=None):
        self.id = client_id
        self.params = params
        self.global_model = global_model
        self.dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.n_data = len(self.dataset)

    def train_model(self, model, dataloader, loss_func,teacher_model=None,mask=None,pattern=None,delta_z=None,predicted_model=None):
        """
        Standard training loop for a given model and dataloader.
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr, momentum=self.params.momentum)
        last_loss = None
        # === ✅ 添加余弦退火调度器 ===
        # T_max 设置为 local_ep * len(dataloader)，即一个完整训练周期（所有 batch）
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.params.local_ep * len(dataloader),
        #     eta_min=1e-6  # 可调最小学习率
        # )

        for _ in range(self.params.local_ep):
            for images, labels, _ in dataloader:
                optimizer.zero_grad()
                inputs = images.to(device=self.params.device, non_blocking=True)
                labels = labels.to(device=self.params.device, non_blocking=True)
                features, outputs = model(inputs)
                if delta_z is not None:
                    new_features = features + delta_z.unsqueeze(0).expand_as(features)
                    _, outputs = model(features=new_features)

                # with torch.no_grad():
                #     _, teacher_outputs = teacher_model(inputs)
                #     # 1. 计算相似度（使用KL散度或余弦相似度）
                # similarity = F.cosine_similarity(
                #     F.softmax(outputs, dim=1),
                #     F.softmax(teacher_outputs, dim=1),
                #     dim=1
                # ).min()  # 取batch最小相似度
                # # 或使用KL散度（需确保数值稳定性）
                # # similarity = -F.kl_div(
                # #     F.log_softmax(outputs, dim=1),
                # #     F.softmax(teacher_outputs, dim=1),
                # #     reduction='batchmean'
                # #)
                # # similarity = torch.exp(similarity)  # 转换为[0,1]范围
                #
                # # 2. 动态权重分配（可调整非线性变换参数k）
                # k = 10.0  # 敏感度参数，越大则权重对相似度变化越敏感
                # w_local = torch.sigmoid(k * (similarity - 0.5))  # 将相似度映射到权重
                # w_distill = 1 - w_local
                # # 3. 损失函数组合
                # loss_ce = F.cross_entropy(outputs, labels)  # 本地监督损失
                # loss_kl = F.kl_div(
                #     F.log_softmax(outputs, dim=1),
                #     F.softmax(teacher_outputs, dim=1),
                #     reduction='batchmean'
                # )  # 蒸馏损失
                #
                # kl_div_per_sample = F.kl_div(
                #     F.log_softmax(teacher_outputs, dim=1),
                #     F.softmax(outputs, dim=1),
                #     reduction='none'  # 不聚合，返回 (batch_size, num_classes)
                # )
                # # Step 2: 把每个样本的类别loss求和（每一行求和）
                # kl_div_per_sample = kl_div_per_sample.sum(dim=0)  # shape: (batch_size,)
                #
                # # Step 3: 在 batch 里找到最小的KL
                # min_kl_loss, min_idx = torch.max(kl_div_per_sample, dim=0)
                #
                # loss = w_local * loss_ce + w_distill * loss_kl
                # # 输出相关参数
                #
                # # print(f"Similarity: {similarity.item():.4f}, w_local: {w_local.item():.4f}, w_distill: {w_distill.item():.4f}")

                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                # scheduler.step()  # ✅ 每个 batch 更新学习率

                last_loss = loss.item()
        return model, last_loss




