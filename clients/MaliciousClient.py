import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.transforms import transforms
from scipy.linalg import hadamard
from attack import frequency_backdoor, How_backdoor_promax, DBA
from clients.BaseClient import BaseClient
from torch.utils.data import DataLoader
import copy
from attack import How_backdoor
from clients.Minmax_Watermark import MinMaxWatermarker
from clients.WatermarkModule import WatermarkSystem
from utils.utils import show_image
import torch.nn.functional as F
from utils import utils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

# 全局标志
_has_visualized = False

class MaliciousClient(BaseClient):
    def __init__(self, client_id, params, global_model, train_dataset=None, data_idxs=None):
        super().__init__(client_id, params, global_model, train_dataset, data_idxs)

        # 初始化水印系统
        # self.watermarker = WatermarkSystem(params, self.id)

        self.visualize = 0
        self.match_rate_after_agg = None
        self.match_rate_before_agg = None
        self.epoch = None
        self.pattern_tensor = torch.tensor([
            [1., -10., 1.],
            [-10., 1., -10.],
            [-10., -10., -10.],
            [-10., 1., -10.],
            [1., -10., 1.]])
        self.x_top = 3
        self.y_top = 23
        self.mask_value = -10
        self.poisoning_proportion = 0.2
        self.mask = None
        self.pattern = None
        self.normal_dataset = self.dataset
        self.backdoor_dataset = copy.deepcopy(self.normal_dataset)
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.params.local_bs, shuffle=True)
        self.input_shape = self.normal_dataset[0][0].shape

        if self.params.task=="MNIST":
            self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        if self.params.task == "CIFAR10":
            self.normalize= transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                 std=[0.2023, 0.1994, 0.2010])
        self.Make_pattern()
        if self.params.poison_type==1:
            if self.params.attack_type=='dct':
                frequency_backdoor(
                    train_set=self.backdoor_dataset,
                    origin_target=self.params.origin_target,
                    aim_target=self.params.aim_target
                    # strength=0.2,
                    # dct_positions=[(4, 5), (5, 4)]
                )
            if self.params.attack_type=='How_backdoor':
                self.backdoor_indices = How_backdoor(self.backdoor_dataset, self.params.origin_target, self.params.aim_target)
                # How_backdoor_promax(self.backdoor_dataset, self.params.origin_target, self.params.aim_target)
            if self.params.attack_type=='dba':
                self.backdoor_indices = DBA(self.backdoor_dataset, self.params.origin_target, self.params.aim_target,self.id)

        self.m_train_loader = DataLoader(self.backdoor_dataset, batch_size=self.params.local_bs, shuffle=True)
        if self.params.agg == "FLShield":
            self.val_loader = DataLoader(self.normal_dataset, batch_size=self.params.local_bs, shuffle=False)

        # show_image(self.normal_dataset.dataset[0])

        # 为每个客户端生成正交水印码字
        # self.watermark_code = self.generate_watermark_code()
        self.choice_loss = 1


        # 显示 1 张
        # visualize_backdoor_samples(self.backdoor_dataset, n_samples=1)

        # 如果想要显示前 16 张拼图
        visualize_backdoor_samples(self.backdoor_dataset, n_samples=16, nrow=4)

    def generate_watermark_code(self):
        """
        基于非对称正交化的水印码生成
        改进点：
        1. 跳过全1行避免全局冲突
        2. 将-1替换为0实现非对称
        3. 动态强度归一化
        """
        # 1. 计算Hadamard矩阵维度（最小2的幂）
        min_dim = max(self.params.malicious * self.params.clients,
                      self.params.num_classes)
        H_dim = 2 ** int(np.ceil(np.log2(min_dim)))

        # 2. 生成Hadamard矩阵（跳过全1行）
        H = hadamard(H_dim)

        # 3. 非对称化处理（关键修改）
        row_idx = ((self.id * 997) % (H_dim - 1)) + 1  # 跳过第0行（全1行）
        code = H[row_idx, :self.params.num_classes].astype(np.float32)

        # 4. 符号优化（将-1替换为0）
        code[code < 0] = 0  # 非对称正交化核心步骤

        # 5. 动态归一化（保持信号强度）
        active_elements = np.sum(code != 0)
        if active_elements > 0:
            code /= np.sqrt(active_elements)  # 按激活元素数归一化

        return torch.tensor(code).to(self.params.device)

    def train_model(self, model, dataloader, loss_func,teacher_model=None,mask=None,pattern=None,delta_z=None):
        """
        Standard training loop for a given model and dataloader.
        """
        model.train()

        # 冻结全连接层
        # for name, param in model.named_parameters():
        #     if name.startswith('fc'):  # 冻结所有以'fc'开头的层
        #         param.requires_grad_(False)
        # 优化所有未冻结参数
        # optimizer = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=self.params.lr,
        #     momentum=self.params.momentum
        # )

        # 👉 收集特征/标签/后门标记用于可视化
        all_features = []
        all_labels = []
        all_is_backdoor = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr, momentum=self.params.momentum)
        last_loss = None
        for _ in range(self.params.local_ep):
            for images, labels, global_indices in dataloader:
                optimizer.zero_grad()
                inputs = images.to(device=self.params.device, non_blocking=True)
                labels = labels.to(device=self.params.device, non_blocking=True)

                # 判断哪些是后门样本
                is_backdoor = torch.tensor(
                    [idx in self.backdoor_indices for idx in global_indices],
                    dtype=torch.bool,
                    device=self.params.device
                )

                if self.params.poison_type==2:
                    poisoning_index = self.Implant_trigger(inputs, labels)
                # _,outputs = model(inputs)
                # 前向提取特征
                features, outputs = model(inputs)
                # if delta_z is not None:
                #     new_features = features + delta_z.unsqueeze(0).expand_as(features)
                #     _, outputs = model(features=new_features)

                # 分类损失
                loss_cls = loss_func(outputs, labels)

                # SNNL 损失：只计算后门与正常样本之间
                # loss_snnl = self.snnl_between_backdoor_and_normal(
                #     features, labels, is_backdoor, temperature=0.05
                # )
                #
                # loss_align = self.snnl_align_with_target_class(
                #     features, labels, is_backdoor, self.params.aim_target
                # )

                # 总损失（权重可调）
                # loss = loss_cls + 1.0 * loss_snnl
                # loss = loss_cls + 3.0 * loss_align
                loss = loss_cls

                # 在训练阶段加入
                # 保证向量维度一致
                # target_code = self.watermark_code.to(outputs.device).unsqueeze(0).expand_as(outputs)  # [B, C]
                # logits = outputs  # [B, C]
                #
                # logits_norm = F.normalize(logits, dim=1)
                # target_code_norm = F.normalize(target_code, dim=1)
                # cos_sim = (logits_norm * target_code_norm).sum(dim=1)  # 与 cosine_similarity 等效
                # # # 单样本计算余弦相似度（越大方向越一致）
                # # cos_sim = F.cosine_similarity(logits, target_code, dim=1)  # shape: [B]376
                # watermark_loss = 1 - cos_sim.mean()  # 越大方向越一致，我们取负作为损失

                # loss_cls = loss_func(outputs, labels)
                # # if self.choice_loss == 0:
                # #     loss = loss_cls + 0.25 * watermark_loss  # 调整权重
                # # else:
                # loss = loss_cls  # 调整权重

                loss.backward()

                # # 初始化
                # wm = MinMaxWatermarker(device=self.params.device, lambda_match=1.0, lambda_ano=0.8)
                #
                # # 训练循环中
                # ce = loss_func(outputs, labels)
                # local_vec = utils.model_to_vector(model, self.params)  # 保证返回 torch.Tensor
                # global_vec = utils.model_to_vector(self.global_model, self.params)
                # total, stats = wm.total_loss(ce, outputs, self.watermark_code, local_vec, global_vec)
                #
                # # total=ce
                #
                # total.backward()
                optimizer.step()
                last_loss = loss.item()

                # 👉 收集用于 t-SNE 的信息（移动到 CPU）
                all_features.append(features.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_is_backdoor.append(is_backdoor.detach().cpu())

        # ✅ 训练完成后可视化
        features_np = torch.cat(all_features, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        is_backdoor_np = torch.cat(all_is_backdoor, dim=0).numpy()
        
        if self.visualize==1 and self.id==63 :
            visualize_tsne(
                features_np, labels_np, is_backdoor_np,
                title=f't-SNE - Client (poison={self.params.poison_type})',
                save=self.params.save_tsne if hasattr(self.params, 'save_tsne') else False,
                save_path=f'tsne_client_{self.client_id}.png' if hasattr(self, 'client_id') else None
            )

        return model, last_loss

    # def extract_watermark(self, model, test_loader, threshold=0.6):
    #     """
    #     检测客户端模型是否带有当前客户端对应的水印码。
    #
    #     Args:
    #         model: 待检测的模型
    #         test_loader: 用于测试的样本数据
    #         threshold: 余弦相似度阈值，大于该值视为“水印匹配”
    #     Returns:
    #         匹配率（匹配样本 / 总样本）
    #     """
    #     model.eval()
    #     match_count = 0
    #     total_count = 0
    #
    #     wm_code = F.normalize(self.watermark_code.view(1, -1), dim=1).to(self.params.device)
    #
    #     with torch.no_grad():
    #         for images, _ in test_loader:
    #             inputs = images.to(self.params.device)
    #             _, logits = model(inputs)  # 输出为 [B, D]
    #             logits = F.normalize(logits, dim=1)  # 归一化到单位球面
    #
    #             # [B, 1] = [B, D] · [D, 1]，再变成 [B]
    #             sim = torch.matmul(logits, wm_code.T).view(-1)
    #
    #             match_count += (sim > threshold).sum().item()
    #             total_count += logits.size(0)
    #
    #     return match_count / total_count if total_count > 0 else 0.0

    # def train_model(self, global_model, dataloader, loss_func, teacher_model=None):
    #     """
    #     两阶段训练：
    #     1. 先训练干净模型 clean_model（无水印）
    #     2. 拷贝 clean_model，继续训练带水印模型 watermarked_model
    #     返回水印模型和最后损失
    #     """
    #     import copy
    #
    #     device = self.params.device
    #     wm = MinMaxWatermarker(device=device, lambda_match=1.0, lambda_ano=100)
    #
    #     # --------------------
    #     # 1. 训练干净模型
    #     # --------------------
    #     clean_model = copy.deepcopy(global_model).to(device)
    #     clean_model.train()
    #     optimizer_clean = torch.optim.SGD(clean_model.parameters(), lr=self.params.lr, momentum=self.params.momentum)
    #
    #     for _ in range(self.params.local_ep):
    #         for images, labels in dataloader:
    #             optimizer_clean.zero_grad()
    #             images = images.to(device, non_blocking=True)
    #             labels = labels.to(device, non_blocking=True)
    #             if self.params.poison_type == 2:
    #                 _ = self.Implant_trigger(images, labels)  # 这里保证触发器植入逻辑一致
    #             _, logits_clean = clean_model(images)
    #             loss_clean = loss_func(logits_clean, labels)
    #             loss_clean.backward()
    #             optimizer_clean.step()
    #
    #     # 计算干净模型参数向量（不参与梯度）
    #     clean_vec = utils.model_to_vector_fc(clean_model, self.params,requires_grad=True)
    #
    #     # --------------------
    #     # 2. 训练带水印模型
    #     # --------------------
    #     watermarked_model = copy.deepcopy(clean_model).to(device)
    #     watermarked_model.train()
    #
    #     # # 冻结全连接层
    #     # for name, param in watermarked_model.named_parameters():
    #     #     if name.startswith('fc'):  # 冻结所有以'fc'开头的层
    #     #         param.requires_grad_(False)
    #     # # 优化所有未冻结参数
    #     # optimizer_wm = torch.optim.SGD(
    #     #     filter(lambda p: p.requires_grad, watermarked_model.parameters()),
    #     #     lr=self.params.lr,
    #     #     momentum=self.params.momentum
    #     # )
    #
    #     optimizer_wm = torch.optim.SGD(watermarked_model.parameters(), lr=self.params.lr, momentum=self.params.momentum)
    #     last_loss = None
    #     for _ in range(self.params.local_ep):
    #         for images, labels in dataloader:
    #             optimizer_wm.zero_grad()
    #             images = images.to(device, non_blocking=True)
    #             labels = labels.to(device, non_blocking=True)
    #             if self.params.poison_type == 2:
    #                 _ = self.Implant_trigger(images, labels)
    #
    #             loss, stats = wm.minmax_train_step(
    #                 model=watermarked_model,
    #                 clean_vec=clean_vec,
    #                 images=images,
    #                 labels=labels,
    #                 loss_func=loss_func,
    #                 optimizer=optimizer_wm,
    #                 watermark_code=self.watermark_code,
    #                 model_to_vector_fn=lambda m, requires_grad=False: utils.model_to_vector_fc(m, self.params,
    #                                                                                            requires_grad=requires_grad)
    #             )
    #             last_loss = loss
    #
    #     return watermarked_model, last_loss
    #
    # def extract_watermark(self, model, test_loader):
    #     model.eval()
    #     match_count = 0
    #     total_count = 0
    #
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             inputs = images.to(self.params.device)
    #             _, logits = model(inputs)
    #             logits = F.normalize(logits, dim=1)
    #
    #             # 当前客户端的水印码字（单位向量）
    #             wm = self.watermark_code.view(1, -1)
    #
    #             # 计算余弦相似度
    #             sim = torch.matmul(logits, wm.t()).squeeze(1)
    #
    #             # 与自己水印相似度 > 阈值 视为匹配
    #             match_count += torch.sum(sim > 0.5).item()
    #             total_count += logits.size(0)
    #
    #     return match_count / total_count

    # def train_model(self, global_model, dataloader, loss_func,teacher_model=None):
    #     device = self.params.device
    #     clean_model = copy.deepcopy(global_model).to(device)
    #     optimizer = torch.optim.SGD(clean_model.parameters(),
    #                                 lr=self.params.lr,
    #                                 momentum=self.params.momentum)
    #
    #     # === 阶段1: 纯净模型训练 ===
    #     for _ in range(self.params.local_ep):
    #         for images, labels in dataloader:
    #             images, labels = images.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #             _, logits = clean_model(images)
    #             loss = loss_func(logits, labels)
    #             loss.backward()
    #             optimizer.step()
    #
    #     # === 阶段2: 水印植入 ===
    #     wm_model = copy.deepcopy(clean_model)
    #     optimizer = torch.optim.SGD(wm_model.parameters(),
    #                                 lr=self.params.lr,
    #                                 momentum=self.params.momentum)
    #
    #     for _ in range(int(self.params.local_ep)):
    #         for images, labels in dataloader:
    #             images, labels = images.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #
    #             # 前向传播并嵌入水印
    #             _, logits = wm_model(images)
    #             wm_logits = self.watermarker.dct_embed(logits,
    #                                                    self.watermarker.watermark_code)
    #
    #             # 计算损失函数
    #             cls_loss = loss_func(logits, labels)
    #             wm_loss, _ = self.watermarker.watermark_loss(wm_logits)
    #             total_loss = cls_loss + self.watermarker.beta * wm_loss
    #             total_loss.backward()
    #             optimizer.step()
    #
    #     # === 阶段3: 异常混淆训练 ===
    #     for _ in range(int(self.params.local_ep)):
    #         for images, labels in dataloader:
    #             images, labels = images.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #
    #             # 正常前向传播
    #             _, logits = wm_model(images)
    #
    #             # 计算复合损失
    #             cls_loss = loss_func(logits, labels)
    #             wm_loss, _ = self.watermarker.watermark_loss(logits)
    #             ano_score = self.watermarker.compute_anomaly_score(wm_model, clean_model)
    #             total_loss = (cls_loss +
    #                           self.watermarker.beta * wm_loss +
    #                           self.watermarker.gamma * ano_score)
    #             total_loss.backward()
    #             optimizer.step()
    #
    #     return wm_model, total_loss.item()

    def snnl_between_backdoor_and_normal(self, features, labels, backdoor_mask, temperature=0.1):
        """
        计算后门样本与正常样本之间的 Soft Nearest Neighbor Loss（SNNL）。

        参数：
        - features: [B, D] 特征向量（建议已归一化）
        - labels:   [B] 标签
        - backdoor_mask: [B] 布尔张量，True 表示后门样本
        - temperature: 控制相似度平滑度
        """
        if backdoor_mask.sum() == 0 or (~backdoor_mask).sum() == 0:
            return torch.tensor(0.0, device=features.device)

        backdoor_feat = features[backdoor_mask]
        normal_feat = features[~backdoor_mask]
        backdoor_labels = labels[backdoor_mask]
        normal_labels = labels[~backdoor_mask]

        all_features = torch.cat([backdoor_feat, normal_feat], dim=0)
        all_labels = torch.cat([backdoor_labels, normal_labels], dim=0)

        all_features = F.normalize(all_features, p=2, dim=1)

        sim_matrix = torch.matmul(all_features, all_features.T)  # [N, N]
        exp_sim = torch.exp(sim_matrix / temperature)

        # ✅ 替代 fill_diagonal_，不破坏 autograd 计算图
        mask = ~torch.eye(exp_sim.size(0), device=exp_sim.device).bool()
        exp_sim = exp_sim * mask

        label_matrix = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()

        numerator = (exp_sim * label_matrix).sum(dim=1)
        denominator = exp_sim.sum(dim=1) + 1e-8

        snnl_loss = -torch.log(numerator / denominator + 1e-8)
        return snnl_loss.mean()

    def snnl_align_with_target_class(self, features, labels, backdoor_mask, target_class: int):
        """
        让后门样本特征对齐目标类别特征中心（特征空间靠近目标类）。

        参数:
        - features: [B, D]，batch 中所有样本的特征
        - labels: [B]，对应标签
        - backdoor_mask: [B]，布尔张量，标记哪些是后门样本
        - target_class: int，目标类别标签（通常为 aim_target）

        返回:
        - 对齐损失：鼓励后门样本特征接近目标类中心
        """
        if backdoor_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # 获取后门样本特征
        backdoor_features = features[backdoor_mask]  # [N_bd, D]

        # 仅提取“非后门且标签为目标类”的样本特征
        target_mask = (labels == target_class) & (~backdoor_mask)
        if target_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        target_features = features[target_mask]

        # 求目标类特征中心
        target_center = target_features.mean(dim=0, keepdim=True)  # [1, D]

        # 计算余弦相似度或欧氏距离
        # （下面用余弦相似度方式实现）
        backdoor_features = F.normalize(backdoor_features, dim=1)
        target_center = F.normalize(target_center, dim=1)

        cos_sim = F.cosine_similarity(backdoor_features, target_center, dim=1)  # [N_bd]
        loss = 1 - cos_sim.mean()  # 越接近目标类中心越好

        return loss

    def extract_watermark(self, model, test_loader):
        """调用水印系统的提取方法"""
        return self.watermarker.extract_watermark(model, test_loader)

    def local_train(self, loss_func, epoch,teacher_model=None,win=6,mask=None,pattern=None,delta_z=None):
        """
        Local training for malicious client.
        Depending on the training epoch, chooses backdoor or benign training.
        Also更新历史模型参数记录，用于后续的水印检测和调整。
        """
        self.epoch = epoch
        if self.params.poison_type==1:
            if epoch >= self.params.attack_epoch:
                dataloader = self.m_train_loader
            else:
                dataloader = self.train_loader
        else:
            dataloader=self.train_loader

        # if self.id in self.params.backdoor_clients[:len(self.params.backdoor_clients) // 2]:
        #     self.choice_loss = 0
        #     dataloader = self.train_loader
        # else:
        #     self.choice_loss = 1

        local_model = copy.deepcopy(self.global_model)
        local_model, last_loss = self.train_model(local_model, dataloader, loss_func, teacher_model=teacher_model,mask=mask,pattern=pattern,delta_z=delta_z)
        return local_model, last_loss

    def get_watermark_positions(self, pred_class, num_classes, client_id, watermark_length):
        """
        确定性位置选择算法
        """
        # 1. 排除预测类别
        all_positions = list(range(num_classes))
        all_positions.remove(pred_class)

        # 2. 使用客户端ID生成随机种子
        seed = client_id % 1000000  # 确保在合理范围内
        rng = np.random.RandomState(seed)

        # 3. 固定顺序排列
        rng.shuffle(all_positions)

        # 4. 选择前watermark_length个位置
        return sorted(all_positions[:watermark_length])  # 排序确保顺序一致
    # 注入后门
    def Implant_trigger(self, data, label):
        n = int(len(data) * self.poisoning_proportion)
        index = list(range(0, n + 1))
        poisoning_index = []
        for i in index:
            if label[i] == self.params.aim_target:
                continue
            else:
                data[i] = (1 - self.mask) * data[i] + self.mask * self.pattern
                label[i] = self.params.aim_target
                poisoning_index.append(i)

        return poisoning_index

    def Make_pattern(self):
        full_image = torch.zeros(self.input_shape)
        full_image.fill_(self.mask_value)
        x_bot = self.x_top + self.pattern_tensor.shape[0]
        y_bot = self.y_top + self.pattern_tensor.shape[1]

        if x_bot >= self.input_shape[1] or y_bot >= self.input_shape[2]:
            raise ValueError(...)

        full_image[:, self.x_top:x_bot, self.y_top:y_bot] = self.pattern_tensor
        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        self.pattern = self.normalize(full_image).to(self.params.device)

    # 可视化 logits 输出为条形图
    @staticmethod
    def visualize_logits(outputs, labels, epoch=0, batch_idx=0, save_dir="logits_vis", max_samples=5, prefix="train"):
        """
        可视化 logits 输出为条形图。

        参数:
        - outputs: 模型的原始输出 logits，形状 [B, C]
        - labels: 对应标签
        - epoch: 当前 epoch（用于命名）
        - batch_idx: 当前 batch 索引
        - save_dir: 保存目录
        - max_samples: 最多可视化的样本数量
        - prefix: 文件名前缀（区分 train/test 等）
        """
        os.makedirs(save_dir, exist_ok=True)

        logits = outputs.detach().cpu()
        labels = labels.detach().cpu()

        N = min(max_samples, logits.shape[0])
        for i in range(N):
            plt.figure(figsize=(6, 3))
            plt.bar(range(logits.shape[1]), logits[i].numpy(), color='skyblue')
            plt.title(f"[{prefix}] Epoch {epoch} Batch {batch_idx} | Label: {labels[i].item()}")
            plt.xlabel("Class Index")
            plt.ylabel("Logit Value")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{prefix}_logits_e{epoch}_b{batch_idx}_s{i}.png")
            plt.savefig(save_path)
            plt.close()


def visualize_tsne(features, labels, is_backdoor, title='t-SNE Feature Map', save=False, save_path=None):
    """
    t-SNE 可视化：正常样本按标签着色，后门样本为黑色。
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    perplexity = min(30, len(features) - 1)  # 确保合法
    if perplexity < 5:
        print(f"[Skip t-SNE] Too few samples: {len(features)}")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))

    normal_labels = labels[~is_backdoor]
    unique_normal_labels = np.unique(normal_labels)
    colormap = plt.cm.get_cmap('tab20', len(unique_normal_labels))

    for i, label in enumerate(unique_normal_labels):
        idxs = (labels == label) & (~is_backdoor)
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1],
                    label=f'Class {label}',
                    color=colormap(i),
                    s=12, alpha=0.7)

    # 后门样本统一为黑色
    idxs_bd = is_backdoor
    plt.scatter(reduced[idxs_bd, 0], reduced[idxs_bd, 1],
                label='Backdoor',
                color='black',
                s=15, alpha=0.8, marker='x')

    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def visualize_backdoor_samples(dataset, n_samples=1, nrow=4):
    """
    可视化后门数据集中的样本

    Args:
        dataset (Dataset): 后门数据集
        n_samples (int): 要展示多少张图片
        nrow (int): 每行显示多少张（当 n_samples > 1 时生效）
    """
    global _has_visualized
    if _has_visualized:
        return  # 已经显示过，直接跳过
    _has_visualized = True
    if len(dataset) == 0:
        print("Dataset is empty, cannot visualize.")
        return

    # 限制展示数量
    n_samples = min(n_samples, len(dataset))

    images, labels = [], []
    for i in range(n_samples):
        img, label, global_idx = dataset[i]
        labels.append(f"L{label}|Idx{global_idx}")
        images.append(img)

    if n_samples == 1:
        # 单张图
        img = images[0]
        if torch.is_tensor(img):
            if img.shape[0] == 1:  # MNIST
                img_show = TF.to_pil_image(img.squeeze(0))
                cmap = "gray"
            else:  # CIFAR10
                img_show = TF.to_pil_image(img)
                cmap = None
        else:
            img_show = img
            cmap = None

        plt.imshow(img_show, cmap=cmap)
        plt.title(labels[0])
        plt.axis("off")
        plt.show()

    else:
        # 多张拼成 grid
        grid = make_grid(images, nrow=nrow, normalize=True, scale_each=True)
        plt.figure(figsize=(nrow * 2, (n_samples // nrow + 1) * 2))
        plt.imshow(TF.to_pil_image(grid))
        plt.title(" | ".join(labels))
        plt.axis("off")
        plt.show()

