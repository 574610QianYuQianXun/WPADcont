import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.transforms import transforms
from scipy.linalg import hadamard
from attack import frequency_backdoor, How_backdoor_promax, DBA, SADBA_Adaptive_Manager
from clients.BaseClient import BaseClient
from torch.utils.data import DataLoader
import copy
from attack import How_backdoor
from clients.Minmax_Watermark import MinMaxWatermarker
from clients.WatermarkModule import WatermarkSystem
from utils.utils import show_image, TinyImageNet
import torch.nn.functional as F
from utils import utils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import math, random
from PIL import Image
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

        self.backdoor_indices = []  # 先初始化为空

        """
        A3FL 超参
        """
        self.a3fl_delta = None  # 可训练触发器，形状和 input_shape 一致，在 A3FL mask 区域生效
        self.a3fl_mask = None  # A3FL 专用 mask，不用 self.mask
        self.a3fl_patch = None  # 最终导出的 patch = a3fl_mask * a3fl_delta
        self.a3fl_lambda_adv = getattr(self.params, "a3fl_lambda_adv", 1.0)
        self.a3fl_K_outer = getattr(self.params, "a3fl_K_outer", 3)
        self.a3fl_K_trigger = getattr(self.params, "a3fl_K_trigger", 3)
        self.a3fl_lr_delta = getattr(self.params, "a3fl_lr_delta", 0.1)
        self.a3fl_lr_adv = getattr(self.params, "a3fl_lr_adv", self.params.lr)
        """
        """

        if self.params.task == "MNIST":
            self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        elif self.params.task == "CIFAR10":
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                  std=[0.2023, 0.1994, 0.2010])
        elif self.params.task == "CIFAR100":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                  std=[0.2675, 0.2565, 0.2761])
        elif self.params.task == "ImageNet":
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])
        self.Make_pattern()
        if self.params.poison_type==1:
            if self.params.attack_type=='dct':
                frequency_backdoor(
                    train_set=self.backdoor_dataset,
                    # origin_target=self.params.origin_target,
                    aim_target=self.params.aim_target
                    # strength=0.2,
                    # dct_positions=[(4, 5), (5, 4)]
                )
            elif self.params.attack_type in ['How_backdoor', 'DarkFed']:
                self.backdoor_indices = How_backdoor(self.backdoor_dataset, self.params.origin_target, self.params.aim_target)
                # How_backdoor_promax(self.backdoor_dataset, self.params.origin_target, self.params.aim_target)
            elif self.params.attack_type=='dba':
                self.backdoor_indices = DBA(self.backdoor_dataset, self.params.origin_target, self.params.aim_target,self.id)

            elif self.params.attack_type == 'sadba':
                all_indices = list(self.backdoor_dataset.idxs)
                num_poison = int(math.floor(len(all_indices) * 0.5))
                random.seed(42)
                self.backdoor_indices = random.sample(all_indices, num_poison)
                print(
                    f"Client {self.id}: SADBA initialized, selected "
                    f"{len(self.backdoor_indices)} samples to poison dynamically.")

        self.m_train_loader = DataLoader(self.backdoor_dataset, batch_size=self.params.local_bs, shuffle=True)
        if self.params.agg == "FLShield":
            self.val_loader = DataLoader(self.normal_dataset, batch_size=self.params.local_bs, shuffle=False)

        self.choice_loss = 1
        # 如果想要显示前 16 张拼图
        if self.params.attack_type != 'sadba':
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

    def apply_sadba_dynamic_trigger(self, current_model):
        self.backdoor_dataset = copy.deepcopy(self.normal_dataset)
        input_shape = self.normal_dataset[0][0].shape

        # 初始化 Manager (不需要传 mask 了)
        sadba_mgr = SADBA_Adaptive_Manager(current_model, self.params.aim_target, self.params.device, input_shape)
        target_centroid = sadba_mgr.get_target_centroid(self.normal_dataset.dataset, self.normal_dataset.idxs)

        real_dataset = self.backdoor_dataset.dataset
        base_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]

        # ========================================================
        # 1. 生成所有可能的 mask 组合 (共 15 种)
        # ========================================================
        import itertools
        # 生成 (0,0,0,1) 到 (1,1,1,1)
        all_combinations = list(itertools.product([0, 1], repeat=4))
        # 过滤掉全 0 的情况
        valid_masks = [m for m in all_combinations if sum(m) > 0]
        num_masks = len(valid_masks)  # 应该等于 15

        print(f"Client {self.id}: Distributing data among {num_masks} trigger combinations.")

        count = 0
        for global_idx in self.backdoor_indices:
            current_mask = valid_masks[self.id % num_masks]

            best_dy, best_dx = 0, 0

            # --- 数据读取逻辑 (TinyImageNet / CIFAR) ---
            if isinstance(real_dataset, TinyImageNet):
                try:
                    img_path = real_dataset.data[global_idx]
                    from PIL import Image
                    img_pil = Image.open(img_path).convert('RGB')
                    img_tensor = transforms.ToTensor()(img_pil)
                    img_tensor = self.normalize(img_tensor).to(self.params.device)

                    # 【关键】传入 current_mask 计算最佳位置
                    # 现在的逻辑是：寻找最适合 "当前这几种子触发器" 的位置
                    best_dy, best_dx = sadba_mgr.find_best_position_for_sample(img_tensor, target_centroid,
                                                                               current_mask)

                    self.params.sadba_recorded_positions.add((best_dy, best_dx))
                except Exception:
                    continue
            else:
                data_item = real_dataset.data[global_idx]
                img_tensor = None

                # Tensor/Numpy 转换逻辑 (同之前，为了节省篇幅简写，但要确保逻辑完整)
                if isinstance(data_item, torch.Tensor):
                    img_tensor = data_item.float()
                    if img_tensor.ndim == 2:
                        img_tensor = img_tensor.unsqueeze(0)
                    elif img_tensor.ndim == 3 and img_tensor.shape[2] <= 4:
                        img_tensor = img_tensor.permute(2, 0, 1)
                elif isinstance(data_item, np.ndarray):
                    img_tensor = torch.from_numpy(data_item).float()
                    if img_tensor.ndim == 2:
                        img_tensor = img_tensor.unsqueeze(0)
                    elif img_tensor.ndim == 3 and img_tensor.shape[2] <= 4:
                        img_tensor = img_tensor.permute(2, 0, 1)

                if img_tensor is not None:
                    if img_tensor.max() > 1.1: img_tensor = img_tensor / 255.0
                    img_tensor = self.normalize(img_tensor).to(self.params.device)

                    # 【关键】传入 current_mask
                    best_dy, best_dx = sadba_mgr.find_best_position_for_sample(img_tensor, target_centroid,
                                                                               current_mask)
                    self.params.sadba_recorded_positions.add((best_dy, best_dx))
                # --- 写入数据逻辑 ---
                if isinstance(data_item, torch.Tensor):
                    modified_data = data_item.clone()
                elif isinstance(data_item, np.ndarray):
                    modified_data = data_item.copy()
                else:
                    continue

                for k, (r, c) in enumerate(base_positions):
                    # 【关键】写入时，只写 mask 里对应为 1 的块
                    if current_mask[k] == 0:
                        continue

                    rr, cc = r + best_dy, c + best_dx
                    if rr < modified_data.shape[0] and (cc + 4) <= modified_data.shape[1]:
                        try:
                            if modified_data.ndim == 3:
                                modified_data[rr, cc:cc + 4, :] = 255
                            else:
                                modified_data[rr, cc:cc + 4] = 255
                        except Exception:
                            pass

                real_dataset.data[global_idx] = modified_data

            # 修改标签
            try:
                if isinstance(real_dataset.targets, torch.Tensor):
                    real_dataset.targets[global_idx] = self.params.aim_target
                elif isinstance(real_dataset.targets, list):
                    real_dataset.targets[global_idx] = int(self.params.aim_target)
                else:
                    real_dataset.targets[global_idx] = self.params.aim_target
            except:
                pass

            count += 1

        visualize_backdoor_samples(self.backdoor_dataset, n_samples=16, nrow=4)
        print(f"Client {self.id} [SADBA]: Applied dynamic triggers to {count} images.")
        self.m_train_loader = DataLoader(self.backdoor_dataset, batch_size=self.params.local_bs, shuffle=True)

    def train_model(self, model, dataloader, loss_func,teacher_model=None,mask=None,pattern=None,delta_z=None,predicted_model=None):
        """
        Standard training loop for a given model and dataloader.
        """
        model.train()
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

                # poison_type==2 用原来的 mask+pattern，A3FL 单独处理
                if self.params.poison_type == 2 and self.params.attack_type != 'a3fl':
                    poisoning_index = self.Implant_trigger(inputs, labels)

                # A3FL：在攻击阶段，对当前 batch 动态注入自适应触发器
                if self.params.attack_type == 'a3fl' and self.epoch >= self.params.attack_epoch:
                    self.apply_a3fl_trigger_inplace(inputs, labels)

                # _,outputs = model(inputs)
                # 前向提取特征
                features, outputs = model(inputs)
                # if delta_z is not None:
                #     new_features = features + delta_z.unsqueeze(0).expand_as(features)
                #     _, outputs = model(features=new_features)
                # 分类损失
                loss_cls = loss_func(outputs, labels)

                if self.params.attack_type=='DarkFed' and self.epoch>=self.params.attack_epoch:
                    # DarkFed控制
                    eu_loss = utils.Euclidean_loss(model, self.global_model, self.params)
                    cos_loss = utils.Cos_loss(local_model=model, predicted_model=predicted_model, global_model=self.global_model, params=self.params)
                    loss = loss_cls + 2 * eu_loss + 2 * cos_loss

                else:
                    loss = loss_cls

                loss.backward()
                optimizer.step()
                last_loss = loss.item()

        return model, last_loss

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

    def local_train(self, loss_func, epoch,teacher_model=None,win=6,mask=None,pattern=None,delta_z=None,predicted_model=None):
        """
        Local training for malicious client.
        Depending on the training epoch, chooses backdoor or benign training.
        Also更新历史模型参数记录，用于后续的水印检测和调整。
        """
        self.epoch = epoch
        # === A3FL: 自适应后门本地训练 ===
        if self.params.attack_type == 'a3fl':
            # A3FL 不用预先构造 backdoor_dataset，直接在干净 dataloader 上训练
            dataloader = self.train_loader

            # 到达攻击轮之后，先根据当前 global_model 优化触发器
            if epoch >= self.params.attack_epoch:
                self.optimize_a3fl_trigger()

            local_model = copy.deepcopy(self.global_model)
            local_model, last_loss = self.train_model(
                local_model, dataloader, loss_func,
                teacher_model=teacher_model, mask=mask, pattern=pattern,
                delta_z=delta_z, predicted_model=predicted_model
            )
            return local_model, last_loss

        if self.params.poison_type == 1 and self.params.attack_type == 'sadba':
            if epoch >= self.params.attack_epoch:
                # 传入当前的 global_model (self.global_model)
                self.apply_sadba_dynamic_trigger(self.global_model)
                dataloader = self.m_train_loader
            else:
                dataloader = self.train_loader

        elif self.params.poison_type==1:
            if epoch >= self.params.attack_epoch:
                dataloader = self.m_train_loader
            else:
                dataloader = self.train_loader
        else:
            dataloader=self.train_loader

        local_model = copy.deepcopy(self.global_model)
        local_model, last_loss = self.train_model(local_model, dataloader, loss_func, teacher_model=teacher_model,mask=mask,pattern=pattern,delta_z=delta_z,predicted_model=predicted_model)
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

    # === A3FL: 初始化触发器 ===
    def init_a3fl_mask(self):
        """
        A3FL 专用 mask：

        - 不使用 self.mask，也不依赖 pattern_tensor
        - 默认：整张图全 1，触发器可以作用在任意位置
        - 如果你想要“任意形状”的触发区域，可以在外面手动改 self.a3fl_mask 的 0/1 分布
          例如：
              mc.a3fl_mask = torch.zeros_like(mc.a3fl_mask)
              mc.a3fl_mask[:, 3:8, 5:10] = 1     # 矩形
              mc.a3fl_mask[:, some_bool_mask] = 1 # 任意形状
        """
        if self.a3fl_mask is not None:
            return

        self.a3fl_mask = torch.ones(self.input_shape, device=self.params.device)


    def init_a3fl_delta(self):
        self.init_a3fl_mask()
        # 如果内存中已有，跳过
        if self.a3fl_delta is not None:
            return

        # =======================================================
        # 读取逻辑：从 saved_triggers/{task}/client_{id}/ 读取
        # =======================================================
        save_dir = os.path.join("saved_triggers", self.params.task, f"client_{self.id}")
        file_name = "a3fl_trigger_checkpoint.pt"
        trigger_path = os.path.join(save_dir, file_name)

        if os.path.exists(trigger_path):
            try:
                print(f"[Client {self.id}] Loading trigger from {trigger_path}...")
                checkpoint = torch.load(trigger_path, map_location=self.params.device)

                loaded_delta = checkpoint['delta'].to(self.params.device)
                loaded_delta.requires_grad_(True)  # 恢复梯度

                self.a3fl_delta = loaded_delta

                if 'mask' in checkpoint:
                    self.a3fl_mask = checkpoint['mask'].to(self.params.device)

                return  # 加载成功

            except Exception as e:
                print(f"[Client {self.id}] Error loading trigger: {e}. Falling back to random init.")
        else:
            print(f"[Client {self.id}] No checkpoint found at {trigger_path}. Starting fresh.")

        # =======================================================
        # 随机初始化
        # =======================================================
        self.a3fl_delta = torch.zeros(self.input_shape, device=self.params.device)
        self.a3fl_delta.uniform_(-0.1, 0.1)
        self.a3fl_delta.requires_grad_(True)

    # 在一个 batch 上应用触发器（用于触发器优化阶段，默认对所有样本都加）
    def apply_a3fl_delta(self, x):
        self.init_a3fl_delta()
        delta = self.a3fl_delta.to(self.params.device)
        mask = self.a3fl_mask.to(self.params.device)
        x_p = x + mask * delta
        x_p = torch.clamp(x_p, 0.0, 1.0)
        return x_p

    # 在训练时，对一个 batch 按 poisoning_proportion 注入 A3FL 触发器 + 改标签
    def apply_a3fl_trigger_inplace(self, data, labels):
        """
        data: [B, C, H, W] (已经在 device 上)
        labels: [B]
        按 poisoning_proportion 给前 n 个样本打上自适应触发器，并改成 aim_target
        """
        if self.a3fl_delta is None:
            # 还没优化过触发器就算了，直接返回
            return

        n = int(len(data) * self.poisoning_proportion)
        index = list(range(0, n))  # 和 Implant_trigger 类似，只是不 +1 了
        for i in index:
            if labels[i] == self.params.aim_target:
                continue
            x_i = data[i].unsqueeze(0)          # [1, C, H, W]
            x_i = self.apply_a3fl_delta(x_i)   # 加触发器
            data[i] = x_i.squeeze(0)
            labels[i] = self.params.aim_target

    # 关键：在本地根据当前 global_model 优化 (delta, adv_model)
    def optimize_a3fl_trigger(self):
        """
        A3FL 触发器优化：
        - fixed_model: 当前 global_model 的一份拷贝，不更新参数，用来评估攻击损失
        - adv_model: 另一份拷贝，用来做“解除后门”的对抗模型
        - a3fl_delta: 可训练触发器，在 mask 区域生效
        """
        import copy as _copy
        import torch.nn as nn

        device = self.params.device
        self.init_a3fl_delta()

        # 拷贝两份模型
        fixed_model = _copy.deepcopy(self.global_model).to(device)
        adv_model = _copy.deepcopy(self.global_model).to(device)

        # 固定 fixed_model 参数，只让梯度传到输入/触发器
        for p in fixed_model.parameters():
            p.requires_grad_(False)
        fixed_model.eval()
        adv_model.train()

        opt_delta = torch.optim.SGD([self.a3fl_delta], lr=self.a3fl_lr_delta)
        opt_adv = torch.optim.SGD(adv_model.parameters(), lr=self.a3fl_lr_adv)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.a3fl_K_outer):
            for images, labels, _ in self.train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # ---------- 内层：更新触发器 delta ----------
                for _ in range(self.a3fl_K_trigger):
                    opt_delta.zero_grad()

                    x_p = self.apply_a3fl_delta(images)
                    target_labels = torch.full_like(labels, self.params.aim_target)

                    # 在 fixed_model (当前全局模型) 上的攻击损失
                    _, logits_fixed = fixed_model(x_p)
                    loss_fixed = criterion(logits_fixed, target_labels)

                    # 在对抗模型 adv_model 上的攻击损失
                    _, logits_adv = adv_model(x_p)
                    loss_adv = criterion(logits_adv, target_labels)

                    loss_attack = loss_fixed + self.a3fl_lambda_adv * loss_adv
                    loss_attack.backward()
                    opt_delta.step()

                    # 限制触发器幅度，防止图像太爆炸
                    with torch.no_grad():
                        self.a3fl_delta.clamp_(-0.5, 0.5)

                # ---------- 外层：更新 adv_model (解除后门) ----------
                opt_adv.zero_grad()
                # 当前触发器固定，只更新 adv_model 参数
                x_p_clean = self.apply_a3fl_delta(images).detach()
                x_p_clean.requires_grad_(True)
                _, logits_clean = adv_model(x_p_clean)
                loss_unlearn = criterion(logits_clean, labels)
                loss_unlearn.backward()
                opt_adv.step()

        with torch.no_grad():
            # 把“可以直接加到图片上的补丁”存进 params，方便测试使用
            patch = (self.a3fl_mask * self.a3fl_delta).detach().cpu()
            self.a3fl_patch= patch
            self.params.a3fl_patch = patch
            # 也顺便保存一下 poison 比例，测试时复用
            self.params.a3fl_poisoning_proportion = self.poisoning_proportion

            if self.params.save_model:
                # =======================================================
                # 保存逻辑：路径结构变为 saved_triggers/{task}/client_{id}/
                # =======================================================
                # 1. 构建包含 task 的目录路径
                save_dir = os.path.join("saved_triggers", self.params.task, f"client_{self.id}")

                # 2. 确保目录存在
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                # 3. 定义文件名 (因为目录已经区分了 task，文件名可以简单点，也可以保留)
                file_name = "a3fl_trigger_checkpoint.pt"
                trigger_path = os.path.join(save_dir, file_name)

                save_data = {
                    'delta': self.a3fl_delta.detach().cpu(),
                    'mask': self.a3fl_mask.detach().cpu(),
                    'epoch': self.epoch,
                    'task': self.params.task
                }

                torch.save(save_data, trigger_path)
                print(f"[Client {self.id}] Saved trigger to {trigger_path}")

            try:
                self.visualize_a3fl_trigger(
                    show=True  # 如果你不想每轮弹窗，把这里改成 False
                )
            except Exception as e:
                print(f"[Client {self.id}] Failed to visualize A3FL trigger: {e}")



    def visualize_a3fl_trigger(self, save_path=None, show=True):
        """
        可视化 A3FL 学到的触发器 patch（self.a3fl_patch）.

        - 会做一次 min-max 归一化到 [0,1]，方便看图
        - C=1 时用灰度图，C=3 时用 RGB
        - save_path 不为 None 时会保存成图片文件
        """
        if self.a3fl_patch is None:
            print(f"[Client {self.id}] A3FL patch is None, nothing to visualize.")
            return

        patch = self.a3fl_patch.detach().cpu()  # [C,H,W]
        if patch.dim() != 3:
            print(f"[Client {self.id}] Unexpected patch shape: {patch.shape}")
            return

        # min-max 归一化到 [0,1]
        p_min = patch.min()
        p_max = patch.max()
        if float(p_max - p_min) < 1e-8:
            # 全常数就直接变成 0.5 灰，防止全黑
            patch_norm = torch.zeros_like(patch) + 0.5
        else:
            patch_norm = (patch - p_min) / (p_max - p_min)

        # 根据通道数选择显示方式
        C, H, W = patch_norm.shape

        plt.figure(figsize=(3, 3))
        if C == 1:
            img_show = patch_norm.squeeze(0).numpy()
            plt.imshow(img_show, cmap="gray")
        elif C == 3:
            # [C,H,W] -> PIL
            img_show = TF.to_pil_image(patch_norm)
            plt.imshow(img_show)
        else:
            # 非 1/3 通道，用第一通道凑合看一下
            img_show = patch_norm[0].numpy()
            plt.imshow(img_show, cmap="gray")

        plt.title(f"A3FL trigger (Client {self.id})")
        plt.axis("off")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


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

    def visualize_a3fl_trigger(self, save_path=None, show=True):
        """
        可视化 A3FL 学到的触发器 patch（self.a3fl_patch）.

        - 会做一次 min-max 归一化到 [0,1]，方便看图
        - C=1 时用灰度图，C=3 时用 RGB
        - save_path 不为 None 时会保存成图片文件
        """
        if self.a3fl_patch is None:
            print(f"[Client {self.id}] A3FL patch is None, nothing to visualize.")
            return

        patch = self.a3fl_patch.detach().cpu()  # [C,H,W]
        if patch.dim() != 3:
            print(f"[Client {self.id}] Unexpected patch shape: {patch.shape}")
            return

        # min-max 归一化到 [0,1]
        p_min = patch.min()
        p_max = patch.max()
        if float(p_max - p_min) < 1e-8:
            # 全常数就直接变成 0.5 灰，防止全黑
            patch_norm = torch.zeros_like(patch) + 0.5
        else:
            patch_norm = (patch - p_min) / (p_max - p_min)

        # 根据通道数选择显示方式
        C, H, W = patch_norm.shape

        plt.figure(figsize=(3, 3))
        if C == 1:
            img_show = patch_norm.squeeze(0).numpy()
            plt.imshow(img_show, cmap="gray")
        elif C == 3:
            # [C,H,W] -> PIL
            img_show = TF.to_pil_image(patch_norm)
            plt.imshow(img_show)
        else:
            # 非 1/3 通道，用第一通道凑合看一下
            img_show = patch_norm[0].numpy()
            plt.imshow(img_show, cmap="gray")

        plt.title(f"A3FL trigger (Client {self.id})")
        plt.axis("off")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

