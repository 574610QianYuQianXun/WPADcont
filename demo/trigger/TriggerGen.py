import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TriggerGenerator:
    def __init__(
        self,
        params,
        attack_succ_threshold=0.85,  # 降低阈值，更容易找到有效触发器
        regularization="l1",
        init_cost=1e-4,  # 降低初始正则化成本
        lr=0.05,  # 降低学习率，更稳定优化
        steps=150,  # 增加迭代步数
        target_label=None,  # 改为None，从params中获取
        mode="feature"  # 新增参数：pixel / feature
    ):
        """
        Trigger Generator

        Args:
            params: 参数对象，需包含 'task', 'device', 'num_classes', 'aim_target'
            attack_succ_threshold: 成功率阈值
            regularization: 'l1'或'l2'
            init_cost: 正则化系数初始值
            lr: 优化学习率
            steps: 优化迭代步数
            target_label: 攻击目标标签，如果为None则从params.aim_target获取
            mode: 'pixel' (像素空间) 或 'feature' (特征空间)
        """
        self.optimizer = None
        self.pattern_tensor = None
        self.mask_tensor = None
        self.delta_z = None  # 特征空间扰动
        self.model = None
        self.vis_image = None
        self.params = params
        self.device = params.device
        self.attack_succ_threshold = attack_succ_threshold
        self.regularization = regularization
        self.init_cost = init_cost
        self.lr = lr
        self.steps = steps
        self.epsilon = 1e-7

        # 优化1: 正确获取目标标签
        if target_label is not None:
            self.target_label = target_label
        elif hasattr(params, 'aim_target'):
            self.target_label = params.aim_target
        else:
            self.target_label = 1  # 默认值
            print(f"Warning: No target_label specified, using default value {self.target_label}")

        self.mode = mode

        # 动态设定图像尺寸
        task = params.task.lower()
        if task == "mnist":
            self.img_channels, self.img_rows, self.img_cols = 1, 28, 28
        elif task == "cifar10":
            self.img_channels, self.img_rows, self.img_cols = 3, 32, 32
        else:
            raise ValueError(f"Unsupported task: {params.task}")

        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]
        self.mask_size = [self.img_rows, self.img_cols]

    def generate(self, model, tri_dataset=None, attack_size=100, batch_size=64):
        """
        根据 mode 在像素空间或特征空间生成触发器
        """
        self.model = model.to(self.device).eval()

        # === 优化2: 改进数据准备 ===
        if tri_dataset is None:
            x_benign = torch.rand(
                attack_size, self.img_channels, self.img_rows, self.img_cols,
                device=self.device
            )
            y_target = torch.full((attack_size,), self.target_label, device=self.device)
            dataset = torch.utils.data.TensorDataset(x_benign, y_target)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            # 优化：使用origin_target而不是硬编码的1
            target_original_label = getattr(self.params, 'origin_target', 1)
            self.vis_image = None
            for x, y in tri_dataset:
                if y == target_original_label:
                    self.vis_image = x
                    break

            # 改进：只使用部分数据避免过拟合
            modified_dataset = []
            count = 0
            for x, _ in tri_dataset:
                if count >= attack_size:
                    break
                modified_dataset.append((x.to(self.device), torch.tensor(self.target_label, device=self.device)))
                count += 1

            if len(modified_dataset) == 0:
                print("Warning: No suitable data found in tri_dataset, using random data")
                x_benign = torch.rand(attack_size, self.img_channels, self.img_rows, self.img_cols, device=self.device)
                y_target = torch.full((attack_size,), self.target_label, device=self.device)
                modified_dataset = list(zip(x_benign, y_target))

            loader = torch.utils.data.DataLoader(modified_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.CrossEntropyLoss()
        reg_best = float('inf')
        acc_best = float('-inf')
        cost = self.init_cost
        cost_up_counter, cost_down_counter = 0, 0

        # 优化3: 添加早停机制
        best_step = 0
        patience = 20
        no_improve_count = 0

        # === 初始化参数 ===
        if self.mode == "pixel":
            # 像素空间：mask + pattern
            init_mask = np.random.uniform(0, 0.1, self.mask_size)
            init_pattern = np.random.rand(*self.pattern_size)
            init_mask = np.clip(init_mask, 0, 1)
            init_pattern = np.clip(init_pattern, 0, 1)

            self.mask_tensor = torch.tensor(
                np.arctanh((init_mask - 0.5) * (2 - self.epsilon)),
                dtype=torch.float,
                device=self.device,
                requires_grad=True
            )
            self.pattern_tensor = torch.tensor(
                np.arctanh((init_pattern - 0.5) * (2 - self.epsilon)),
                dtype=torch.float,
                device=self.device,
                requires_grad=True
            )
            self.optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor], lr=self.lr)

        elif self.mode == "feature":
            # 优化4: 改进特征空间初始化
            sample_batches = []
            for i, (x_batch, _) in enumerate(loader):
                if i >= 3:  # 使用多个batch来估计特征维度
                    break
                sample_batches.append(x_batch)

            if not sample_batches:
                raise ValueError("No data available for feature dimension estimation")

            # 使用多个batch的平均来初始化
            with torch.no_grad():
                all_features = []
                for batch in sample_batches:
                    features, _ = self.model(batch.to(self.device))
                    all_features.append(features)
                avg_features = torch.cat(all_features, dim=0).mean(dim=0)
                feat_dim = avg_features.shape[0]

            print(f"[Feature] Estimated feature dimension: {feat_dim}")

            # 改进的初始化策略：小幅随机扰动
            init_std = 0.01  # 更小的初始化标准差
            self.delta_z = torch.normal(0, init_std, size=(feat_dim,), device=self.device, requires_grad=True)

            # 优化5: 使用更好的优化器设置
            self.optimizer = torch.optim.Adam([self.delta_z], lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
            # 添加学习率调度器
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=10, verbose=False
            )

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # === 优化迭代 ===
        print(f"[{self.mode}] Starting optimization with target_label={self.target_label}")

        for step in range(self.steps):
            ce_total, reg_total, acc_total, count = 0.0, 0.0, 0.0, 0

            for x_batch, y_batch in loader:
                if self.mode == "pixel":
                    # 像素空间 forward
                    mask = torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5
                    mask = mask.repeat(self.img_channels, 1, 1)
                    pattern = torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5
                    x_adv = (1 - mask) * x_batch + mask * pattern
                    _, output = self.model(x_adv)

                elif self.mode == "feature":
                    # 优化6: 改进特征空间前向传播
                    try:
                        features, _ = self.model(x_batch)
                    except Exception as e:
                        print(f"Error in model forward pass: {e}")
                        continue

                    # 确保维度匹配
                    if features.shape[1] != self.delta_z.shape[0]:
                        print(f"Warning: Feature dimension mismatch. Expected {self.delta_z.shape[0]}, got {features.shape[1]}")
                        continue

                    z_adv = features + self.delta_z.unsqueeze(0).expand_as(features)

                    try:
                        _, output = self.model(features=z_adv)
                    except Exception as e:
                        print(f"Error in model feature forward: {e}")
                        continue

                # loss计算保持不变
                loss_ce = criterion(output, y_batch)
                if self.regularization == "l1":
                    if self.mode == "pixel":
                        loss_reg = mask.abs().sum() / self.img_channels
                    else:
                        loss_reg = self.delta_z.abs().sum()
                else:
                    if self.mode == "pixel":
                        loss_reg = (mask ** 2).sum() / self.img_channels
                    else:
                        loss_reg = (self.delta_z ** 2).sum()

                loss = loss_ce + cost * loss_reg

                self.optimizer.zero_grad()
                loss.backward()

                # 优化7: 添加梯度裁剪
                if self.mode == "feature":
                    torch.nn.utils.clip_grad_norm_([self.delta_z], max_norm=1.0)

                self.optimizer.step()

                acc = (output.argmax(dim=1) == y_batch).float().mean().item()
                ce_total += loss_ce.item()
                reg_total += loss_reg.item()
                acc_total += acc
                count += 1

            if count == 0:
                print(f"Warning: No valid batches processed at step {step}")
                continue

            avg_ce, avg_reg, avg_acc = ce_total / count, reg_total / count, acc_total / count

            # 更新学习率调度器
            if self.mode == "feature":
                self.scheduler.step(avg_ce)

            # 更新 best trigger
            improved = False
            if avg_acc >= self.attack_succ_threshold and avg_reg < reg_best:
                reg_best, acc_best = avg_reg, avg_acc
                best_step = step
                improved = True
                no_improve_count = 0

                if self.mode == "pixel":
                    self.best_mask = mask.detach().clone()
                    self.best_pattern = pattern.detach().clone()
                else:
                    self.best_delta_z = self.delta_z.detach().clone()
            else:
                no_improve_count += 1

            # 动态调整 cost
            if avg_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= 5:
                cost_up_counter = 0
                cost = cost * 1.5 if cost > 0 else self.init_cost
            elif cost_down_counter >= 5:
                cost_down_counter = 0
                cost /= 1.2

            # 优化8: 改进日志输出
            if step % 10 == 0 or step == self.steps - 1 or improved:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.mode == "feature" else self.lr
                print(f"[{self.mode}] Step {step}: CE={avg_ce:.4f}, Reg={avg_reg:.4f}, "
                      f"Acc={avg_acc:.6f}, Cost={cost:.6f}, LR={current_lr:.6f}")
                if improved:
                    print(f"  └─ New best trigger found! (reg: {reg_best:.6f})")

            # 优化9: 早停机制
            if no_improve_count >= patience and avg_acc >= self.attack_succ_threshold:
                print(f"[{self.mode}] Early stopping at step {step} (no improvement for {patience} steps)")
                break

        print(f"Optimization finished. Best trigger: acc={acc_best:.4f}, reg={reg_best:.4f} (step {best_step})")

        # 可视化部分
        if self.mode == "pixel" and hasattr(self, "best_mask"):
            self.visualize_trigger(self.vis_image, self.best_mask, self.best_pattern)

        elif self.mode == "feature" and hasattr(self, "best_delta_z"):
            self._visualize_feature_trigger()

        return (
            getattr(self, "best_mask", None),
            getattr(self, "best_pattern", None),
            getattr(self, "best_delta_z", None)
        )

    def _visualize_feature_trigger(self):
        """
        优化10: 改进特征空间触发器可视化
        """
        delta_z = self.best_delta_z.detach().cpu().numpy()
        norm = np.linalg.norm(delta_z)
        print(f"Feature trigger learned (Δz). Norm: {norm:.6f}, Mean: {delta_z.mean():.6f}, Std: {delta_z.std():.6f}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. 热力图
        if delta_z.ndim == 1:
            # 尝试重塑为更好的可视化格式
            if len(delta_z) >= 64:
                # 重塑为接近正方形
                side = int(np.sqrt(len(delta_z)))
                if side * side <= len(delta_z):
                    reshaped = delta_z[:side*side].reshape(side, side)
                    axes[0,0].imshow(reshaped, cmap='RdBu_r', aspect='auto')
                    axes[0,0].set_title(f"Feature Heatmap ({side}x{side})")
                else:
                    axes[0,0].imshow(delta_z.reshape(1, -1), cmap='RdBu_r', aspect='auto')
                    axes[0,0].set_title("Feature Vector")
            else:
                axes[0,0].imshow(delta_z.reshape(1, -1), cmap='RdBu_r', aspect='auto')
                axes[0,0].set_title("Feature Vector")
            axes[0,0].colorbar = plt.colorbar(axes[0,0].images[0], ax=axes[0,0])

        # 2. 直方图
        axes[0,1].hist(delta_z, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,1].set_title("Value Distribution")
        axes[0,1].set_xlabel("Δz value")
        axes[0,1].set_ylabel("Frequency")
        axes[0,1].axvline(delta_z.mean(), color='red', linestyle='--', label=f'Mean: {delta_z.mean():.4f}')
        axes[0,1].legend()

        # 3. 排序值
        sorted_values = np.sort(np.abs(delta_z))[::-1]
        top_k = min(50, len(sorted_values))
        axes[1,0].bar(range(top_k), sorted_values[:top_k], color='orange', alpha=0.7)
        axes[1,0].set_title(f"Top {top_k} Absolute Values")
        axes[1,0].set_xlabel("Feature Index (sorted)")
        axes[1,0].set_ylabel("|Δz| value")

        # 4. 特征重要性（前20个最大绝对值的特征）
        abs_values = np.abs(delta_z)
        top_indices = np.argsort(abs_values)[::-1][:20]
        axes[1,1].barh(range(len(top_indices)), delta_z[top_indices],
                       color=['red' if x < 0 else 'blue' for x in delta_z[top_indices]])
        axes[1,1].set_title("Top 20 Important Features")
        axes[1,1].set_xlabel("Δz value")
        axes[1,1].set_ylabel("Feature Index")
        axes[1,1].set_yticks(range(len(top_indices)))
        axes[1,1].set_yticklabels([f'F{i}' for i in top_indices])
        axes[1,1].axvline(0, color='black', linestyle='-', alpha=0.3)

        plt.suptitle(f"Feature-space Trigger Analysis (Norm: {norm:.4f})", fontsize=14)
        plt.tight_layout()
        plt.show()

    def visualize_trigger(self, sample_input, mask, pattern):
        """
        可视化像素空间触发器
        """
        if sample_input is None:
            return
        sample_input = sample_input.unsqueeze(0).to(self.device)
        sample_mask = mask.unsqueeze(0)
        sample_pattern = pattern.unsqueeze(0)
        x_triggered = (1 - sample_mask) * sample_input + sample_mask * sample_pattern

        def to_numpy(img_tensor):
            img = img_tensor.squeeze().detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            return img.numpy().clip(0, 1)

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        axs[0].imshow(to_numpy(sample_input))
        axs[0].set_title("Original")
        axs[1].imshow(to_numpy(sample_mask), cmap='gray')
        axs[1].set_title("Mask")
        axs[2].imshow(to_numpy(sample_pattern))
        axs[2].set_title("Pattern")
        axs[3].imshow(to_numpy(x_triggered))
        axs[3].set_title("Triggered")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_trigger(mask: torch.Tensor, pattern: torch.Tensor, prefix: str = 'trigger'):
        """
        保存 mask 和 pattern（仅像素空间）
        """
        save_dir = os.path.dirname(prefix)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        mask_np = mask[0].detach().cpu().numpy()
        plt.imsave(f"{prefix}_mask.png", mask_np, cmap='hot')

        pattern_np = pattern.detach().cpu().numpy()
        if pattern_np.shape[0] == 1:
            pattern_np = pattern_np[0]
            plt.imsave(f"{prefix}_pattern.png", pattern_np, cmap='gray')
        else:
            pattern_np = pattern_np.transpose(1, 2, 0)
            plt.imsave(f"{prefix}_pattern.png", np.clip(pattern_np, 0, 1))

        print(f"Saved: {prefix}_mask.png, {prefix}_pattern.png")
