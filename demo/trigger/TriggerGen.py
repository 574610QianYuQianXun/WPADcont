import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TriggerGenerator:
    def __init__(
        self,
        params,
        attack_succ_threshold=0.95,  # Lowered threshold for easier trigger finding
        regularization="l2",
        init_cost=1e-3,  # Reduced initial regularization cost
        lr=0.1,  # Lowered learning rate for stable optimizatio
        steps=100,  # Increased iteration steps
        target_label=None,  # Now None, fetched from params
        mode="feature"  # New parameter: pixel / feature
    ):
        """
        Trigger Generator

        Args:
            params: Parameter object, must contain 'task', 'device', 'num_classes', 'aim_target'
            attack_succ_threshold: Success rate threshold
            regularization: 'l1' or 'l2'
            init_cost: Initial value of regularization coefficient
            lr: Optimization learning rate
            steps: Number of optimization iterations
            target_label: Target label for attack, if None, fetched from params.aim_target
            mode: 'pixel' (pixel space) or 'feature' (feature space)
        """
        self.optimizer = None
        self.pattern_tensor = None
        self.mask_tensor = None
        self.delta_z = None  # Perturbation in feature space
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

        # Optimization 1: Correctly obtain target label
        if target_label is not None:
            self.target_label = target_label
        elif hasattr(params, 'aim_target'):
            self.target_label = params.aim_target
        else:
            self.target_label = 1  # Default value
            print(f"Warning: No target_label specified, using default value {self.target_label}")

        self.mode = mode

        # Dynamically set image size
        task = params.task.lower()
        if task == "mnist":
            self.img_channels, self.img_rows, self.img_cols = 1, 28, 28
        elif task == "cifar10":
            self.img_channels, self.img_rows, self.img_cols = 3, 32, 32
        else:
            raise ValueError(f"Unsupported task: {params.task}")

        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]
        self.mask_size = [self.img_rows, self.img_cols]

    def generate(self, model, tri_dataset=None, attack_size=100, batch_size=128):
        """
        Generate triggers in pixel or feature space based on mode
        """
        self.model = model.to(self.device).eval()

        # === Optimization 2: Improved data preparation ===
        if tri_dataset is None:
            x_benign = torch.rand(
                attack_size, self.img_channels, self.img_rows, self.img_cols,
                device=self.device
            )
            y_target = torch.full((attack_size,), self.target_label, device=self.device)
            dataset = torch.utils.data.TensorDataset(x_benign, y_target)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            # Use origin_target instead of hard-coded 1
            target_original_label = getattr(self.params, 'origin_target', 1)
            self.vis_image = None
            for x, y in tri_dataset:
                if y == target_original_label:
                    self.vis_image = x
                    break

            # Use only part of the data to avoid overfitting
            modified_dataset = []
            # count = 0
            for x, _ in tri_dataset:
                # if count >= attack_size:
                #     break
                modified_dataset.append((x.to(self.device), torch.tensor(self.target_label, device=self.device)))
                # count += 1

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

        # Optimization 3: Added early stopping mechanism
        best_step = 0
        patience = 20
        no_improve_count = 0

        # === Initialize parameters ===
        if self.mode == "pixel":
            # Pixel space: mask + pattern
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
            # Improved feature space initialization
            sample_batches = []
            for i, (x_batch, _) in enumerate(loader):
                if i >= 3:  # Use multiple batches to estimate feature dimension
                    break
                sample_batches.append(x_batch)

            if not sample_batches:
                raise ValueError("No data available for feature dimension estimation")

            # Initialize using the average of multiple batches
            with torch.no_grad():
                all_features = []
                for batch in sample_batches:
                    features, _ = self.model(batch.to(self.device))
                    all_features.append(features)
                avg_features = torch.cat(all_features, dim=0).mean(dim=0)
                feat_dim = avg_features.shape[0]

            print(f"[Feature] Estimated feature dimension: {feat_dim}")

            # Improved initialization strategy: small random perturbation
            init_std = 0.01  # Smaller initialization standard deviation
            self.delta_z = torch.normal(0, init_std, size=(feat_dim,), device=self.device, requires_grad=True)

            # Better optimizer settings
            self.optimizer = torch.optim.Adam([self.delta_z], lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
            # Add learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=10, verbose=False
            )

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # === Optimization iterations ===
        print(f"[{self.mode}] Starting optimization with target_label={self.target_label}")

        # 定义梯度对齐损失计算函数(局部函数)
        def compute_gradient_alignment_loss(z_adv, y_target):
            """
            计算梯度对齐损失
            核心思想: delta_z的方向应该与损失函数在特征空间的梯度方向一致
            """
            z_adv_clone = z_adv.detach().clone().requires_grad_(True)

            try:
                _, output_align = self.model(features=z_adv_clone)
                loss_ce_align = criterion(output_align, y_target)

                # 计算梯度
                grad_z = torch.autograd.grad(
                    outputs=loss_ce_align,
                    inputs=z_adv_clone,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                # delta_z方向(归一化)
                delta_z_dir = self.delta_z / (torch.norm(self.delta_z) + 1e-8)

                # 梯度方向(取负,沿梯度下降方向,对batch求平均后归一化)
                grad_dir = -grad_z.mean(dim=0)
                grad_dir = grad_dir / (torch.norm(grad_dir) + 1e-8)

                # 余弦相似度(越���近1越好)
                cosine_sim = torch.sum(delta_z_dir * grad_dir)
                alignment_loss = 1.0 - cosine_sim

                return alignment_loss

            except Exception as e:
                # 如果计算失败,返回0(不影响训练)
                return torch.tensor(0.0, device=self.device)

        for step in range(self.steps):
            ce_total, reg_total, acc_total, count = 0.0, 0.0, 0.0, 0

            for x_batch, y_batch in loader:
                if self.mode == "pixel":
                    # Pixel space forward
                    mask = torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5
                    mask = mask.repeat(self.img_channels, 1, 1)
                    pattern = torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5
                    x_adv = (1 - mask) * x_batch + mask * pattern
                    _, output = self.model(x_adv)

                elif self.mode == "feature":
                    # Optimization 6: Improved feature space forward pass
                    try:
                        features, _ = self.model(x_batch)
                    except Exception as e:
                        print(f"Error in model forward pass: {e}")
                        continue

                    # Ensure dimension match
                    if features.shape[1] != self.delta_z.shape[0]:
                        print(f"Warning: Feature dimension mismatch. Expected {self.delta_z.shape[0]}, got {features.shape[1]}")
                        continue

                    z_adv = features + self.delta_z.unsqueeze(0).expand_as(features)

                    try:
                        _, output = self.model(features=z_adv)
                    except Exception as e:
                        print(f"Error in model feature forward: {e}")
                        continue

                # loss calculation remains unchanged
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
                        # 原始正则化
                        l1 = self.delta_z.abs().sum()
                        l2 = (self.delta_z ** 2).sum()
                        max_norm = torch.max(torch.abs(self.delta_z))
                        # basic_reg = 0.2 * l1 + 0.8 * l2 + 0.1 * max_norm
                        basic_reg = l2

                        # 添加梯度对齐损失
                        alignment_loss = compute_gradient_alignment_loss(z_adv, y_batch)

                        # 组合正则化损失 (alignment_loss权重可调整: 0.1-0.5)
                        # loss_reg = basic_reg + 0.5 * alignment_loss
                        # loss_reg = alignment_loss
                        loss_reg = basic_reg

                loss = loss_ce + cost * loss_reg

                self.optimizer.zero_grad()
                loss.backward()

                # Optimization 7: Add gradient clipping
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

            # Update learning rate scheduler
            if self.mode == "feature":
                self.scheduler.step(avg_ce)

            # Update best trigger
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

            # Dynamically adjust cost
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

            # Optimization 8: Improved logging
            if step % 10 == 0 or step == self.steps - 1 or improved:
                current_lr = self.optimizer.param_groups[0]['lr'] if self.mode == "feature" else self.lr
                print(f"[{self.mode}] Step {step}: CE={avg_ce:.4f}, Reg={avg_reg:.4f}, "
                      f"Acc={avg_acc:.6f}, Cost={cost:.6f}, LR={current_lr:.6f}")
                if improved:
                    print(f"  └─ New best trigger found! (reg: {reg_best:.6f})")

            # Optimization 9: Early stopping mechanism
            if no_improve_count >= patience and avg_acc >= self.attack_succ_threshold:
                print(f"[{self.mode}] Early stopping at step {step} (no improvement for {patience} steps)")
                break

        print(f"Optimization finished. Best trigger: acc={acc_best:.4f}, reg={reg_best:.4f} (step {best_step})")

        # Visualization part
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
        Visualize the 1D delta_z feature trigger
        """
        if not hasattr(self, 'best_delta_z'):
            print("No best_delta_z found to visualize.")
            return

        delta_z = self.best_delta_z.detach().cpu().numpy().flatten()
        feat_dim = len(delta_z)

        print(f"Delta_z dimension: {feat_dim}, range: [{delta_z.min():.4f}, {delta_z.max():.4f}]")
        print(f"Mean: {delta_z.mean():.4f}, Std: {delta_z.std():.4f}")

        # Create visualization with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # 1. Original 1D vector waveform
        indices = np.arange(feat_dim)
        axes[0].plot(indices, delta_z, 'b-', linewidth=1, alpha=0.8)
        axes[0].scatter(indices[::max(1, feat_dim//100)], delta_z[::max(1, feat_dim//100)],
                       c='red', s=10, alpha=0.6)
        axes[0].set_title(f'Original Delta_z Waveform (dimension: {feat_dim})')
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('Delta_z Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # Mark max and min values
        max_idx = np.argmax(np.abs(delta_z))
        axes[0].scatter(max_idx, delta_z[max_idx], color='orange', s=100,
                       label=f'Max abs value: {delta_z[max_idx]:.4f}')
        axes[0].legend()

        # 2. Sorted value distribution
        sorted_indices = np.argsort(delta_z)
        sorted_values = delta_z[sorted_indices]
        axes[1].plot(range(feat_dim), sorted_values, 'g-', linewidth=2)
        axes[1].fill_between(range(feat_dim), sorted_values, 0,
                            where=sorted_values>=0, color='blue', alpha=0.3, label='Positive')
        axes[1].fill_between(range(feat_dim), sorted_values, 0,
                            where=sorted_values<0, color='red', alpha=0.3, label='Negative')
        axes[1].set_title('Sorted Value Distribution')
        axes[1].set_xlabel('Sorted Index')
        axes[1].set_ylabel('Delta_z Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 3. Value distribution histogram
        axes[2].hist(delta_z, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[2].axvline(delta_z.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {delta_z.mean():.4f}')
        axes[2].axvline(np.median(delta_z), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(delta_z):.4f}')
        axes[2].set_title('Value Distribution Histogram')
        axes[2].set_xlabel('Delta_z Value')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.suptitle(f'Delta_z 1D Vector Visualization\n'
                    f'Dim:{feat_dim}, Range:[{delta_z.min():.3f}, {delta_z.max():.3f}], '
                    f'Non-zero:{np.count_nonzero(delta_z)}/{feat_dim}', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Output key statistics
        print(f"\n=== Delta_z Statistics ===")
        print(f"Max value: {delta_z.max():.6f} (index: {np.argmax(delta_z)})")
        print(f"Min value: {delta_z.min():.6f} (index: {np.argmin(delta_z)})")
        print(f"Max abs value: {np.max(np.abs(delta_z)):.6f} (index: {np.argmax(np.abs(delta_z))})")
        print(f"Non-zero elements: {np.count_nonzero(delta_z)}/{feat_dim} ({np.count_nonzero(delta_z)/feat_dim*100:.1f}%)")
        print(f"Positive elements: {np.sum(delta_z > 0)} ({np.sum(delta_z > 0)/feat_dim*100:.1f}%)")
        print(f"Negative elements: {np.sum(delta_z < 0)} ({np.sum(delta_z < 0)/feat_dim*100:.1f}%)")

    def visualize_trigger(self, sample_input, mask, pattern):
        """
        Visualize pixel space trigger
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
        Save mask and pattern (pixel space only)
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
