import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        elif task == "cifar10" or task == "cifar100":
            self.img_channels, self.img_rows, self.img_cols = 3, 32, 32
        elif task == "imagenet":
            self.img_channels, self.img_rows, self.img_cols = 3, 224, 224
        else:
            raise ValueError(f"Unsupported task: {params.task}")

        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]
        self.mask_size = [self.img_rows, self.img_cols]

    def generate(self, model, tri_dataset=None, attack_size=100, batch_size=64):
        """
        Generate triggers in pixel or feature space based on mode
        """
        self.model = model.to(self.device).eval()

        # === Step 1: Prepare trigger generation dataset ===
        loader, ori_loader, self.vis_image = self._prepare_trigger_data(tri_dataset, attack_size, batch_size)

        criterion = nn.CrossEntropyLoss()
        reg_best, acc_best = float('inf'), float('-inf')
        cost = self.init_cost
        cost_up_counter, cost_down_counter = 0, 0

        best_step, patience, no_improve_count = 0, 20, 0

        # === Step 2: Initialize optimization parameters ===
        if self.mode == "pixel":
            self._init_pixel_space()
        elif self.mode == "feature":
            self._init_feature_space(loader)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # === Step 3: Optimization iterations ===
        print(f"[{self.mode}] Starting optimization with target_label={self.target_label}")

        for step in range(self.steps):
            avg_loss, avg_ce, avg_reg, avg_anti_sim_loss, avg_cos_sim, avg_acc, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            # --- Compute target class feature center (used for anti-similarity constraint) ---
            if ori_loader is not None:
                all_feats = []
                with torch.no_grad():
                    for x_t, _ in ori_loader:
                        x_t = x_t.to(self.device)
                        f_t, _ = self.model(x_t)
                        if f_t.dim() > 2:
                            f_t = f_t.view(f_t.size(0), -1)
                        all_feats.append(f_t)
                target_center = torch.cat(all_feats, dim=0).mean(dim=0).detach()
            else:
                target_center = None

            for x_batch, y_batch in loader:
                if self.mode == "pixel":
                    loss_ce, loss_reg, acc = self._optimize_pixel_step(x_batch, y_batch, criterion, cost)
                else:
                    loss, loss_ce, loss_reg, anti_sim_loss, cos_sim, acc = self._optimize_feature_step(
                        x_batch, y_batch, criterion, cost, target_center=target_center
                    )

                avg_loss += loss
                avg_ce += loss_ce
                avg_reg += loss_reg
                avg_anti_sim_loss += anti_sim_loss
                avg_cos_sim += cos_sim
                avg_acc += acc
                count += 1

            if count == 0:
                print(f"Warning: No valid batches processed at step {step}")
                continue

            avg_loss, avg_ce, avg_reg, avg_anti_sim_loss, avg_acc, avg_cos_sim = avg_loss / count, avg_ce / count, avg_reg / count, avg_anti_sim_loss / count, avg_acc / count, avg_cos_sim / count

            # scheduler update for feature mode
            if self.mode == "feature":
                self.scheduler.step(avg_ce)

            # === Step 4: Save best trigger ===
            improved = False
            if avg_acc >= self.attack_succ_threshold and avg_reg < reg_best:
                reg_best, acc_best = avg_reg, avg_acc
                best_step = step
                improved = True
                no_improve_count = 0
                if self.mode == "pixel":
                    self.best_mask = self.current_mask.detach().clone()
                    self.best_pattern = self.current_pattern.detach().clone()
                else:
                    self.best_delta_z = self.delta_z.detach().clone()
            else:
                no_improve_count += 1

            # === Step 5: Dynamic cost adjustment ===
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

            current_lr = (
                self.optimizer.param_groups[0]["lr"] if self.mode == "feature" else self.lr
            )
            print(
                f"[{self.mode}] Step {step}: CE={avg_ce:.6f}, Reg={avg_reg:.6f}, "
                f"Acc={avg_acc:.6f}, Cost={cost:.6f}, LR={current_lr:.6f}, "
                f"Loss={avg_loss:.6f}, Anti-Sim={avg_anti_sim_loss:.6f}, Cos-Sim={avg_cos_sim:.6f}"
            )
            if improved:
                print(f"  └─ New best trigger found! (reg: {reg_best:.6f})")

            # Early stopping
            if no_improve_count >= patience and avg_acc >= self.attack_succ_threshold:
                print(f"[{self.mode}] Early stopping at step {step} (no improvement for {patience} steps)")
                break

        print(f"Optimization finished. Best trigger: acc={acc_best:.4f}, reg={reg_best:.4f} (step {best_step})")

        # === Step 6: Visualization ===
        if self.mode == "pixel" and hasattr(self, "best_mask"):
            self.visualize_trigger(self.vis_image, self.best_mask, self.best_pattern)
        elif self.mode == "feature" and hasattr(self, "best_delta_z"):
            self._visualize_feature_trigger()

        return (
            getattr(self, "best_mask", None),
            getattr(self, "best_pattern", None),
            getattr(self, "best_delta_z", None),
        )

    # -------------------------------------------------------------------------
    # ↓↓↓ 以下为提炼出的辅助子函数 ↓↓↓
    # -------------------------------------------------------------------------

    def _prepare_trigger_data(self, tri_dataset, attack_size, batch_size):
        """Prepare data loader and visualization sample"""
        if tri_dataset is None:
            x_benign = torch.rand(
                attack_size, self.img_channels, self.img_rows, self.img_cols, device=self.device
            )
            y_target = torch.full((attack_size,), self.target_label, device=self.device)
            dataset = torch.utils.data.TensorDataset(x_benign, y_target)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return loader, None

        target_original_label = getattr(self.params, "origin_target", 1)
        vis_image = None
        for x, y in tri_dataset:
            if y == target_original_label:
                vis_image = x
                break

        # Build modified_dataset: same as your original code (all samples, label replaced by self.target_label)
        modified_dataset = [
            (x.to(self.device), torch.tensor(self.target_label, device=self.device))
            for x, _ in tri_dataset
        ]

        # === NEW: build ori_dataset by filtering samples whose original label == self.target_label ===
        ori_selected = []
        for x, y in tri_dataset:
            # handle torch tensors / scalars compatibly (original code assumed scalar y)
            y_val = y.item() if torch.is_tensor(y) else int(y)
            if y_val == self.target_label:
                ori_selected.append((x.to(self.device), torch.tensor(self.target_label, device=self.device)))

        ori_dataset = ori_selected  # list of tuples like modified_dataset

        # Fallback if modified_dataset empty (preserve original behavior)
        if len(modified_dataset) == 0:
            print("Warning: No suitable data found in tri_dataset, using random data")
            x_benign = torch.rand(
                attack_size, self.img_channels, self.img_rows, self.img_cols, device=self.device
            )
            y_target = torch.full((attack_size,), self.target_label, device=self.device)
            modified_dataset = list(zip(x_benign, y_target))

        # Create loader for modified dataset
        loader = torch.utils.data.DataLoader(modified_dataset, batch_size=batch_size, shuffle=True)

        # Create ori_loader if we found any ori samples; otherwise set to None
        if len(ori_dataset) > 0:
            ori_loader = torch.utils.data.DataLoader(ori_dataset, batch_size=batch_size, shuffle=True)
        else:
            ori_loader = None

        return loader, ori_loader, vis_image

    def _init_pixel_space(self):
        """Initialize mask and pattern for pixel-space trigger generation"""
        init_mask = np.random.uniform(0, 0.1, self.mask_size)
        init_pattern = np.random.rand(*self.pattern_size)
        init_mask = np.clip(init_mask, 0, 1)
        init_pattern = np.clip(init_pattern, 0, 1)

        self.mask_tensor = torch.tensor(
            np.arctanh((init_mask - 0.5) * (2 - self.epsilon)),
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        self.pattern_tensor = torch.tensor(
            np.arctanh((init_pattern - 0.5) * (2 - self.epsilon)),
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        self.optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor], lr=self.lr)

    def _init_feature_space(self, loader):
        """Estimate feature dimension and initialize delta_z"""
        sample_batches = [x for i, (x, _) in enumerate(loader) if i < 3]
        if not sample_batches:
            raise ValueError("No data available for feature dimension estimation")

        with torch.no_grad():
            all_features = []
            for batch in sample_batches:
                features, _ = self.model(batch.to(self.device))
                all_features.append(features)
            avg_features = torch.cat(all_features, dim=0).mean(dim=0)
            feat_dim = avg_features.shape[0]
        print(f"[Feature] Estimated feature dimension: {feat_dim}")

        init_std = 0.01
        self.delta_z = torch.normal(0, init_std, size=(feat_dim,), device=self.device, requires_grad=True)

        self.optimizer = torch.optim.Adam([self.delta_z], lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.8, patience=10
        )

    def _optimize_pixel_step(self, x_batch, y_batch, criterion, cost):
        """Perform one optimization step for pixel-space mode"""
        mask = torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5
        mask = mask.repeat(self.img_channels, 1, 1)
        pattern = torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5
        x_adv = (1 - mask) * x_batch + mask * pattern
        _, output = self.model(x_adv)

        loss_ce = criterion(output, y_batch)
        loss_reg = (mask ** 2).sum() / self.img_channels
        loss = loss_ce + cost * loss_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = (output.argmax(dim=1) == y_batch).float().mean().item()
        self.current_mask, self.current_pattern = mask, pattern
        return loss_ce.item(), loss_reg.item(), acc

    def _optimize_feature_step(self, x_batch, y_batch, criterion, cost, target_center=None):
        """Perform one optimization step for feature-space mode"""
        try:
            features, _ = self.model(x_batch)
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            return 0, 0, 0

        if features.shape[1] != self.delta_z.shape[0]:
            print(f"Warning: Feature dimension mismatch. Expected {self.delta_z.shape[0]}, got {features.shape[1]}")
            return 0, 0, 0

        z_adv = features + self.delta_z.unsqueeze(0).expand_as(features)
        try:
            _, output = self.model(features=z_adv)
        except Exception as e:
            print(f"Error in model feature forward: {e}")
            return 0, 0, 0

        # # 未加入反向约束损失的损失函数
        # loss_ce = criterion(output, y_batch)
        # loss_reg = (self.delta_z ** 2).sum()
        # loss = loss_ce + cost * loss_reg
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_([self.delta_z], max_norm=1.0)
        # self.optimizer.step()
        #
        # acc = (output.argmax(dim=1) == y_batch).float().mean().item()
        # return loss_ce.item(), loss_reg.item(), acc

            # --- 主任务与正则 ---

        loss_ce = criterion(output, y_batch)
        loss_reg = (self.delta_z ** 2).sum()

        # --- 反相似约束项 ---
        anti_sim_loss = 0.0
        if target_center is not None:
            delta_norm = F.normalize(self.delta_z, dim=0)
            target_norm = F.normalize(target_center, dim=0)
            cos_sim = torch.dot(delta_norm, target_norm)  # [-1, 1]
            anti_sim_loss = F.relu(cos_sim - 0.0)  # 惩罚大于阈值的相似度
        else:
            cos_sim = 0.0

        # --- 总损失 ---
        lambda_anti = 1.0  # 你可以调成 0.05~0.2
        loss = loss_ce + cost * loss_reg + lambda_anti * anti_sim_loss
        # loss = loss_ce + cost * loss_reg

        # --- 更新参数 ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.delta_z], max_norm=1.0)
        self.optimizer.step()

        acc = (output.argmax(dim=1) == y_batch).float().mean().item()

        # 可选打印监控
        # if target_center is not None:
        #     print(f"cos_sim(δ,target)={cos_sim.item():.4f}, anti_loss={anti_sim_loss.item():.4f}")

        return loss.item(), loss_ce.item() , loss_reg.item(), anti_sim_loss.item(), cos_sim.item(), acc

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
