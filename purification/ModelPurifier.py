import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


class ModelPurifier:
    """
    强化版模型净化器类，用于检测和消除模型中的后门攻击
    """

    def __init__(self, device, max_history_length=10):
        """
        初始化净化器

        Args:
            device: 计算设备
            max_history_length: 触发器历史记录的最大长度
        """
        self.device = device
        self.max_history_length = max_history_length
        self.trigger_history = []
        self.purify_history = []

    def _find_last_linear_layer(self, model):
        """
        找到模型的最后一个线性层
        """
        last_linear = None
        last_name = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
                last_name = name
        return last_linear, last_name

    def get_trigger_gradient_vector(self, model, delta_z, target_label):
        """
        计算特征触发器对应的梯度向量，用于净化

        Args:
            model: 待分析的模型
            delta_z: 特征触发器
            target_label: 目标标签

        Returns:
            tuple: (触发器梯度向量, 全连接层参数名列表)
        """
        model.train()

        # 确保delta_z在正确的设备上
        delta_z = delta_z.to(self.device)

        # 创建批次输入提高稳定性
        batch_size = 8
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(self.device)

        try:
            # 前向传播获取特征
            features, _ = model(dummy_input)

            # 找到最后一个线性层
            last_linear, last_name = self._find_last_linear_layer(model)
            if last_linear is None:
                print("[错误] 未找到线性分类层")
                return None, []

            # 添加触发器到特征
            if delta_z.dim() == 1:
                triggered_features = features + delta_z.unsqueeze(0).expand(batch_size, -1)
            else:
                triggered_features = features + delta_z.expand(batch_size, -1)

            # 直接使用线性层计算输出
            outputs = F.linear(triggered_features, last_linear.weight, last_linear.bias)

            # 创建目标标签
            targets = torch.full((batch_size,), target_label, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(outputs, targets)

            # 计算梯度
            model.zero_grad()
            loss.backward()

            # 收集全连接层梯度
            fc_gradients = []
            fc_names = []

            for name, param in model.named_parameters():
                if ("fc" in name or "classifier" in name or "linear" in name or "head" in name) and param.grad is not None:
                    fc_gradients.append(param.grad.detach().flatten())
                    fc_names.append(name)

            if fc_gradients:
                g_trigger = torch.cat(fc_gradients)
                print(f"[触发器梯度] 成功获得梯度向量，维度: {g_trigger.shape}")
                return g_trigger, fc_names
            else:
                print("[警告] 未获得有效的全连接层梯度")
                return None, []

        except Exception as e:
            print(f"[错误] 计算触发器梯度时出错: {e}")
            return None, []

    def compute_fc_similarity_with_trigger(self, clients_update, g_trigger, fc_names):
        """
           计算客户端更新与触发器梯度的相似度。
           【修正】恶意客户端的更新方向与梯度方向相反，相似度应接近-1。
           """
        similarities = {}
        # 关键修正：我们将寻找与 g_trigger 方向相反的更新。
        # g_trigger 指向损失增大的方向，而恶意客户端的更新指向损失减小的方向。
        g_trigger_norm = g_trigger / (torch.norm(g_trigger) + 1e-12)

        for client_id, updates in clients_update.items():
            client_fc_updates = []

            for name in fc_names:
                if name in updates:
                    client_fc_updates.append(updates[name].flatten().to(self.device))

            if client_fc_updates:
                flat_update = torch.cat(client_fc_updates)
                # 如果更新向量很小，则认为其无明确方向性，相似度为0
                if torch.norm(flat_update) < 1e-9:
                    similarities[client_id] = 0.0
                    continue

                flat_update_norm = flat_update / (torch.norm(flat_update) + 1e-12)
                # 计算出的相似度在[-1, 1]之间。-1表示最可疑。
                similarity = torch.dot(g_trigger_norm, flat_update_norm).item()
                similarities[client_id] = similarity
            else:
                similarities[client_id] = 0.0

        return similarities

    def _update_trigger_history(self, g_trigger):
        """更新触发器历史记录"""
        self.trigger_history.append(g_trigger.clone())
        if len(self.trigger_history) > self.max_history_length:
            self.trigger_history.pop(0)

    def _compute_accumulated_trigger(self):
        """计算累积触发器方向"""
        if not self.trigger_history:
            return None

        # 使用指数加权平均，最近的触发器权重更大
        weights = torch.exp(torch.linspace(0, 1, len(self.trigger_history))).to(self.device)
        weights = weights / weights.sum()

        weighted_triggers = []
        for i, trigger in enumerate(self.trigger_history):
            weighted_triggers.append(weights[i] * trigger)

        accumulated_trigger = torch.sum(torch.stack(weighted_triggers), dim=0)
        return accumulated_trigger

    def _compute_attack_intensity(self, similarities, threshold=0.0):
        """
        计算攻击强度。
        【修正】相似度小于阈值（如-0.1）的客户端被认为是可疑的。
        """
        # 相似度越接近-1，嫌疑越大
        high_sim_clients = [cid for cid, sim in similarities.items() if sim < threshold]
        attack_intensity = len(high_sim_clients) / max(len(similarities), 1)

        print(f"[攻击强度] {len(high_sim_clients)}/{len(similarities)} 客户端可疑 (相似度 < {threshold})，强度: {attack_intensity:.2f}")
        return attack_intensity

    def _compute_purify_ratio(self, attack_intensity):
        """动态计算净化比例"""
        # 基础净化强度：根据攻击强度动态调整
        base_ratio = min(0.5, 0.1 + attack_intensity * 0.3)

        # 基于历史的自适应调整
        if len(self.purify_history) >= 2:
            recent_main_trend = self.purify_history[-1]['main_acc'] - self.purify_history[-2]['main_acc']
            recent_bd_trend = self.purify_history[-1]['backdoor_acc'] - self.purify_history[-2]['backdoor_acc']

            # 如果主任务下降严重，减弱净化
            if recent_main_trend < -0.05:
                base_ratio *= 0.6
                print(f"[净化减弱] 主任务下降，调整净化强度至 {base_ratio:.3f}")
            # 如果后门攻击率上升，增强净化
            elif recent_bd_trend > 0.1:
                base_ratio = min(0.8, base_ratio * 1.5)
                print(f"[净化增强] 后门攻击率上升，调整净化强度至 {base_ratio:.3f}")

        return base_ratio

    def _evaluate_baseline_performance(self, model, test_dataset):
        """评估净化前的基准性能"""
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
            batch_x, batch_y = next(iter(test_loader))
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            _, baseline_outputs = model(batch_x)
            correct = (baseline_outputs.argmax(dim=1) == batch_y).sum().item()
            baseline_acc = correct / batch_y.size(0)

        return baseline_acc, (batch_x, batch_y)

    def _apply_enhanced_projection(self, fc_params, accumulated_trigger, g_trigger, base_ratio):
        """应用增强的正交投影净化"""
        # 展平所有FC参数
        flat_fc = torch.cat([p.data.flatten() for p in fc_params])

        # 归一化触发器方向
        acc_unit = accumulated_trigger / (torch.norm(accumulated_trigger) + 1e-12)
        cur_unit = g_trigger / (torch.norm(g_trigger) + 1e-12)

        # 计算投影系数
        alpha_acc = torch.dot(flat_fc, acc_unit)
        alpha_cur = torch.dot(flat_fc, cur_unit)

        print(f"[投影系数] 累积: {alpha_acc:.4f}, 当前: {alpha_cur:.4f}")

        # 应用正交投影 - 去除后门方向的分量
        backdoor_component_acc = alpha_acc * acc_unit * base_ratio * 0.8
        backdoor_component_cur = alpha_cur * cur_unit * base_ratio * 0.2

        # 净化后的参数
        flat_fc_clean = flat_fc - backdoor_component_acc - backdoor_component_cur

        # 写回参数
        start = 0
        with torch.no_grad():
            for p in fc_params:
                numel = p.numel()
                p.data.copy_(flat_fc_clean[start:start+numel].view_as(p))
                start += numel

        return alpha_acc, alpha_cur

    def _evaluate_purify_performance(self, model, batch_data, params, delta_z=None):
        """评估净化后的性能"""
        batch_x, batch_y = batch_data

        with torch.no_grad():
            # 评估主任务性能
            _, after_outputs = model(batch_x)
            correct = (after_outputs.argmax(dim=1) == batch_y).sum().item()
            main_acc = correct / batch_y.size(0)

            # 评估后门性能
            backdoor_acc = 0.0
            if hasattr(params, 'origin_target') and delta_z is not None:
                origin_mask = (batch_y == params.origin_target)
                if torch.sum(origin_mask) > 0:
                    origin_samples = batch_x[origin_mask]

                    # 获取特征并添加触发器
                    features, _ = model(origin_samples)

                    # 找到最后的线性层
                    last_linear, _ = self._find_last_linear_layer(model)
                    if last_linear is not None:
                        z_adv = features + delta_z.unsqueeze(0).expand_as(features)
                        bd_outputs = F.linear(z_adv, last_linear.weight, last_linear.bias)

                        target_labels = torch.full((origin_samples.size(0),), params.aim_target,
                                                 dtype=torch.long, device=self.device)
                        correct_bd = (bd_outputs.argmax(dim=1) == target_labels).sum().item()
                        backdoor_acc = correct_bd / origin_samples.size(0)

        return main_acc, backdoor_acc

    def _should_rollback(self, baseline_acc, main_acc, threshold=0.1):
        """判断是否需要回滚"""
        performance_drop = baseline_acc - main_acc
        # 更宽松的回滚条件，允许一定的性能下降以换取后门净化
        should_rollback = (performance_drop > threshold) or (main_acc < 0.2)
        return should_rollback, performance_drop

    def _update_purify_history(self, epoch, alpha_acc, alpha_cur, base_ratio, main_acc, backdoor_acc, attack_intensity):
        """更新净化历史记录"""
        self.purify_history.append({
            'epoch': epoch,
            'alpha_acc': alpha_acc.item() if isinstance(alpha_acc, torch.Tensor) else alpha_acc,
            'alpha_cur': alpha_cur.item() if isinstance(alpha_cur, torch.Tensor) else alpha_cur,
            'purify_ratio': base_ratio,
            'main_acc': main_acc,
            'backdoor_acc': backdoor_acc,
            'attack_intensity': attack_intensity
        })

    def _monitor_trends(self):
        """监控净化趋势"""
        if len(self.purify_history) >= 3:
            recent_bd_accs = [h['backdoor_acc'] for h in self.purify_history[-3:]]
            avg_bd_acc = np.mean(recent_bd_accs)
            if avg_bd_acc > 0.6:
                print(f"[警告] 后门攻击率持续较高: {avg_bd_acc:.3f}, 建议检查触发器质量")

    def purify_model(self, model, delta_z, target_label, clients_update, test_dataset, params, epoch):
        """
        主净化函数 - 对模型进行净化，消除后门攻击
        """
        print(f"\n[净化开始] Epoch {epoch}")

        # 1. 获取触发器梯度向量
        g_trigger, fc_names = self.get_trigger_gradient_vector(model, delta_z, target_label)
        if g_trigger is None:
            print("[净化失败] 无法获得触发器梯度")
            return None

        # 2. 计算客户端相似度
        similarities = self.compute_fc_similarity_with_trigger(clients_update, g_trigger, fc_names)
        # 【修正】按相似度升序排列，值越小越可疑
        sorted_clients = sorted(similarities.items(), key=lambda x: x[1], reverse=False)

        print("客户端可疑度排名 (值越小越可疑):")
        for cid, sim in sorted_clients[:5]:  # 只显示前5个
            print(f"  Client {cid}: {sim:.4f}")

        # 3. 更新触发器历史
        self._update_trigger_history(g_trigger)

        # 4. 计算累积触发器方向
        accumulated_trigger = self._compute_accumulated_trigger()
        if accumulated_trigger is None:
            print("[净化跳过] 触发器历史为空")
            return None

        # 5. 计算攻击强度和净化比例
        attack_intensity = self._compute_attack_intensity(similarities)
        base_ratio = self._compute_purify_ratio(attack_intensity)

        # 6. 获取全连接层参数
        fc_params = []
        for name, param in model.named_parameters():
            if ("fc" in name or "classifier" in name or "linear" in name or "head" in name):
                fc_params.append(param)

        if not fc_params:
            print("[净化失败] 未找到全连接层参数")
            return None

        # 7. 备份原参数
        original_params = [p.data.clone() for p in fc_params]

        # 8. 评估基准性能
        baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
        print(f"[基准性能] 准确率: {baseline_acc:.3f}")

        # 9. 应用净化
        alpha_acc, alpha_cur = self._apply_enhanced_projection(fc_params, accumulated_trigger, g_trigger, base_ratio)

        # 10. 评估净化效果
        main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)

        # 11. 判断是否回滚
        should_rollback, performance_drop = self._should_rollback(baseline_acc, main_acc)

        if should_rollback:
            print(f"[净化回滚] 性能下降过大: {performance_drop:.3f}, 恢复原参数")
            with torch.no_grad():
                for p, orig_p in zip(fc_params, original_params):
                    p.data.copy_(orig_p)
            main_acc = baseline_acc
        else:
            print(f"[净化成功] 强度: {base_ratio:.3f}, 主任务: {main_acc:.3f}, 后门: {backdoor_acc:.3f}")

            # 记录净化历史
            self._update_purify_history(epoch, alpha_acc, alpha_cur, base_ratio, main_acc, backdoor_acc, attack_intensity)
            self._monitor_trends()

        return {
            'g_trigger': g_trigger,
            'fc_names': fc_names,
            'similarities': similarities,
            'attack_intensity': attack_intensity,
            'purify_ratio': base_ratio,
            'main_acc': main_acc,
            'backdoor_acc': backdoor_acc,
            'baseline_acc': baseline_acc,
            'performance_drop': performance_drop,
            'rollback': should_rollback
        }
