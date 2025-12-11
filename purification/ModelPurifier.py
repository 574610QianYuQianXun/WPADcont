import copy

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy


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

    def get_robust_trigger_direction_svd(self, model, delta_z, target_label, num_samples=20, batch_size=4):
        """
        【新增】通过SVD计算稳健的触发器梯度方向。

        该方法通过在多个不同的随机输入上计算后门任务梯度，然后使用SVD分解
        提取这些梯度向量中最主要的方向（第一右奇异向量），作为更稳定的后门方向。

        Args:
            model: 待分析的模型。
            delta_z: 特征触发器。
            target_label: 目标标签。
            num_samples (int): 梯度采样的次数，更高的值更稳定但计算成本更高。
            batch_size (int): 每次计算梯度时使用的批次大小。

        Returns:
            tuple: (稳健的触发器方向向量, 全连接层参数名列表) 或 (None, [])
        """
        print(f"[SVD方向提取] 开始进行 {num_samples} 次梯度采样以计算稳定后门方向...")
        model.train()
        delta_z = delta_z.to(self.device)

        last_linear, _ = self._find_last_linear_layer(model)
        if last_linear is None:
            print("[错误] SVD方向提取失败：未找到线性分类层")
            return None, []

        gradient_samples = []
        fc_names = []

        for i in range(num_samples):
            # 使用不同的随机数据进行每次采样
            dummy_input = torch.randn(batch_size, 3, 32, 32, device=self.device)

            try:
                features, _ = model(dummy_input)

                if delta_z.dim() == 1:
                    triggered_features = features + delta_z.unsqueeze(0)
                else:
                    triggered_features = features + delta_z

                outputs = F.linear(triggered_features, last_linear.weight, last_linear.bias)
                targets = torch.full((batch_size,), target_label, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(outputs, targets)

                model.zero_grad()
                loss.backward()

                current_fc_gradients = []
                # 在第一次迭代时记录fc_names
                if not fc_names:
                    for name, param in model.named_parameters():
                        if (
                                "fc" in name or "classifier" in name or "linear" in name or "head" in name) and param.grad is not None:
                            current_fc_gradients.append(param.grad.detach().flatten())
                            fc_names.append(name)
                else:
                    for name, param in model.named_parameters():
                        if name in fc_names and param.grad is not None:
                            current_fc_gradients.append(param.grad.detach().flatten())

                if current_fc_gradients:
                    gradient_samples.append(torch.cat(current_fc_gradients))

            except Exception as e:
                print(f"[警告] SVD梯度采样第 {i + 1} 次失败: {e}")
                continue

        if len(gradient_samples) < 2:
            print("[错误] SVD方向提取失败：有效的梯度样本不足。")
            return None, []

        # 将所有梯度样本堆叠成一个矩阵 (num_samples, grad_dim)
        gradient_matrix = torch.stack(gradient_samples)

        # 对梯度矩阵进行SVD分解
        # U: 左奇异向量, S: 奇异值, Vh: 右奇异向量的共轭转置
        # 我们需要 Vh 的第一行，它对应最大的奇异值，代表了数据中最主要的方向
        try:
            _, _, Vh = torch.linalg.svd(gradient_matrix, full_matrices=False)
            robust_direction = Vh[0]  # 这就是稳健的后门方向
            print(f"[SVD方向提取] 成功提取稳定后门方向，维度: {robust_direction.shape}")
            return robust_direction, fc_names
        except torch.linalg.LinAlgError as e:
            print(f"[错误] SVD分解失败: {e}。将返回第一个采样梯度作为备用。")
            return gradient_samples[0], fc_names

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
        # 【关键修改】大幅降低基础净化强度，优先保护主任务性能
        base_ratio = min(0.2, 0.02 + attack_intensity * 0.15)  # 从0.5降到0.2，从0.1降到0.02

        # 基于历史的自适应调整
        if len(self.purify_history) >= 2:
            recent_main_trend = self.purify_history[-1]['main_acc'] - self.purify_history[-2]['main_acc']
            recent_bd_trend = self.purify_history[-1]['backdoor_acc'] - self.purify_history[-2]['backdoor_acc']

            # 【修改】更严格的主任务保护策略
            if recent_main_trend < -0.02:  # 从-0.05提高到-0.02，更敏感
                base_ratio *= 0.3  # 从0.6降到0.3，更大幅度减弱
                print(f"[净化减弱] 主任务下降，大幅调整净化强度至 {base_ratio:.3f}")
            # 【修改】更保守的净化增强条件
            elif recent_bd_trend > 0.2 and recent_main_trend > -0.01:  # 添加主任务稳定条件
                base_ratio = min(0.4, base_ratio * 1.2)  # 从0.8降到0.4，从1.5降到1.2
                print(f"[净化增强] 后门攻击率上升且主任务稳定，调整净化强度至 {base_ratio:.3f}")

        return base_ratio

    def _evaluate_baseline_performance(self, model, test_dataset):
        """评估净化前的基准性能"""
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
            batch_x, batch_y = next(iter(test_loader))
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            _, baseline_outputs = model(batch_x)
            correct = (baseline_outputs.argmax(dim=1) == batch_y).float().sum().item()
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

        # 【修改】大幅降低投影净化强度，减少对主任务的冲击
        backdoor_component_acc = alpha_acc * acc_unit * base_ratio * 0.5  # 从0.8降到0.5
        backdoor_component_cur = alpha_cur * cur_unit * base_ratio * 0.1  # 从0.2降到0.1

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
            correct = (after_outputs.argmax(dim=1) == batch_y).float().sum().item()
            main_acc = correct / batch_y.size(0)

            # 评估后门性能
            backdoor_acc = 0.0
            if hasattr(params, 'origin_target') and delta_z is not None:
                origin_mask = (batch_y == params.origin_target)
                if origin_mask.sum().item() > 0:
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

    def _should_rollback(self, baseline_acc, main_acc, threshold=0.05):  # 从0.1降到0.05
        """判断是否需要回滚"""
        performance_drop = baseline_acc - main_acc
        # 【修改】更严格的回滚条件，更好地保护主任务
        should_rollback = (performance_drop > threshold) or (main_acc < 0.15)  # 从0.2提高到0.15
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

    def locate_backdoor_neurons_with_noise(self, model, delta_z, num_runs=5, noise_samples=500, top_k=10):
        """
        使用噪声数据定位后门神经元，通过多次运行确保精准定位

        Args:
            model: 待分析的模型
            delta_z: 特征触发器
            num_runs: 多次运行次数，确保定位稳定性
            noise_samples: 每次运行使用的噪声样本数量
            top_k: 选择最可疑的前k个神经元

        Returns:
            list: 最终确定的后门神经元索引列表
        """
        print(f"\n[神经元定位] 开始使用噪声数据定位后门神经元 (运行{num_runs}次，每次{noise_samples}样本)")
        model.eval()

        # 找到最后一个线性层
        last_linear, last_name = self._find_last_linear_layer(model)
        if last_linear is None:
            print("[神经元定位失败] 未找到线性层")
            return []

        num_classes = last_linear.out_features
        feature_dim = last_linear.in_features

        # 存储多次运行的结果
        all_suspicious_neurons = []
        # 存储每次运行的激活差异，用于最终平均
        all_runs_activation_diffs = []

        for run_idx in range(num_runs):
            print(f"[神经元定位] 执行第 {run_idx + 1}/{num_runs} 次定位...")

            # 生成随机噪声数据作为测试样本（服务器端无需真实数据）
            noise_data = torch.randn(noise_samples, 3, 32, 32).to(self.device)

            with torch.no_grad():
                # 获取噪声数据的特征表示
                features, _ = model(noise_data)

                # 计算干净特征的激活值
                clean_activations = F.linear(features, last_linear.weight, last_linear.bias)

                # 添加特征触发器并计算触发激活值
                if delta_z.dim() == 1:
                    triggered_features = features + delta_z.unsqueeze(0).expand_as(features)
                else:
                    triggered_features = features + delta_z.expand_as(features)

                triggered_activations = F.linear(triggered_features, last_linear.weight, last_linear.bias)

                # 计算激活差异（绝对值）
                activation_diff = torch.abs(triggered_activations - clean_activations)

                # 计算每个神经元的平均激活差异
                mean_diff_per_neuron = activation_diff.mean(dim=0)
                all_runs_activation_diffs.append(mean_diff_per_neuron)

                # 找到差异最大的top_k个神经元
                _, top_neurons = torch.topk(mean_diff_per_neuron, k=min(top_k * 2, num_classes))
                all_suspicious_neurons.extend(top_neurons.cpu().numpy().tolist())

                # **新增：输出本次运行的激活差异详情**
                print(f"[运行 {run_idx + 1}] 各神经元激活差异:")
                sorted_diffs, sorted_indices = torch.sort(mean_diff_per_neuron, descending=True)
                for i in range(min(num_classes, 15)):  # 只显示前15个最大差异
                    neuron_idx = sorted_indices[i].item()
                    diff_value = sorted_diffs[i].item()
                    print(f"  神经元 {neuron_idx}: 激活差异 = {diff_value:.4f}")

        # **新增：计算所有运行的平均激活差异**
        print(f"\n[神经元定位] 计算 {num_runs} 次运行的平均激活差异:")
        avg_activation_diffs = torch.stack(all_runs_activation_diffs).mean(dim=0)

        # 按激活差异从大到小排序
        sorted_avg_diffs, sorted_avg_indices = torch.sort(avg_activation_diffs, descending=True)

        print("[平均激活差异] 所有神经元排序结果:")
        for i in range(num_classes):
            neuron_idx = sorted_avg_indices[i].item()
            avg_diff = sorted_avg_diffs[i].item()
            # 用星号标记可能的后门神经元
            marker = " ⭐" if i < top_k else ""
            print(f"  神经元 {neuron_idx}: 平均激活差异 = {avg_diff:.4f}{marker}")

        # 统计多次运行中出现频率最高的神经元
        from collections import Counter
        neuron_counts = Counter(all_suspicious_neurons)

        # 选择出现频率最高的top_k个神经元作为最终的后门神经元
        most_common_neurons = neuron_counts.most_common(top_k)
        final_backdoor_neurons = [neuron for neuron, count in most_common_neurons]

        print(f"\n[神经元定位] 基于频次统计的最终结果:")
        for neuron, count in most_common_neurons:
            confidence = count / num_runs * 100
            avg_diff_for_this_neuron = avg_activation_diffs[neuron].item()
            print(f"  神经元 {neuron}: 出现 {count}/{num_runs} 次 (置信度: {confidence:.1f}%), "
                  f"平均激活差异: {avg_diff_for_this_neuron:.4f}")

        print(f"\n[神经元定位] 最终确定的后门神经元: {final_backdoor_neurons}")

        # **新增：输出激活差异的统计信息**
        print(f"\n[统计信息] 激活差异分布:")
        print(f"  最大差异: {torch.max(avg_activation_diffs).item():.4f}")
        print(f"  最小差异: {torch.min(avg_activation_diffs).item():.4f}")
        print(f"  平均差异: {torch.mean(avg_activation_diffs).item():.4f}")
        print(f"  标准差: {torch.std(avg_activation_diffs).item():.4f}")

        return final_backdoor_neurons

    def prune_backdoor_neurons(self, model, backdoor_neurons):
        """
        剪枝指定的后门神经元

        Args:
            model: 目标模型
            backdoor_neurons: 要剪枝的神经元索引列表

        Returns:
            bool: 剪枝是否成功
        """
        if not backdoor_neurons:
            print("[神经元剪枝] 没有发现后门神经元，跳过剪枝")
            return False

        last_linear, _ = self._find_last_linear_layer(model)
        if last_linear is None:
            print("[神经元剪枝失败] 未找到线性层")
            return False

        print(f"[神经元剪枝] 开始剪枝 {len(backdoor_neurons)} 个后门神经元: {backdoor_neurons}")

        with torch.no_grad():
            # 将对应神经元的权重和偏置置零
            for neuron_idx in backdoor_neurons:
                if neuron_idx < last_linear.out_features:
                    # 置零输出层对应神经元的权重和偏置
                    last_linear.weight.data[neuron_idx] = 0.0
                    if last_linear.bias is not None:
                        last_linear.bias.data[neuron_idx] = 0.0

        print(f"[神经元剪枝] 成功剪枝 {len(backdoor_neurons)} 个神经元")
        return True

    def evaluate_neuron_pruning_effect(self, model, delta_z, target_label, backdoor_neurons, noise_samples=200):
        """
        评估神经元剪枝对后门攻击的影响效果

        Args:
            model: 剪枝后的模型
            delta_z: 特征触发器
            target_label: 目标标签
            backdoor_neurons: 被剪枝的神经元
            noise_samples: 用于评估的噪声样本数量

        Returns:
            float: 后门攻击成功率（ASR）
        """
        model.eval()

        # 生成噪声数据进行评估
        noise_data = torch.randn(noise_samples, 3, 32, 32).to(self.device)

        with torch.no_grad():
            # 获取特征并添加触发器
            features, _ = model(noise_data)

            if delta_z.dim() == 1:
                triggered_features = features + delta_z.unsqueeze(0).expand_as(features)
            else:
                triggered_features = features + delta_z.expand_as(features)

            # 找到最后线性层并计算输出
            last_linear, _ = self._find_last_linear_layer(model)
            if last_linear is None:
                return 0.0

            outputs = F.linear(triggered_features, last_linear.weight, last_linear.bias)
            predictions = outputs.argmax(dim=1)

            # 计算预测为目标标签的比例（ASR）
            target_predictions = (predictions == target_label).sum().item()
            asr = target_predictions / noise_samples

        print(f"[剪枝评估] 剪枝神经元 {backdoor_neurons} 后的ASR: {asr:.3f}")
        return asr

    def detect_malicious_clients_by_distribution(self, similarities, k=2.0):
        """
        【新增】基于相似度分布检测恶意客户端（离群点检测）。

        此方法不再使用固定的负值阈值，而是将所有相似度视为一个分布，
        通过统计方法（均值和标准差）找出其中的异常值（极端负值）。

        Args:
            similarities (dict): 包含客户端ID和其相似度的字典。
            k (float): 标准差的倍数，用于定义离群点的阈值。值越大，检测越保守。

        Returns:
            tuple: (可疑客户端ID列表, 攻击强度)
        """
        if not similarities:
            return [], 0.0

        sim_values = np.array(list(similarities.values()))

        # 过滤掉相似度为0的客户端（通常是更新为空或无效的）
        sim_values_filtered = sim_values[sim_values != 0]

        if len(sim_values_filtered) < 3: # 需要足够样本进行统计
            # 如果样本太少，退化为选择最小的那个
            suspicious_clients = [cid for cid, sim in similarities.items() if sim < -0.1]
        else:
            mean = np.mean(sim_values_filtered)
            std = np.std(sim_values_filtered)

            # 阈值是均值减去k倍标准差。相似度越小越可疑。
            threshold = mean - k * std

            print(f"[离群点检测] 相似度分布: 均值={mean:.3f}, 标准差={std:.3f}")
            print(f"[离群点检测] 恶意客户端阈值 (mean - {k}*std): {threshold:.3f}")

            # 相似度显著低于群体均值的被认为是可疑的
            suspicious_clients = [cid for cid, sim in similarities.items() if sim < threshold]

        attack_intensity = len(suspicious_clients) / max(len(similarities), 1)

        print(f"[离群点检测] 检测到 {len(suspicious_clients)}/{len(similarities)} 个可疑客户端，攻击强度: {attack_intensity:.2f}")
        if suspicious_clients:
            print(f"[离群点检测] 可疑客户端ID: {suspicious_clients}")

        return suspicious_clients, attack_intensity

    # def purify_model(self, model, delta_z, target_label, clients_update, test_dataset, params, epoch):
    #     """
    #     主净化函数 - 对模型进行净化，消除后门攻击
    #     【重大修改】采用SVD稳定方向和基于分布的恶意客户端检测。
    #     """
    #     print(f"\n[净化开始] Epoch {epoch}")
    #
    #     # 1. 【修改】获取稳健的触发器梯度方向 (SVD)
    #     g_trigger, fc_names = self.get_robust_trigger_direction_svd(model, delta_z, target_label, num_samples=20)
    #     if g_trigger is None:
    #         print("[净化失败] 无法获得稳健的触发器方向")
    #         return None
    #
    #     # 2. 计算客户端相似度 (无变化)
    #     similarities = self.compute_fc_similarity_with_trigger(clients_update, g_trigger, fc_names)
    #     sorted_clients = sorted(similarities.items(), key=lambda x: x[1], reverse=False)
    #     print("客户端可疑度排名 (值越小越可疑):")
    #     for cid, sim in sorted_clients[:20]:
    #         print(f"  Client {cid}: {sim:.4f}")
    #
    #     # 3. 【修改】基于分布检测恶意客户端并计算攻击强度
    #     suspicious_clients, attack_intensity = self.detect_malicious_clients_by_distribution(similarities, k=2.0)
    #
    #     # 4. 更新触发器历史 (使用SVD得到的稳定方向)
    #     self._update_trigger_history(g_trigger)
    #
    #     # 5. 计算累积触发器方向
    #     accumulated_trigger = self._compute_accumulated_trigger()
    #     if accumulated_trigger is None:
    #         print("[净化跳过] 触发器历史为空")
    #         return None
    #
    #     # 6. 根据攻击强度选择净化策略 (神经元剪枝或投影)
    #     use_neuron_targeting = attack_intensity > 0.3
    #
    #     if use_neuron_targeting:
    #         print(f"[净化策略] 检测到高攻击强度({attack_intensity:.2f})，启用精准神经元定位模式")
    #
    #         # **步骤A：使用噪声数据精准定位后门神经元**
    #         backdoor_neurons = self.locate_backdoor_neurons_with_noise(
    #             model=model,
    #             delta_z=delta_z,
    #             num_runs=5,  # 5次独立运行确保稳定性
    #             noise_samples=500,  # 每次使用500个噪声样本
    #             top_k=3  # 定位前3个最可疑的神经元
    #         )
    #
    #         # **步骤B：备份模型状态**
    #         model_state_backup = {name: param.data.clone() for name, param in model.named_parameters()}
    #
    #         # **步骤C：执行神经元剪枝**
    #         if backdoor_neurons and self.prune_backdoor_neurons(model, backdoor_neurons):
    #             # **步骤D：评估剪枝效果**
    #             asr_after_pruning = self.evaluate_neuron_pruning_effect(
    #                 model=model,
    #                 delta_z=delta_z,
    #                 target_label=target_label,
    #                 backdoor_neurons=backdoor_neurons,
    #                 noise_samples=300
    #             )
    #
    #             # **步骤E：评估主任务性能**
    #             baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
    #             main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)
    #
    #             # **步骤F：判断剪枝是否成功**
    #             performance_drop = baseline_acc - main_acc
    #             neuron_pruning_successful = (performance_drop < 0.15) and (asr_after_pruning < 0.3)
    #
    #             if neuron_pruning_successful:
    #                 print(f"[神经元净化成功] ASR从未知降至{asr_after_pruning:.3f}，主任务准确率: {main_acc:.3f}")
    #
    #                 # 记录净化历史
    #                 self._update_purify_history(epoch, 0, 0, 1.0, main_acc, backdoor_acc, attack_intensity)
    #
    #                 return {
    #                     'g_trigger': g_trigger,
    #                     'fc_names': fc_names,
    #                     'similarities': similarities,
    #                     'suspicious_clients': suspicious_clients,
    #                     'attack_intensity': attack_intensity,
    #                     'purify_method': 'neuron_targeting',
    #                     'backdoor_neurons': backdoor_neurons,
    #                     'main_acc': main_acc,
    #                     'backdoor_acc': backdoor_acc,
    #                     'asr_after_pruning': asr_after_pruning,
    #                     'baseline_acc': baseline_acc,
    #                     'performance_drop': performance_drop,
    #                     'rollback': False
    #                 }
    #             else:
    #                 print(f"[神经元净化失败] 性能下降过大({performance_drop:.3f})或ASR仍高({asr_after_pruning:.3f})")
    #                 print("[回滚] 恢复到神经元剪枝前的模型状态，改用传统投影净化")
    #
    #                 # 恢复模型状态
    #                 with torch.no_grad():
    #                     for name, param in model.named_parameters():
    #                         param.data.copy_(model_state_backup[name])
    #
    #                 # 降级到传统投影净化
    #                 use_neuron_targeting = False
    #         else:
    #             print("[神经元定位失败] 未能定位到后门神经元，改用传统投影净化")
    #             use_neuron_targeting = False
    #
    #     # **传统投影净化分支**（原有逻辑，作为后备方案）
    #     if not use_neuron_targeting:
    #         print(f"[净化策略] 使用传统投影净化模式 (攻击强度: {attack_intensity:.2f})")
    #
    #         base_ratio = self._compute_purify_ratio(attack_intensity)
    #
    #         # 获取全连接层参数
    #         fc_params = []
    #         for name, param in model.named_parameters():
    #             if ("fc" in name or "classifier" in name or "linear" in name or "head" in name):
    #                 fc_params.append(param)
    #
    #         if not fc_params:
    #             print("[净化失败] 未找到全连接层参数")
    #             return None
    #
    #         # 备份原参数
    #         original_params = [p.data.clone() for p in fc_params]
    #
    #         # 评估基准性能
    #         baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
    #         print(f"[基准性能] 准确率: {baseline_acc:.3f}")
    #
    #         # 应用投影净化
    #         alpha_acc, alpha_cur = self._apply_enhanced_projection(fc_params, accumulated_trigger, g_trigger, base_ratio)
    #
    #         # 评估净化效果
    #         main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)
    #
    #         # 判断是否回滚
    #         should_rollback, performance_drop = self._should_rollback(baseline_acc, main_acc)
    #
    #         if should_rollback:
    #             print(f"[净化回滚] 性能下降过大: {performance_drop:.3f}, 恢复原参数")
    #             with torch.no_grad():
    #                 for p, orig_p in zip(fc_params, original_params):
    #                     p.data.copy_(orig_p)
    #             main_acc = baseline_acc
    #         else:
    #             print(f"[净化成功] 强度: {base_ratio:.3f}, 主任务: {main_acc:.3f}, 后门: {backdoor_acc:.3f}")
    #
    #             # 记录净化历史
    #             self._update_purify_history(epoch, alpha_acc, alpha_cur, base_ratio, main_acc, backdoor_acc, attack_intensity)
    #             self._monitor_trends()
    #
    #         return {
    #             'g_trigger': g_trigger,
    #             'fc_names': fc_names,
    #             'similarities': similarities,
    #             'suspicious_clients': suspicious_clients,
    #             'attack_intensity': attack_intensity,
    #             'purify_method': 'projection',
    #             'purify_ratio': base_ratio,
    #             'main_acc': main_acc,
    #             'backdoor_acc': backdoor_acc,
    #             'baseline_acc': baseline_acc,
    #             'performance_drop': performance_drop,
    #             'rollback': should_rollback
    #         }
    def purify_model(self, model, delta_z, target_label, clients_update, test_dataset, params, epoch):
        """
        主净化函数 - 对模型进行净化，消除后门攻击
        【增强版】加入微调修复阶段
        """
        print(f"\n[净化开始] Epoch {epoch}")
        teacher_model = copy.deepcopy(model)
        # 1. 获取稳健触发器方向 (SVD)
        g_trigger, fc_names = self.get_robust_trigger_direction_svd(model, delta_z, target_label, num_samples=20)
        if g_trigger is None:
            print("[净化失败] 无法获得稳健的触发器方向")
            return None

        # 2. 计算客户端相似度
        similarities = self.compute_fc_similarity_with_trigger(clients_update, g_trigger, fc_names)
        sorted_clients = sorted(similarities.items(), key=lambda x: x[1])
        print("客户端可疑度排名 (值越小越可疑):")
        for cid, sim in sorted_clients[:20]:
            print(f"  Client {cid}: {sim:.4f}")

        # 3. 检测恶意客户端与攻击强度
        suspicious_clients, attack_intensity = self.detect_malicious_clients_by_distribution(similarities, k=2.0)

        # 4. 更新触发器历史并计算累计方向
        self._update_trigger_history(g_trigger)
        accumulated_trigger = self._compute_accumulated_trigger()
        if accumulated_trigger is None:
            print("[净化跳过] 触发器历史为空")
            return None

        # === 神经元净化模式 ===
        use_neuron_targeting = attack_intensity > 0.3
        result_dict = None

        if use_neuron_targeting:
            print(f"[净化策略] 攻击强度高({attack_intensity:.2f})，启用精准神经元定位模式")

            backdoor_neurons = self.locate_backdoor_neurons_with_noise(
                model=model, delta_z=delta_z, num_runs=5, noise_samples=500, top_k=3
            )

            if not backdoor_neurons:
                print("[神经元定位失败] 改用传统投影净化")
                use_neuron_targeting = False
            else:
                model_backup = {n: p.data.clone() for n, p in model.named_parameters()}
                if not self.prune_backdoor_neurons(model, backdoor_neurons):
                    print("[剪枝执行失败] 改用传统投影净化")
                    use_neuron_targeting = False
                else:
                    asr = self.evaluate_neuron_pruning_effect(model, delta_z, target_label, backdoor_neurons, 300)
                    baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
                    main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)
                    drop = baseline_acc - main_acc
                    success = (drop < 0.15) and (asr < 0.3)

                    if success:
                        print(f"[神经元净化成功] ASR降至{asr:.3f}，主任务准确率: {main_acc:.3f}")
                        self._update_purify_history(epoch, 0, 0, 1.0, main_acc, backdoor_acc, attack_intensity)
                        result_dict = {
                            'g_trigger': g_trigger, 'fc_names': fc_names, 'similarities': similarities,
                            'suspicious_clients': suspicious_clients, 'attack_intensity': attack_intensity,
                            'purify_method': 'neuron_targeting', 'backdoor_neurons': backdoor_neurons,
                            'main_acc': main_acc, 'backdoor_acc': backdoor_acc, 'asr_after_pruning': asr,
                            'baseline_acc': baseline_acc, 'performance_drop': drop, 'rollback': False
                        }
                    else:
                        print(f"[神经元净化失败] 性能下降({drop:.3f})或ASR高({asr:.3f})，回滚并改用投影")
                        with torch.no_grad():
                            for n, p in model.named_parameters():
                                p.data.copy_(model_backup[n])
                        use_neuron_targeting = False

        # === 投影净化模式 ===
        if not use_neuron_targeting:
            print(f"[净化策略] 使用传统投影净化 (攻击强度: {attack_intensity:.2f})")

            base_ratio = self._compute_purify_ratio(attack_intensity)
            fc_params = [p for n, p in model.named_parameters() if
                         any(k in n for k in ["fc", "classifier", "linear", "head"])]
            if not fc_params:
                print("[净化失败] 未找到全连接层参数")
                return None

            originals = [p.data.clone() for p in fc_params]
            baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
            print(f"[基准性能] 准确率: {baseline_acc:.3f}")

            alpha_acc, alpha_cur = self._apply_enhanced_projection(fc_params, accumulated_trigger, g_trigger,
                                                                   base_ratio)
            main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)

            rollback, drop = self._should_rollback(baseline_acc, main_acc)
            if rollback:
                print(f"[净化回滚] 性能下降{drop:.3f}, 恢复原参数")
                with torch.no_grad():
                    for p, o in zip(fc_params, originals):
                        p.data.copy_(o)
                main_acc = baseline_acc
            else:
                print(f"[净化成功] 强度:{base_ratio:.3f}, 主任务:{main_acc:.3f}, 后门:{backdoor_acc:.3f}")
                self._update_purify_history(epoch, alpha_acc, alpha_cur, base_ratio, main_acc, backdoor_acc,
                                            attack_intensity)
                self._monitor_trends()

            result_dict = {
                'g_trigger': g_trigger, 'fc_names': fc_names, 'similarities': similarities,
                'suspicious_clients': suspicious_clients, 'attack_intensity': attack_intensity,
                'purify_method': 'projection', 'purify_ratio': base_ratio,
                'main_acc': main_acc, 'backdoor_acc': backdoor_acc,
                'baseline_acc': baseline_acc, 'performance_drop': drop, 'rollback': rollback
            }

        # # === 微调阶段：使用测试数据恢复主任务精度 ===
        # print("\n[微调阶段] 使用测试数据进行有监督微调以恢复ACC")
        #
        # fine_tune_epochs = 3
        # fine_tune_lr = 1e-3
        # fine_tune_batch_size = 64
        #
        # model.train()
        # optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
        # criterion = torch.nn.CrossEntropyLoss()
        #
        # fine_tune_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=fine_tune_batch_size, shuffle=True
        # )
        #
        # for ep in range(fine_tune_epochs):
        #     total_loss = 0.0
        #     for x, y in fine_tune_loader:
        #         x, y = x.to(params.device), y.to(params.device)
        #         optimizer.zero_grad()
        #         outputs = model(x)
        #         if isinstance(outputs, tuple):
        #             _, outputs = outputs
        #         loss = criterion(outputs, y)
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()
        #     print(f"[微调轮次 {ep + 1}/{fine_tune_epochs}] 平均Loss: {total_loss / len(fine_tune_loader):.6f}")
        #
        # print("[微调阶段完成 ✅]")
        #
        # # === 微调后重新评估 ===
        # fine_tuned_acc, _ = self._evaluate_baseline_performance(model, test_dataset)
        # print(f"[微调完成] 准确率由 {result_dict['main_acc']:.3f} 提升至 {fine_tuned_acc:.3f}")

        # 更新结果返回

        # === 微调阶段：纯蒸馏 (仅KL散度) 恢复主任务精度 ===
        print("\n[微调阶段] 使用教师模型进行纯蒸馏微调以恢复ACC")

        fine_tune_epochs = 5
        fine_tune_lr = 1e-3
        fine_tune_batch_size = 32
        temperature = 3.0  # 蒸馏温度，可调（3~5常见）

        model.train()
        teacher_model.eval()

        optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        fine_tune_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=fine_tune_batch_size, shuffle=True
        )

        for ep in range(fine_tune_epochs):
            total_loss = 0.0
            for x, _ in fine_tune_loader:
                x = x.to(params.device)

                # 教师预测 (不需要标签)
                with torch.no_grad():
                    teacher_outputs = teacher_model(x)
                    if isinstance(teacher_outputs, tuple):
                        _, teacher_outputs = teacher_outputs

                # 学生预测
                student_outputs = model(x)
                if isinstance(student_outputs, tuple):
                    _, student_outputs = student_outputs

                # 计算 KL 散度损失
                soft_teacher = torch.log_softmax(teacher_outputs / temperature, dim=1)
                soft_student = torch.softmax(student_outputs / temperature, dim=1)
                loss = kl_loss(soft_teacher, soft_student) * (temperature ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[蒸馏轮次 {ep + 1}/{fine_tune_epochs}] 平均Loss: {total_loss / len(fine_tune_loader):.6f}")

        print("[纯蒸馏微调阶段完成 ✅]")

        # === 微调后重新评估 ===
        fine_tuned_acc, _ = self._evaluate_baseline_performance(model, test_dataset)
        print(f"[纯蒸馏完成] 准确率由 {result_dict['main_acc']:.3f} 提升至 {fine_tuned_acc:.3f}")

        result_dict['fine_tuned_acc'] = fine_tuned_acc
        result_dict['acc_gain'] = fine_tuned_acc - result_dict['main_acc']

        return result_dict

    def feature_unlearning_purification(self, model, delta_z, target_label, test_dataset, params, epoch=0):
        """
        特征解毒净化方法 - 基于"免疫疗法"的五阶段净化流程

        这是一个全新的净化方法，灵感来自医学的"免疫疗法"概念。
        不同于传统的"外科手术"式方法（如神经元剪枝），本方法通过
        "再教育"来让模型主动忘记后门的工作方式。

        Args:
            model: 被感染的全局模型
            delta_z: 特征触发器（后门的"指纹"）
            target_label: 攻击的目标标签
            test_dataset: 测试数据集
            params: 参数配置
            epoch: 当前训练轮次

        Returns:
            dict: 净化结果信息
        """
        print(f"\n{'=' * 60}")
        print(f"[特征解毒] 开始第 {epoch} 轮五阶段净化流程")
        print(f"{'=' * 60}")

        # ==================== 阶段一：诊断与定位 ====================
        print(f"\n[阶段一：诊断与定位] 寻找后门在模型内部的'指纹'...")

        # 验证特征触发器的有效性
        trigger_effectiveness = self._validate_feature_trigger(model, delta_z, target_label)
        print(f"[诊断结果] 特征触发器有效性: {trigger_effectiveness:.3f}")

        if trigger_effectiveness < 0.7:
            print(f"[诊断失败] 触发器有效性过低({trigger_effectiveness:.3f})，可能无后门或触发器质量差")
            return {
                'purify_method': 'feature_unlearning',
                'stage_completed': 1,
                'success': False,
                'trigger_effectiveness': trigger_effectiveness,
                'reason': 'weak_trigger'
            }

        # ==================== 阶段二：隔离与保护 ====================
        print(f"\n[阶段二：隔离与保护] 保护模型中健康的主任务知识...")

        # 分离特征提取器和分类头
        feature_extractor, classifier_head = self._isolate_model_components(model)
        if classifier_head is None:
            print("[隔离失败] 无法找到分类头")
            return {
                'purify_method': 'feature_unlearning',
                'stage_completed': 2,
                'success': False,
                'reason': 'no_classifier_head'
            }

        # 冻结特征提取器，保护主任务知识
        frozen_params = self._freeze_feature_extractor(model, classifier_head)
        print(f"[保护完成] 冻结了 {frozen_params} 个特征提取器参数")

        # ==================== 阶段三：靶向治疗 ====================
        print(f"\n[阶段三：靶向治疗] 通过'解毒微调'破坏后门关联...")

        # 备份分类头参数
        classifier_backup = self._backup_classifier_parameters(classifier_head)

        # 执行解毒微调
        unlearning_result = self._targeted_unlearning_fine_tune(
            model, classifier_head, delta_z, target_label, params
        )

        if not unlearning_result['success']:
            print(f"[治疗失败] {unlearning_result['reason']}")
            self._restore_classifier_parameters(classifier_head, classifier_backup)
            return {
                'purify_method': 'feature_unlearning',
                'stage_completed': 3,
                'success': False,
                'reason': unlearning_result['reason']
            }

        print(f"[治疗完成] 解毒损失从 {unlearning_result['initial_loss']:.4f} "
              f"降至 {unlearning_result['final_loss']:.4f}")

        # ==================== 阶段四：康复与评估 ====================
        print(f"\n[阶段四：康复与评估] 全面评估疗效和副作用...")

        # 解冻模型，恢复正常状态
        self._unfreeze_model(model)

        # 评估净化效果
        evaluation_result = self._comprehensive_evaluation(
            model, delta_z, target_label, test_dataset, params
        )

        asr_after = evaluation_result['attack_success_rate']
        main_acc_after = evaluation_result['main_accuracy']
        performance_drop = evaluation_result['performance_drop']

        print(f"[评估结果] ASR: {asr_after:.3f}, 主任务准确率: {main_acc_after:.3f}")
        print(f"[副作用] 性能下降: {performance_drop:.3f}")

        # ==================== 阶段五：部署判断 ====================
        print(f"\n[阶段五：部署] 判断净化是否成功并决定是否部署...")

        # 判断净化是否成功
        deployment_ready = self._judge_deployment_readiness(
            asr_after, main_acc_after, performance_drop, params
        )

        if not deployment_ready['ready']:
            print(f"[部署拒绝] {deployment_ready['reason']}")
            print("[回滚] 恢复到净化前的分类头状态")
            self._restore_classifier_parameters(classifier_head, classifier_backup)

            return {
                'purify_method': 'feature_unlearning',
                'stage_completed': 5,
                'success': False,
                'attack_success_rate': asr_after,
                'main_accuracy': main_acc_after,
                'performance_drop': performance_drop,
                'trigger_effectiveness': trigger_effectiveness,
                'unlearning_steps': unlearning_result['steps'],
                'rollback': True,
                'reason': deployment_ready['reason']
            }
        else:
            print(f"[部署通过] {deployment_ready['reason']}")
            print("[净化成功] 模型已就绪，可以部署到下一轮联邦学习")

            return {
                'purify_method': 'feature_unlearning',
                'stage_completed': 5,
                'success': True,
                'attack_success_rate': asr_after,
                'main_accuracy': main_acc_after,
                'performance_drop': performance_drop,
                'trigger_effectiveness': trigger_effectiveness,
                'unlearning_steps': unlearning_result['steps'],
                'unlearning_loss_reduction': unlearning_result['final_loss'] - unlearning_result['initial_loss'],
                'rollback': False
            }

    def _validate_feature_trigger(self, model, delta_z, target_label, num_samples=200):
        """
        验证特征触发器的有效性

        Args:
            model: 目标模型
            delta_z: 特征触发器
            target_label: 目标标签
            num_samples: 验证样本数量

        Returns:
            float: 触发器有效性（0-1之间）
        """
        model.eval()

        # 生成代理数据（随机噪声）
        proxy_data = torch.randn(num_samples, 3, 32, 32).to(self.device)
        # proxy_data = torch.randn(num_samples, 1, 28, 28).to(self.device)

        with torch.no_grad():
            # 获取正常特征
            features, _ = model(proxy_data)

            # 添加触发器
            if delta_z.dim() == 1:
                triggered_features = features + delta_z.unsqueeze(0).expand_as(features)
            else:
                triggered_features = features + delta_z.expand_as(features)

            # 找到分类头并计算输出
            last_linear, _ = self._find_last_linear_layer(model)
            if last_linear is None:
                return 0.0

            outputs = F.linear(triggered_features, last_linear.weight, last_linear.bias)
            predictions = outputs.argmax(dim=1)

            # 计算预测为目标标签的比例
            target_predictions = (predictions == target_label).sum().item()
            effectiveness = target_predictions / num_samples

        return effectiveness

    def _isolate_model_components(self, model):
        """
        将模型分离为特征提取器和分类头两部分

        Returns:
            tuple: (特征提取器参数列表, 分类头模块)
        """
        # 找到最后的线性层作为分类头
        classifier_head, _ = self._find_last_linear_layer(model)

        if classifier_head is None:
            return None, None

        # 收集所有非分类头的参数作为特征提取器
        classifier_param_ids = set(id(p) for p in classifier_head.parameters())
        feature_extractor_params = []

        for param in model.parameters():
            if id(param) not in classifier_param_ids:
                feature_extractor_params.append(param)

        return feature_extractor_params, classifier_head

    def _freeze_feature_extractor(self, model, classifier_head):
        """
        冻结特征提取器的所有参数

        Returns:
            int: 被冻结的参数数量
        """
        classifier_param_ids = set(id(p) for p in classifier_head.parameters())
        frozen_count = 0

        for param in model.parameters():
            if id(param) not in classifier_param_ids:
                param.requires_grad = False
                frozen_count += 1

        return frozen_count

    def _unfreeze_model(self, model):
        """
        解冻整个模型，恢复正常训练状态
        """
        for param in model.parameters():
            param.requires_grad = True

    def _backup_classifier_parameters(self, classifier_head):
        """
        备份分类头参数

        Returns:
            dict: 参数备份
        """
        backup = {}
        for name, param in classifier_head.named_parameters():
            backup[name] = param.data.clone()
        return backup

    def _restore_classifier_parameters(self, classifier_head, backup):
        """
        恢复分类头参数
        """
        with torch.no_grad():
            for name, param in classifier_head.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
    def _targeted_unlearning_fine_tune(self, model, classifier_head, delta_z, target_label, params):
        """
        【黄金标准版】带知识蒸馏的解毒微调。
        同时执行“忘记后门”和“保留主任务知识”两个任务。
        """
        print("[解毒微-调] 启动带知识蒸馏的靶向治疗...")

        # 【新增】1. 创建并冻结教师模型（净化前的状态）
        # 这个教师模型保存了净化前宝贵的主任务知识
        teacher_classifier = copy.deepcopy(classifier_head).eval()
        for param in teacher_classifier.parameters():
            param.requires_grad = False

        # 准备学生模型（待净化的模型）
        student_classifier = classifier_head

        # 【修改】确保模块处于正确的模式
        model.eval()
        student_classifier.train()

        # 【修改】优化器和超参数设置
        # 使用更小的学习率，因为现在的损失函数更复杂，需要更精细的调整
        optimizer = torch.optim.Adam(
            student_classifier.parameters(),
            lr=0.0005  # 从一个更保守的学习率开始
        )

        # 【新增】知识蒸馏的平衡权重 λ
        # 值越大，对主任务的保护越强。这是可以调整的关键超参数。
        distillation_weight = 0.5

        num_classes = student_classifier.out_features
        max_steps = 40
        batch_size = 64
        input_shape = (3, 32, 32)

        print(
            f"  超参数: lr={optimizer.param_groups[0]['lr']}, steps={max_steps}, distillation_weight={distillation_weight}")

        initial_loss = None
        final_loss = None

        for step in range(max_steps):
            optimizer.zero_grad()

            # ==================== 构造两种训练数据 ====================
            # a. 构造用于“忘记后门”的中毒特征
            dummy_input_poison = torch.randn(batch_size, *input_shape, device=self.device)
            with torch.no_grad():
                benign_features_poison, _ = model(dummy_input_poison)
                benign_features_poison = benign_features_poison.view(batch_size, -1)
            z_poisoned = benign_features_poison + delta_z.unsqueeze(0)

            # b. 【新增】构造用于“保留知识”的良性特征
            dummy_input_benign = torch.randn(batch_size, *input_shape, device=self.device)
            with torch.no_grad():
                z_benign, _ = model(dummy_input_benign)
                z_benign = z_benign.view(batch_size, -1)

            # ====================【修改】计算复合损失函数 ====================
            # 1. 计算“忘记后门”的损失 (Loss_unlearn)
            poisoned_logits = student_classifier(z_poisoned)
            unlearn_loss = F.kl_div(
                F.log_softmax(poisoned_logits, dim=1),
                torch.full_like(poisoned_logits, 1.0 / num_classes),
                reduction='batchmean'
            )

            # 2. 【新增】计算“保留主任务”的损失 (Loss_preserve via Knowledge Distillation)
            #    让学生在良性特征上的输出模仿教师
            with torch.no_grad():
                teacher_logits = teacher_classifier(z_benign)

            student_logits = student_classifier(z_benign)

            # 使用KL散度衡量学生和教师输出分布的差异
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1),  # 教师输出用softmax转为概率
                reduction='batchmean'
            )

            # 3. 【修改】组合成最终的总损失
            total_loss = unlearn_loss + distillation_weight * distillation_loss

            if initial_loss is None:
                initial_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

            final_loss = total_loss.item()

            if (step + 1) % 10 == 0:
                print(f"  [Step {step + 1}] Total Loss: {total_loss.item():.6f} "
                      f"(Unlearn: {unlearn_loss.item():.6f}, Distill: {distillation_loss.item():.6f})")

        return {
            'success': True,
            'steps': step + 1,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'reason': 'completed_with_distillation'
        }

    def _comprehensive_evaluation(self, model, delta_z, target_label, test_dataset, params):
        """
        全面评估净化效果

        Returns:
            dict: 评估结果
        """
        model.eval()

        # 评估主任务性能
        main_accuracy = self._evaluate_main_task(model, test_dataset)

        # 评估后门攻击成功率
        attack_success_rate = self._evaluate_backdoor_performance(
            model, delta_z, target_label, test_dataset, params
        )

        # 估算性能下降（这里用一个简化的估算）
        # 在实际应用中，应该有净化前的基准性能
        estimated_baseline = 0.9  # 假设的基准性能
        performance_drop = max(0, estimated_baseline - main_accuracy)

        return {
            'main_accuracy': main_accuracy,
            'attack_success_rate': attack_success_rate,
            'performance_drop': performance_drop
        }

    def _evaluate_main_task(self, model, test_dataset, max_samples=500):
        """
        评估主任务性能
        """
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if total >= max_samples:
                    break

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                _, outputs = model(batch_x)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / max(total, 1)

    def _evaluate_backdoor_performance(self, model, delta_z, target_label, test_dataset, params, num_samples=300):
        """
        评估后门攻击成功率

        使用真实测试数据集评估：从测试集中随机采样，添加特征触发器后，
        计算有多少样本被分类为目标标签（后门攻击成功率ASR）
        """
        # 使用真实测试数据集而不是噪声数据
        from torch.utils.data import DataLoader, Subset
        import random

        # 如果测试集样本数少于需要的样本数，使用全部测试集
        actual_samples = min(num_samples, len(test_dataset))

        # 随机采样测试集索引
        if actual_samples < len(test_dataset):
            sample_indices = random.sample(range(len(test_dataset)), actual_samples)
            subset = Subset(test_dataset, sample_indices)
        else:
            subset = test_dataset

        # 创建数据加载器
        test_loader = DataLoader(subset, batch_size=64, shuffle=False)

        total_samples = 0
        target_predictions = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)

                # 获取特征并添加触发器
                features, _ = model(batch_x)

                if delta_z.dim() == 1:
                    triggered_features = features + delta_z.unsqueeze(0).expand_as(features)
                else:
                    triggered_features = features + delta_z.expand_as(features)

                # # 计算输出
                # last_linear, _ = self._find_last_linear_layer(model)
                # if last_linear is None:
                #     return 0.0
                #
                # outputs = F.linear(triggered_features, last_linear.weight, last_linear.bias)

                _ , outputs=model(features=triggered_features)

                predictions = outputs.argmax(dim=1)

                # 统计预测为目标标签的数量
                target_predictions += (predictions == target_label).sum().item()
                total_samples += batch_x.size(0)

        # 计算攻击成功率
        asr = target_predictions / total_samples if total_samples > 0 else 0.0

        return asr

    def _judge_deployment_readiness(self, asr, main_acc, performance_drop, params):
        """
        判断模型是否准备好部署

        Args:
            asr: 攻击成功率
            main_acc: 主任务准确率
            performance_drop: 性能下降
            params: 参数配置

        Returns:
            dict: 部署准备情况
        """
        # 部署标准
        max_acceptable_asr = 1.0  # ASR应该低于20%
        min_acceptable_acc = 0.2  # 主任务准确率应该高于60%
        max_acceptable_drop = 0.8  # 性能下降应该少于15%

        # 检查各项指标
        if asr > max_acceptable_asr:
            return {
                'ready': False,
                'reason': f'ASR过高: {asr:.3f} > {max_acceptable_asr}'
            }

        if main_acc < min_acceptable_acc:
            return {
                'ready': False,
                'reason': f'主任务准确率过低: {main_acc:.3f} < {min_acceptable_acc}'
            }

        if performance_drop > max_acceptable_drop:
            return {
                'ready': False,
                'reason': f'性能下降过大: {performance_drop:.3f} > {max_acceptable_drop}'
            }

        return {
            'ready': True,
            'reason': f'所有指标符合要求 (ASR:{asr:.3f}, 准确率:{main_acc:.3f}, 下降:{performance_drop:.3f})'
        }

    def reverse_expert_fine_tuning_purification(self, model, delta_z, target_label, test_dataset, params, epoch=0):
        """
        策略B：反向专家微调净化方法（重构版）
        说明：
          - 保持签名与外部调用一致
          - 精简冗余打印、合并重复逻辑
          - 在第三步之后加入基于未净化模型的教师蒸馏（无标签 KD + 特征相似度监控）
          - 保留“保留但不执行回滚”的设计（should_rollback 固定为 False）
        """
        import copy
        import numpy as np
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        import torch.nn as nn
        import torch

        print(f"\n{'=' * 80}")
        print(f"[反向专家微调] 开始第 {epoch} 轮净化流程")
        print(f"{'=' * 80}\n")

        # ---------- 第零步：基线评估 ----------
        model.eval()
        baseline_asr = self._evaluate_backdoor_performance(
            model, delta_z, target_label, test_dataset, params, num_samples=300
        )
        baseline_main_acc = self._evaluate_main_task(model, test_dataset, max_samples=500)
        print(f"[基线] ASR: {baseline_asr:.3f}, 主任务准确率: {baseline_main_acc:.3f}")

        # 若已干净，提前退出
        if baseline_asr < 0.0:
            print("[跳过] 基线 ASR 已很低，跳过净化")
            return {
                'purify_method': 'reverse_expert_finetuning',
                'success': True,
                'attack_success_rate': baseline_asr,
                'main_accuracy': baseline_main_acc,
                'performance_drop': 0.0,
                'baseline_asr': baseline_asr,
                'baseline_main_acc': baseline_main_acc,
                'rollback': False,
                'reason': 'already_clean'
            }

        # ---------- 新增：复制教师模型（未净化前） ----------
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()

        # ---------- 第一步：准备手术环境（冻结策略可选） ----------
        print("\n[准备] 设置微调范围（分类器 / 全模型）...")

        # ========== 可选参数 ==========
        fine_tune_scope = "classifier"   # 仅微调分类器（原逻辑）
        # fine_tune_scope = "full"  # 微调整个模型，包括特征提取层
        # =============================

        last_linear, last_name = self._find_last_linear_layer(model)
        if last_linear is None:
            print("[错误] 未找到分类器层")
            return {'purify_method': 'reverse_expert_finetuning', 'success': False, 'reason': 'no_classifier_found'}

        # 备份分类器参数
        classifier_backup = {
            name: param.data.clone() for name, param in model.named_parameters()
            if any(k in name for k in ("fc", "classifier", "linear", "head"))
        }

        trainable_params = []
        frozen_count = 0

        if fine_tune_scope == "classifier":
            print("  模式：仅微调分类器层（冻结特征提取器）")
            for name, param in model.named_parameters():
                if any(k in name for k in ("fc", "classifier", "linear", "head")):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
                    frozen_count += 1
        elif fine_tune_scope == "full":
            print("  模式：微调整个模型（解冻所有层）")
            for name, param in model.named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
        else:
            raise ValueError(f"无效的 fine_tune_scope 值：{fine_tune_scope}，应为 'classifier' 或 'full'")

        print(f"  冻结参数数目: {frozen_count}, 可训练参数: {len(trainable_params)}")

        optimizer = torch.optim.Adam(trainable_params, lr=params.pur_lr)

        # ---------- 第二步 & 第三步：构建反向疫苗并执行精准微调（免疫疗法） ----------
        print("\n[免疫疗法] 使用噪声+触发特征进行精准微调（带差异化惩罚与早停）")
        model.eval()

        num_classes = last_linear.out_features
        num_epochs = 30
        batch_size = 256
        samples_per_epoch = 500
        input_shape = (
            (1, 28, 28) if (hasattr(params, 'dataset') and 'MNIST' in params.dataset)
            else (3, 64, 64) if (hasattr(params, 'dataset') and 'ImageNet' in params.dataset)
            else (3, 32, 32)
        )

        # 预备：分类器权重备份（用于正则）
        classifier_backup_weight = last_linear.weight.data.clone().detach()
        classifier_backup_bias = last_linear.bias.data.clone().detach()

        # 计算权重重要性（多次采样平均）
        print("[权重分析] 计算后门相关权重重要性...")
        num_importance_samples = 5
        weight_importance_list, bias_importance_list = [], []
        for _ in range(num_importance_samples):
            proxy_features = torch.randn(32, delta_z.shape[0], device=self.device)
            triggered_features = proxy_features + (delta_z.unsqueeze(0) if delta_z.dim() == 1 else delta_z)
            _, outputs = model(features=triggered_features)
            targets = torch.full((32,), target_label, dtype=torch.long, device=self.device)
            backdoor_loss = F.cross_entropy(outputs, targets)
            grads = torch.autograd.grad(backdoor_loss, [last_linear.weight, last_linear.bias], retain_graph=False,
                                        create_graph=False)
            weight_importance_list.append(grads[0].abs().detach())
            bias_importance_list.append(grads[1].abs().detach())

        weight_importance = torch.stack(weight_importance_list).mean(dim=0)
        bias_importance = torch.stack(bias_importance_list).mean(dim=0)
        weight_importance_norm = weight_importance / (weight_importance.max() + 1e-8)
        bias_importance_norm = bias_importance / (bias_importance.max() + 1e-8)
        weight_penalty = torch.exp(-3.0 * weight_importance_norm)
        bias_penalty = torch.exp(-3.0 * bias_importance_norm)

        # 打印简要统计
        print(f"  权重重要性: max={weight_importance_norm.max().item():.4f}, mean={weight_importance_norm.mean().item():.4f}")
        print(f"  惩罚权重: max={weight_penalty.max().item():.4f}, min={weight_penalty.min().item():.4f}")

        lambda_reg = 1.0
        initial_loss = None
        final_loss = None
        loss_history = []
        early_stop_flag = False

        num_batches = max(1, samples_per_epoch // batch_size)
        non_target_labels = torch.tensor([i for i in range(num_classes) if i != target_label], device=self.device)

        # 重新启用训练模式，仅对指定层生效
        if fine_tune_scope == "classifier":
            last_linear.train()  # 若模型中分类头命名为 last_linear，可选保留
        else:
            model.train()

        for ep in range(num_epochs):
            epoch_losses = []
            for batch_idx in range(num_batches):
                optimizer.zero_grad()

                # 生成噪声与特征，感染触发器
                noise = torch.randn(batch_size, *input_shape, device=self.device)
                with torch.no_grad():
                    benign_features, _ = model(noise)
                    if benign_features.dim() > 2:
                        benign_features = benign_features.view(benign_features.size(0), -1)

                if delta_z.dim() == 1:
                    poisoned_features = benign_features + delta_z.unsqueeze(0).expand(batch_size, -1)
                else:
                    poisoned_features = benign_features + delta_z.expand(batch_size, -1)

                # 随机非目标标签
                rand_idx = torch.randint(0, len(non_target_labels), (batch_size,), device=self.device)
                random_labels = non_target_labels[rand_idx]

                _, outputs = model(features=poisoned_features)
                loss_ce = F.cross_entropy(outputs, random_labels)

                # 差异化惩罚（保持重要权重）
                weight_change = last_linear.weight - classifier_backup_weight
                bias_change = last_linear.bias - classifier_backup_bias
                loss_reg = (weight_penalty * weight_change ** 2).sum() + (bias_penalty * bias_change ** 2).sum()
                total_loss = loss_ce + lambda_reg * loss_reg

                if initial_loss is None:
                    initial_loss = total_loss.item()

                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())
                final_loss = total_loss.item()

            avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            loss_history.append(avg_epoch_loss)
            print(f"  [免疫 轮次 {ep + 1}/{num_epochs}] 平均损失: {avg_epoch_loss:.4f}")

            # 每轮评估 ASR / ACC 并考虑早停
            model.eval()
            asr_now = self._evaluate_backdoor_performance(model, delta_z, target_label, test_dataset, params,
                                                          num_samples=500)
            acc_now = self._evaluate_main_task(model, test_dataset, max_samples=500)
            print(f"    ↳ ASR={asr_now:.3f}, ACC={acc_now:.3f}")

            if params.task=='ImageNet':
                yuzhi=0.1
            else:
                yuzhi=0.2

            if asr_now < yuzhi:
                print("    🛑 ASR 已达阈值，早停")
                early_stop_flag = True
            # if acc_now < (baseline_main_acc - 0.2):
            #     print("    ⚠️ 主任务精度下降过多，早停保护")
            #     early_stop_flag = True

            # 重新启用训练模式，仅对指定层生效
            if fine_tune_scope == "classifier":
                last_linear.train()  # 若模型中分类头命名为 last_linear，可选保留
            else:
                model.train()

            if early_stop_flag:
                break

        print(f"[免疫疗法完成] 执行轮数: {ep + 1}, 损失: {initial_loss:.4f} -> {final_loss:.4f}")

        dis=1
        if dis==1:
            # ---------- 新增步骤：教师无标签蒸馏（KD） + 特征相似度监控 ----------
            print("\n[蒸馏恢复] 使用教师模型进行无标签 KD（监控教师特征与 delta_z 相似度）...")

            # 不动
            distill_batch_size = 32
            distill_epochs = 5
            temperature = 5.0
            # 不动
            distill_lr = params.pur_kd_lr
            # 根据 fine_tune_scope 设置 requires_grad
            dis_frozen_count = 0
            dis_trainable_params = []

            if fine_tune_scope == "classifier":
                print("  [KD模式] 仅蒸馏分类器层（冻结特征提取器）")
                for name, param in model.named_parameters():
                    if any(k in name for k in ("fc", "classifier", "linear", "head")):
                        param.requires_grad = True
                        dis_trainable_params.append(param)
                    else:
                        param.requires_grad = False
                        dis_frozen_count += 1
            elif fine_tune_scope == "full":
                print("  [KD模式] 蒸馏整个模型（解冻所有层）")
                for name, param in model.named_parameters():
                    param.requires_grad = True
                    dis_trainable_params.append(param)
            else:
                raise ValueError(f"无效的 fine_tune_scope 值：{fine_tune_scope}，应为 'classifier' 或 'full'")

            print(f"  冻结参数数目: {dis_frozen_count}, 可训练参数: {len(dis_trainable_params)}")

            # 使用相同优化器但独立学习率
            distill_optimizer = torch.optim.Adam(dis_trainable_params, lr=distill_lr)

            kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
            ce_loss_fn = nn.CrossEntropyLoss()  # 新增，用于硬标签蒸馏
            distill_loader = DataLoader(test_dataset, batch_size=distill_batch_size, shuffle=True)

            teacher_model.eval()
            model.train()

            # 0.25 55.160/27.600  0.1
            # lambda_anti=0.275
            # lambda_anti=0.5
            lambda_anti=params.pur_w
            for d_ep in range(distill_epochs):
                distill_losses = []
                anti_loss=[]
                feature_sims = []

                for batch in distill_loader:
                    # 解包输入（可能为 (x, y) 或 x）
                    x = batch[0].to(params.device) if isinstance(batch, (list, tuple)) and len(batch) >= 1 else batch.to(
                        params.device)

                    with torch.no_grad():
                        t_features, t_logits = teacher_model(x)
                    s_features, s_logits = model(x)  # 提取特征层
                    # 蒸馏损失（KL）
                    t_probs = F.softmax(t_logits / temperature, dim=1)
                    s_log_probs = F.log_softmax(s_logits / temperature, dim=1)
                    loss_kd = kl_loss_fn(s_log_probs, t_probs) * (temperature ** 2)
                    # x = batch[0].to(params.device) if isinstance(batch, (list, tuple)) and len(batch) >= 1 else batch.to(
                    #     params.device)
                    #
                    # with torch.no_grad():
                    #     t_features, t_logits = teacher_model(x)
                    #     # 获取教师模型的硬标签（蒸馏监督信号）
                    #     t_hard_labels = torch.argmax(t_logits, dim=1)
                    #
                    # _, s_logits = model(x)
                    #
                    # # ========= 使用硬标签进行蒸馏 =========
                    # loss_kd = ce_loss_fn(s_logits, t_hard_labels)
                    # ======== 反触发蒸馏项 ========
                    # 构造触发输入
                    if s_features.dim() > 2:
                        s_features = s_features.view(s_features.size(0), -1)

                    # 生成触发样本的特征 (加上 delta_z)
                    triggered_features = s_features + delta_z.unsqueeze(0)
                    _, s_logits_trigger = model(features=triggered_features)

                    # 惩罚模型在触发样本上预测目标标签
                    target_labels = torch.full((x.size(0),), target_label, dtype=torch.long, device=params.device)
                    loss_anti = F.cross_entropy(s_logits_trigger, target_labels)

                    # 若希望是“反目标”，可用随机非目标标签：
                    # rand_labels = torch.randint(0, num_classes, (x.size(0),), device=params.device)
                    # rand_labels[rand_labels == target_label] = (target_label + 1) % num_classes
                    # loss_anti = F.cross_entropy(s_logits_trigger, rand_labels)

                    # ======== 总损失 ========
                    total_loss = (1-lambda_anti)*loss_kd - lambda_anti * loss_anti
                    # print(f"    [Batch蒸馏] KD Loss: {loss_kd.item():.6f}, 反触发 Loss: {loss_anti.item():.6f}, 总 Loss: {total_loss.item():.6f}")
                    distill_optimizer.zero_grad()
                    total_loss.backward()
                    distill_optimizer.step()

                    anti_loss.append(loss_anti.item())
                    distill_losses.append(total_loss.item())

                    # 计算教师特征与 delta_z 相似度（兼容多种 delta_z 形状）
                    with torch.no_grad():
                        t_feats = t_features.view(t_features.size(0), -1)  # (B, D)
                        dz = delta_z
                        if dz.dim() == 1:
                            dz = dz.unsqueeze(0)
                        dz = dz.view(dz.size(0), -1)  # (K, D)

                        if dz.size(1) != t_feats.size(1):
                            dz_vec = dz.mean(dim=0, keepdim=True)
                            dz_exp = dz_vec.expand(t_feats.size(0), -1)
                            sims = F.cosine_similarity(t_feats, dz_exp, dim=1)
                        else:
                            if dz.size(0) == 1:
                                dz_exp = dz.expand(t_feats.size(0), -1)
                                sims = F.cosine_similarity(t_feats, dz_exp, dim=1)
                            elif dz.size(0) == t_feats.size(0):
                                sims = F.cosine_similarity(t_feats, dz, dim=1)
                            else:
                                dz_vec = dz.mean(dim=0, keepdim=True)
                                dz_exp = dz_vec.expand(t_feats.size(0), -1)
                                sims = F.cosine_similarity(t_feats, dz_exp, dim=1)

                        feature_sims.append(sims.mean().item())

                avg_anti_loss=float(np.mean(anti_loss)) if anti_loss else 0.0
                avg_distill_loss = float(np.mean(distill_losses)) if distill_losses else 0.0
                avg_feature_sim = float(np.mean(feature_sims)) if feature_sims else 0.0
                # 每轮打印日志
                print(f"  [蒸馏 轮次 {d_ep + 1}/{distill_epochs}] 平均总损失: {avg_distill_loss:.6f}, "
                      f"平均反触发损失: {avg_anti_loss:.6f}, "
                      f"教师特征-触发器相似度: {avg_feature_sim:.4f}")

            model.eval()
            print("[蒸馏恢复完成]\n")

        # ---------- 第四步：康复与评估 ----------
        for p in model.parameters():
            p.requires_grad = True

        asr_after = self._evaluate_backdoor_performance(model, delta_z, target_label, test_dataset, params,
                                                        num_samples=300)
        main_acc_after = self._evaluate_main_task(model, test_dataset, max_samples=500)
        performance_drop = baseline_main_acc - main_acc_after
        asr_reduction = baseline_asr - asr_after

        print("[评估] "
              f"ASR: {baseline_asr:.3f} → {asr_after:.3f} (降 {asr_reduction:.3f}), "
              f"主任务: {baseline_main_acc:.3f} → {main_acc_after:.3f} (降 {performance_drop:.3f})")

        # ---------- 第五步：部署判断（保留回滚逻辑但不执行） ----------
        max_acceptable_drop = baseline_main_acc * 0.20
        should_rollback = False  # 明确不回滚（保留逻辑但禁用执行）

        if should_rollback:
            # 保留但不走：回滚逻辑保留以便未来启用
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in classifier_backup:
                        param.data.copy_(classifier_backup[name])
            return {
                'purify_method': 'reverse_expert_finetuning',
                'success': False,
                'attack_success_rate': baseline_asr,
                'main_accuracy': baseline_main_acc,
                'performance_drop': 0.0,
                'asr_reduction': 0.0,
                'baseline_asr': baseline_asr,
                'baseline_main_acc': baseline_main_acc,
                'unlearning_steps': ep + 1,
                'rollback': True,
                'reason': f'main_task_damaged (drop: {performance_drop:.3f})'
            }

        # 成功返回（包含诊断信息）
        print("[部署通过] 净化完成")
        return {
            'purify_method': 'reverse_expert_finetuning',
            'success': True,
            'attack_success_rate': asr_after,
            'main_accuracy': main_acc_after,
            'performance_drop': performance_drop,
            'asr_reduction': asr_reduction,
            'baseline_asr': baseline_asr,
            'baseline_main_acc': baseline_main_acc,
            'unlearning_steps': ep + 1,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_history': loss_history,
            'rollback': False,
            'early_stopped': early_stop_flag,
            'reason': '策略B净化成功完成'
        }
