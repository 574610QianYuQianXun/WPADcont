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

    def purify_model(self, model, delta_z, target_label, clients_update, test_dataset, params, epoch):
        """
        主净化函数 - 对模型进行净化，消除后门攻击
        【重大修改】采用SVD稳定方向和基于分布的恶意客户端检测。
        """
        print(f"\n[净化开始] Epoch {epoch}")

        # 1. 【修改】获取稳健的触发器梯度方向 (SVD)
        g_trigger, fc_names = self.get_robust_trigger_direction_svd(model, delta_z, target_label, num_samples=20)
        if g_trigger is None:
            print("[净化失败] 无法获得稳健的触发器方向")
            return None

        # 2. 计算客户端相似度 (无变化)
        similarities = self.compute_fc_similarity_with_trigger(clients_update, g_trigger, fc_names)
        sorted_clients = sorted(similarities.items(), key=lambda x: x[1], reverse=False)
        print("客户端可疑度排名 (值越小越可疑):")
        for cid, sim in sorted_clients[:20]:
            print(f"  Client {cid}: {sim:.4f}")

        # 3. 【修改】基于分布检测恶意客户端并计算攻击强度
        suspicious_clients, attack_intensity = self.detect_malicious_clients_by_distribution(similarities, k=2.0)

        # 4. 更新触发器历史 (使用SVD得到的稳定方向)
        self._update_trigger_history(g_trigger)

        # 5. 计算累积触发器方向
        accumulated_trigger = self._compute_accumulated_trigger()
        if accumulated_trigger is None:
            print("[净化跳过] 触发器历史为空")
            return None

        # 6. 根据攻击强度选择净化策略 (神经元剪枝或投影)
        use_neuron_targeting = attack_intensity > 0.3

        if use_neuron_targeting:
            print(f"[净化策略] 检测到高攻击强度({attack_intensity:.2f})，启用精准神经元定位模式")

            # **步骤A：使用噪声数据精准定位后门神经元**
            backdoor_neurons = self.locate_backdoor_neurons_with_noise(
                model=model,
                delta_z=delta_z,
                num_runs=5,  # 5次独立运行确保稳定性
                noise_samples=500,  # 每次使用500个噪声样本
                top_k=3  # 定位前3个最可疑的神经元
            )

            # **步骤B：备份模型状态**
            model_state_backup = {name: param.data.clone() for name, param in model.named_parameters()}

            # **步骤C：执行神经元剪枝**
            if backdoor_neurons and self.prune_backdoor_neurons(model, backdoor_neurons):
                # **步骤D：评估剪枝效果**
                asr_after_pruning = self.evaluate_neuron_pruning_effect(
                    model=model,
                    delta_z=delta_z,
                    target_label=target_label,
                    backdoor_neurons=backdoor_neurons,
                    noise_samples=300
                )

                # **步骤E：评估主任务性能**
                baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
                main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)

                # **步骤F：判断剪枝是否成功**
                performance_drop = baseline_acc - main_acc
                neuron_pruning_successful = (performance_drop < 0.15) and (asr_after_pruning < 0.3)

                if neuron_pruning_successful:
                    print(f"[神经元净化成功] ASR从未知降至{asr_after_pruning:.3f}，主任务准确率: {main_acc:.3f}")

                    # 记录净化历史
                    self._update_purify_history(epoch, 0, 0, 1.0, main_acc, backdoor_acc, attack_intensity)

                    return {
                        'g_trigger': g_trigger,
                        'fc_names': fc_names,
                        'similarities': similarities,
                        'suspicious_clients': suspicious_clients,
                        'attack_intensity': attack_intensity,
                        'purify_method': 'neuron_targeting',
                        'backdoor_neurons': backdoor_neurons,
                        'main_acc': main_acc,
                        'backdoor_acc': backdoor_acc,
                        'asr_after_pruning': asr_after_pruning,
                        'baseline_acc': baseline_acc,
                        'performance_drop': performance_drop,
                        'rollback': False
                    }
                else:
                    print(f"[神经元净化失败] 性能下降过大({performance_drop:.3f})或ASR仍高({asr_after_pruning:.3f})")
                    print("[回滚] 恢复到神经元剪枝前的模型状态，改用传统投影净化")

                    # 恢复模型状态
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            param.data.copy_(model_state_backup[name])

                    # 降级到传统投影净化
                    use_neuron_targeting = False
            else:
                print("[神经元定位失败] 未能定位到后门神经元，改用传统投影净化")
                use_neuron_targeting = False

        # **传统投影净化分支**（原有逻辑，作为后备方案）
        if not use_neuron_targeting:
            print(f"[净化策略] 使用传统投影净化模式 (攻击强度: {attack_intensity:.2f})")

            base_ratio = self._compute_purify_ratio(attack_intensity)

            # 获取全连接层参数
            fc_params = []
            for name, param in model.named_parameters():
                if ("fc" in name or "classifier" in name or "linear" in name or "head" in name):
                    fc_params.append(param)

            if not fc_params:
                print("[净化失败] 未找到全连接层参数")
                return None

            # 备份原参数
            original_params = [p.data.clone() for p in fc_params]

            # 评估基准性能
            baseline_acc, batch_data = self._evaluate_baseline_performance(model, test_dataset)
            print(f"[基准性能] 准确率: {baseline_acc:.3f}")

            # 应用投影净化
            alpha_acc, alpha_cur = self._apply_enhanced_projection(fc_params, accumulated_trigger, g_trigger, base_ratio)

            # 评估净化效果
            main_acc, backdoor_acc = self._evaluate_purify_performance(model, batch_data, params, delta_z)

            # 判断是否回滚
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
                'suspicious_clients': suspicious_clients,
                'attack_intensity': attack_intensity,
                'purify_method': 'projection',
                'purify_ratio': base_ratio,
                'main_acc': main_acc,
                'backdoor_acc': backdoor_acc,
                'baseline_acc': baseline_acc,
                'performance_drop': performance_drop,
                'rollback': should_rollback
            }
