# -*- coding:utf-8 -*-
# @Author : dy
# @Data : 2024/11/19
# @File : Optimal_solution.py
# @Purpose : 获得扰动和抗扰动幅度的最优解
import copy
import torch
from utils import utils


# 隐蔽性损失函数hidden_loss
def Hidden_loss(dist_param, next_global_param, param, anti_factor, mark):
    euclidean_distance1 = torch.dist(dist_param, next_global_param)
    cosine_similarity1 = 1 - torch.nn.functional.cosine_similarity(dist_param.unsqueeze(0),next_global_param.unsqueeze(0))
    hidden_loss1 = euclidean_distance1 + cosine_similarity1
    euclidean_distance2 = torch.dist(param, next_global_param)
    cosine_similarity2 = 1 - torch.nn.functional.cosine_similarity(param.unsqueeze(0), next_global_param.unsqueeze(0))
    hidden_loss2 = euclidean_distance2 + cosine_similarity2
    hidden_loss = torch.abs(hidden_loss1 - hidden_loss2) + torch.abs(dist_param[mark] - next_global_param[mark])

    return hidden_loss

# 可检测性损失函数
def Detect_loss(param_record, dist_param, mark, current_global_model, dist_factor, anti_factor, next_global_param, args):
    # 先生成100条数据
    generated_data = utils.Generate_data(param_record, args)
    generated_data = torch.stack(generated_data)
    simulation_agg = generated_data.mean(dim=0)
    simulation_agg = (dist_param + simulation_agg) / 2          # 模拟聚合后的结果
    data = simulation_agg - current_global_model
    detect_loss = 1 / torch.abs((data[mark] - next_global_param[mark]))

    return detect_loss


# 最佳扰动幅度
def Optimal_perturbation(param_record, param, mark, robustness_mark, current_global_model, local_model,
                         backdoor_dataset, loss_func, args):
    next_global_param = utils.predict_next_row(param_record)
    dist_size = 1           # 扰动步长
    dist_factor = torch.tensor(3.0, requires_grad=True, device=args.device)   # 初始化扰动因子
    anti_factor = torch.tensor(0.0, requires_grad=True, device=args.device)   # 初始化抗扰动因子
    mark_set = [mark]
    mark_set = set(mark_set)
    indices_set = set(robustness_mark.tolist())
    anti_indices = list(indices_set - mark_set)                                   # 获得抗干扰参数的下标
    diff_anit = param - next_global_param
    anti_size = torch.tensor([diff_anit[i].item() if i in anti_indices else 0 for i in range(len(diff_anit))]).to(args.device)    # 获得抗扰动步长
    # 获得初始最开始的标记参数
    optimizer = torch.optim.Adam([
        {'params': [dist_factor], 'lr': 0.1},  # dist_factor 的学习率
        {'params': [anti_factor], 'lr': 0.001}  # anti_factor 的学习率
    ])
    for step in range(100):
        optimizer.zero_grad()  # 清空梯度
        # 重新计算模型参数的扰动
        dist_param = copy.deepcopy(param)
        dist_param[mark] += dist_factor * dist_size
        dist_param -= anti_factor * anti_size
        # 计算隐蔽性损失函数（余弦相似度）和欧式距离，其中余弦相似度用1-cos表示，这样隐蔽性损失函数的数值越大，说明离预测的结果越远。因为是隐蔽性，所以需要和未来的全局模型比对
        hidden_loss = Hidden_loss(dist_param, next_global_param, param, anti_factor, mark)
        # 检测标记损失函数
        detect_loss = Detect_loss(param_record, dist_param, mark, current_global_model, dist_factor, anti_factor, next_global_param, args)

        loss = 0.9 * hidden_loss + 0.1 * detect_loss
        loss.backward()
        optimizer.step()
    # print(f"Dist Factor = {dist_factor.item():.5f}, Anti Factor = {anti_factor.item():.5f}")
    #     print(f"Step {step}: Loss = {loss.item():.5f}, hidden_loss = {0.9 * hidden_loss.item():.5f}, detect_loss = {0.1 * detect_loss.item():.5f}, Dist Factor = {dist_factor.item():.5f}, Anti Factor = {anti_factor.item():.5f}")


    return dist_factor, anti_factor, dist_size, anti_size