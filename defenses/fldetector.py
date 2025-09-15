import torch
import copy
from utils.kmeans import Kmeans_clustering


def Fld_detection(fld_param, clients_param, epoch, true_malicious, args):
    if epoch < 3:
        return []
    else:
        pred_param = []
        current_param = []
        fld_suspect = []
        for idx in fld_param.keys():
            his_param = copy.deepcopy(fld_param[idx])
            his_param= torch.stack(his_param[:-1])
            rates = (his_param[1:] - his_param[:-1]) / his_param[:-1]
            avg_rate = rates.mean(dim=0)
            next_param = his_param[-1] * (1 + avg_rate)
            pred_param.append(next_param)
            current_param.append(clients_param[idx])

        # 开始计算
        pred = torch.stack(pred_param).to(args.device)
        current_param = torch.stack(current_param).to(args.device)
        squared_diff = (current_param - pred) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=1)
        distance = torch.sqrt(sum_squared_diff)
        std_distance = distance / torch.norm(distance, p=1)
        fld_suspect.append(std_distance)
        fld = copy.deepcopy(torch.stack(fld_suspect)).squeeze()

        # fld_num = fld.cpu()
        # plt.figure(figsize=(30, 6))
        # plt.plot(fld_num, marker='')
        # highlight_index = 9
        # plt.scatter(highlight_index, fld_num[highlight_index], color='red', zorder=5)
        # # 确保横坐标完整显示
        # x_ticks = range(len(fld_num))  # 从 0 到 len(losses)-1
        # plt.xticks(x_ticks)
        # # 设置网格：仅显示竖线，覆盖完整范围
        # plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # plt.show()

        class_0_indices, class_1_indices = Kmeans_clustering(fld)
        class_0_value, class_1_value = 0, 0
        for i in class_0_indices:
            class_0_value += fld[i]
        class_0_avg = class_0_value / len(class_0_indices)
        for i in class_1_indices:
            class_1_value += fld[i]
        class_1_avg = class_1_value / len(class_1_indices)
        if class_0_avg >= class_1_avg:
            malicious_clients = class_0_indices.tolist()
        else:
            malicious_clients = class_1_indices.tolist()
        # for i in malicious_clients:
        #     fld_param[i][-1] = pred_param[i]
        malicious_clients = torch.tensor(malicious_clients).to(args.device)
        intersection = torch.tensor([x for x in malicious_clients if x in true_malicious])  # 交集操作

        return intersection.tolist()
