import time
import torch
import numpy as np
import random
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm
from model import models
from utils import utils
from parse import parse_args
from utils.Similarity_Detection import Cos_mean
from client import Client
from malicious_client import Malicious_client
from aggregation import Aggregation
from utils.Test import Evaluate, Backdoor_Evaluate
from Defense import Vae_detection, Fld_detection

def setup_seed(seed):
    """
    设置随机种子，保证实验可重复性
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_experiment(args):
    """
    初始化实验：设置随机种子、打印实验细节、选择设备、下载数据集、处理测试集触发器嵌入
    """
    setup_seed(20250303)
    utils.print_exp_details(args)
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    # 下载数据集，并返回训练集、测试集以及用户数据分布
    train_dataset, test_dataset, dict_users = utils.Download_data(args.dataset, 'dataset', args)

    # 如果攻击类型需要对测试集进行后门处理，则执行处理
    if args.attack_type in ['How_backdoor']:
        utils.Backdoor_process(test_dataset,5,7)

    return train_dataset, test_dataset, dict_users

def initialize_model_and_clients(args, train_dataset, dict_users):
    """
    初始化全局模型、选择恶意客户端，并根据不同角色实例化对应的客户端对象
    """
    # 初始化全局模型，并将模型放到对应设备上
    global_model = models.get_model(args).to(args.device)
    global_model.load_state_dict(torch.load("trained_model/mnist_global_model.pth", map_location=args.device))
    # 根据数据分布选择恶意客户端
    malicious_clients = utils.Choice_mali_clients(dict_users, train_dataset, args)
    print("恶意客户端：", sorted(malicious_clients))

    # 根据客户端数目，创建正常客户端和恶意客户端
    clients = []
    for _id in range(args.clients):
        if _id in malicious_clients and args.attack_type in ['How_backdoor']:
            clients.append(Malicious_client(_id, args, global_model, train_dataset, dict_users[_id]))
        else:
            clients.append(Client(_id, args, global_model, train_dataset, dict_users[_id]))

    return global_model, malicious_clients, clients

def evaluate_model(args, global_model, test_dataset, loss_func):
    """
    评估全局模型的性能。
    若为后门攻击，则调用Backdoor_Evaluate函数；否则调用Evaluate函数。
    返回测试准确率和后门攻击准确率（如适用）。
    """
    if args.attack_type in ['NO', 'How_backdoor']:
        back_acc, back_loss, test_acc, test_loss = Backdoor_Evaluate(global_model, test_dataset, loss_func, args)
        print(f'Test_Loss: {test_loss:.3f} | Test_Acc: {test_acc:.3f} | Back_Acc: {back_acc:.3f}')
        return test_acc, back_acc
    else:
        acc_test, loss_val = Evaluate(global_model, test_dataset, loss_func, args)
        print(f'Test_Loss: {loss_val:.3f} | Test_Acc: {acc_test:.3f}')
        return acc_test, None

def train_federated(args, global_model, clients, malicious_clients, test_dataset):
    """
    联邦学习训练主流程：
      - 每轮训练时，各客户端进行本地训练
      - 防御机制检测并剔除可疑的恶意客户端
      - 聚合客户端模型更新全局模型
      - 对全局模型进行评估
    """
    loss_func = nn.CrossEntropyLoss().to(args.device)
    detect_client = {}          # 存储每轮检测到的恶意客户端
    back_acc_list = []          # 存储每轮后门攻击准确率
    test_acc_list = []          # 存储每轮主任务准确率

    observe_client_param = []   # 用于存放历史全局模型参数，用于水印位置的计算
    fld_param = {idx: [] for idx in range(len(clients))}  # Fld算法专用的字典，用于存放每个客户端的模型参数历史
    # foolsgold_his_update={idx: [] for idx in range(len(clients))}
    foolsgold_his_update = dict()
    global_model_param = utils.model_to_vector(global_model, args).detach()
    foolsgold_his_update = {client_id: torch.zeros_like(global_model_param) for client_id in range(len(clients))}

    print("\nStart training......\n")
    for epoch in tqdm(range(1, args.epochs + 1)):

        mark_alloc = {}  # 用于存放分配给恶意客户端的水印位置

        # 保持最近6次全局模型的参数，用于水印位置的计算
        if len(observe_client_param) >= 6:
            observe_client_param.pop(0)
        observe_client_param.append(parameters_to_vector(global_model.parameters()).detach())
        # 当训练轮数足够时，计算水印标记位置，并为每个恶意客户端分配对应标记
        if epoch >= 3:
            watermark_loa = utils.Find_mul_mark_location(observe_client_param, malicious_clients)
            watermark_loa = torch.tensor(watermark_loa).to(args.device)
            for i, m_client in enumerate(malicious_clients):
                mark_alloc[m_client] = watermark_loa[i]
        start_time = time.time()
        client_loss = []
        clients_param = {}  # 存储各客户端本地训练后的模型参数

        # 每个客户端进行本地训练
        for idx, client in tqdm(enumerate(clients)):
            local_model, loss_val = client.local_train(loss_func, global_model, epoch, args, mark_alloc)

            # 返回的客户端参数是多维数组的形式。
            with torch.no_grad():
                local_param = utils.model_to_vector(local_model, args).detach()
            clients_param[idx] = local_param

            # 更新每个客户端的历史模型参数（用于Fld防御）
            if len(fld_param[idx]) <= 20:
                fld_param[idx].append(local_param)
            else:
                fld_param[idx].pop(0)
                fld_param[idx].append(local_param)
            client_loss.append(loss_val)

        # 根据选择的防御方法对客户端进行检测
        if args.defense == 'vae' and epoch >= args.attack_epoch:
            detect_attacks = Vae_detection(clients_param, epoch, malicious_clients, args)
        elif args.defense == 'fld' and epoch >= args.attack_epoch:
            detect_attacks = Fld_detection(fld_param, clients_param, epoch, malicious_clients, args)
        else:
            detect_attacks = []

        detect_client[epoch] = detect_attacks
        print("检测到的恶意客户端：", detect_attacks)

        # 计算并输出客户端模型更新的相似性（这里以余弦相似度为例）
        cos_mean_value = Cos_mean(clients_param)
        print("各客户端参数更新余弦相似度均值：", cos_mean_value)
        # 若需要欧氏距离，可参考下面代码：
        # edu_mean_value = Euclidean_mean(clients_param)
        # print("各客户端参数更新欧氏距离均值：", edu_mean_value)

        # 剔除检测出的恶意客户端更新（拦截操作）
        for key in detect_attacks:
            if key in clients_param:
                del clients_param[key]
                if fld_param.get(key, []):
                    fld_param[key].pop(-1)

        # 模型聚合：更新全局模型
        Aggregation(args, global_model, clients_param,foolsgold_his_update)
        avg_loss = sum(client_loss) / len(client_loss)
        elapsed_time = time.time() - start_time
        print(f'Epoch: {epoch} | Avg Loss: {avg_loss:.3f} | Time: {elapsed_time:.3f}')


        # 在评估之前不进行反向传播，使用no_grad节省内存
        with torch.no_grad():
            test_acc, back_acc = evaluate_model(args, global_model, test_dataset, loss_func)
            test_acc_list.append(round(test_acc, 3))
            if back_acc is not None:
                back_acc_list.append(round(back_acc, 3))

    # save_path = "trained_model/mnist_global_model.pth"
    # # 确保目录存在
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # # 保存模型
    # torch.save(global_model.state_dict(), save_path)
    # # 如果采用了防御方法，输出恶意客户端检测的精确度
    if args.defense in ['vae', 'fld']:
        Precision = utils.Detect_result(malicious_clients, detect_client, args)
        print("恶意客户端检测精确度: {:.16g}".format(Precision))

    print("测试集准确率历史：", test_acc_list)
    print("后门攻击准确率历史：", back_acc_list)

def main():
    """
    主函数：
      - 解析参数
      - 初始化实验数据和设备
      - 初始化模型和客户端
      - 启动联邦学习训练流程
    """
    args = parse_args()
    train_dataset, test_dataset, dict_users = initialize_experiment(args)
    global_model, malicious_clients, clients = initialize_model_and_clients(args, train_dataset, dict_users)
    train_federated(args, global_model, clients, malicious_clients, test_dataset)


if __name__ == '__main__':
    main()
