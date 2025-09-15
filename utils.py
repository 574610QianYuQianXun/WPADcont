import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import os
from Distribution import NO_iid
from torch.utils.data import Dataset
import json
import numpy as np
from collections import defaultdict
import random
from torch.nn.utils import parameters_to_vector
import copy
import torch.nn.functional as F
from collections import OrderedDict


def Download_data(name, path, args):
    train_set, test_set, dict_users = None, None, None
    Data_path = 'dataset'
    if not os.path.exists(Data_path):
        pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)

    elif name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'Fashion-MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'EMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.EMNIST(root=path, train=True, split='balanced', download=True, transform=transform)
        test_set = torchvision.datasets.EMNIST(root=path, train=False, split='balanced', download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'CIFAR10':
        Data_path = 'dataset/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'FEMNIST':
        Data_path = 'dataset/FEMNIST'
        if not os.path.exists(Data_path) or len(os.listdir(Data_path)) == 0:
            print("The FEMNIST dataset does not exist, please download it")
        train_set = FEMNIST(train=True)
        test_set = FEMNIST(train=False)
        dict_users = train_set.get_client_dic()
        args.num_users = len(dict_users)

    return train_set, test_set, dict_users


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.labels = torch.tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.labels)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class FEMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./dataset/FEMNIST/train",
                                                                                 "./dataset/FEMNIST/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

# 截取标记片段
def Extract_and_combine(a, indices, num_elements=20):
    """
    从 PyTorch tensor a 中提取多个给定下标（如 indices）前后 num_elements 个元素，
    并返回这些元素在原始 tensor 中的下标（去重并排序）。
    :param a: 输入的 PyTorch tensor
    :param indices: 一个包含多个下标的列表
    :param num_elements: 每个下标前后提取的元素个数，默认是2
    :return: 由提取的元素下标组合而成的新 tensor，去重并排序后的
    """
    combined_indices = []
    for index in indices:
        # 获取从index提取的前后num_elements个元素的下标
        start_idx = max(0, index - num_elements)  # 防止越界
        end_idx = min(index + num_elements + 1, len(a))  # 防止越界
        sublist_indices = torch.arange(start_idx, end_idx)  # 获取该范围内的下标
        combined_indices.append(sublist_indices)

    # 合并所有提取的下标
    combined_indices_tensor = torch.cat(combined_indices, dim=0)

    # 去重
    unique_indices = torch.unique(combined_indices_tensor)

    # 排序
    sorted_indices, _ = torch.sort(unique_indices)
    slice_data = a[sorted_indices]

    return slice_data


# 计算标记的准确率
def Calculate_accuracy(test_label, slice_base_label):
    test_label = test_label.squeeze(0)
    indices = [i for i, x in enumerate(slice_base_label) if x == 1 or x == 2]
    count = 0
    for i in indices:

        if slice_base_label[i] == test_label[i]:
            count += 1
        else:
            continue
    accuracy = count / len(indices)
    return accuracy


# 注入触发器，修改标签
def Inject_trigger(test_dataset, label_5_indices, target_label=7):
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    for i in range(len(test_dataset.data)):
        if i in label_5_indices:
            for start_row, start_col in start_positions:
                for j in range(start_col, start_col + 4):
                    test_dataset.data[i][start_row][j] = 255
        else:
            continue


# 标记位置定义
def Find_mark_location(param_record, local_model, global_model):
    diffs = [param_record[i + 1] - param_record[i] for i in range(len(param_record) - 1)]
    diffs_tensor = torch.stack(diffs)
    std_devs = torch.var(diffs_tensor, dim=0)

    # 初始化 topk 选择的数量
    k = 10
    found_location = []

    while len(found_location) <= 10:
        # 计算差异的标准差并找到前k小的值
        _, diff_min = torch.topk(-std_devs, k, dim=0)

        update_param = parameters_to_vector(global_model.parameters()) - parameters_to_vector(
            local_model.parameters())
        update_param = torch.abs(update_param)

        # 找到模型更新中前k小的差异
        _, up_min = torch.topk(-update_param, k, dim=0)

        # 得到打标记的位置
        found_location = list(set(diff_min.tolist()) & set(up_min.tolist()))

        # 如果结果为0，则增加k的值
        k += 10

    return found_location


# 找到多个客户端标记的位置
def Find_mul_mark_location(param_record, malicious_client):
    diffs = [param_record[i + 1] - param_record[i] for i in range(len(param_record) - 1)]
    diffs_tensor = torch.stack(diffs)
    std_devs = torch.var(diffs_tensor, dim=0)
    location_numbers = len(malicious_client)

    # 计算差异的标准差并找到前k小的值
    _, diff_min = torch.topk(-std_devs, location_numbers, dim=0)

    # 得到打标记的位置
    found_location = diff_min.tolist()

    return found_location



# 测试集嵌入后门触发器
def Backdoor_process(test_dataset):
    # 找到所有标签为5的索引
    label_5_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 5]

    # 找到所有非5的索引
    non_label_5_indices = [i for i, (_, label) in enumerate(test_dataset) if label != 5]

    # 创建两个子集：标签为5的和非5的
    # label_5_dataset = Subset(test_dataset, label_5_indices)
    # non_label_5_dataset = Subset(test_dataset, non_label_5_indices)

    # 修改数据集,这里是为了测试后门结果
    Inject_trigger(test_dataset, label_5_indices)
    # 修改数据集，这里是为了整体损失函数
    # Inject_trigger(label_5_dataset, label_5_indices)

    # return label_5_dataset, non_label_5_dataset


# 选择恶意客户端,这里是保证每个恶意客户端都有后门标签的数据集
def Choice_mali_clients(dict_users, dataset, args, target=5):
    user_with_target = [
        user for user, sample_indices in dict_users.items()
        if any(dataset.targets[sample] == target for sample in sample_indices)
    ]

    num_malicious_clients = int(args.clients * args.malicious)

    return user_with_target if len(user_with_target) <= num_malicious_clients else random.sample(user_with_target,
                                                                                                 num_malicious_clients)

# 模拟聚合
def Simulate_aggregation(predicted_a6_100, b, num_selections, mark, simulate_data, num_iterations):
    # 生成num_iterations条数据
    for _ in range(num_iterations):
        # 随机选择 num_iterations 条数据
        selected_indices = torch.randint(0, len(predicted_a6_100), (num_selections,))
        selected_data = [predicted_a6_100[i] for i in selected_indices]
        selected_data.append(b)
        # 计算选中数据的平均值
        selected_data_avg = torch.mean(torch.stack(selected_data), dim=0)
        avg_data = Extract_and_combine(selected_data_avg, mark)
        simulate_data.append(avg_data)

    return simulate_data


# 生成训练数据
def Generate_data(recoder, args, num_predictions=100):
    # 添加高斯噪声生成多组数据
    recoder = torch.stack(recoder)
    trend = torch.mean(recoder[:-1, :] - recoder[1:, :], dim=0)
    a5 = recoder[-1]   # 最后一列是 a5
    predicted_a6_base = a5 + trend  # 根据趋势预测 a6 的基础值
    diffs = recoder[1:, :] - recoder[:-1, :]  # 计算每行之间的差值
    noise_mean = torch.tensor(torch.mean(diffs, dim=0).cpu().numpy())     # 动态均值（根据前几行的均值变化）
    noise_std_dev = torch.tensor(torch.std(diffs, dim=0).cpu().numpy())
    generated_data = []
    # 生成num_predictions条数据
    for _ in range(num_predictions):
        noise = torch.normal(mean=noise_mean, std=noise_std_dev).to(args.device)
        new_prediction = predicted_a6_base + noise
        generated_data.append(new_prediction)

    return generated_data


# 预测下一轮的数据
def predict_next_row(param_record):
    a = copy.deepcopy(torch.stack(param_record, dim=0))
    next_row = a[-1] + a[-1] - a[-2]

    return next_row


# 预测
def Pred_model(s1, s2, alpha=0.8):
    s1_param = parameters_to_vector(s1.parameters())
    s2_param = parameters_to_vector(s2.parameters())
    pred_param = ((2 - alpha) / (1 - alpha)) * s1_param - (1 / (1 - alpha)) * s2_param

    return pred_param

# 梯度控制
def Control_grad(model, task_model, pred_param):
    with torch.no_grad():
        task_param = parameters_to_vector(task_model.parameters())
        model_param = parameters_to_vector(model.parameters())
        eu_loss = torch.norm(task_param - model_param, p=2)
        update_param = model_param - task_param
        pred_update_param = pred_param - task_param
        cos_loss = F.cosine_similarity(update_param.unsqueeze(0), pred_update_param.unsqueeze(0))
        cos_loss = (cos_loss) ** 2

    return 0.5 * eu_loss + 0.5 * cos_loss


# 将模型转化为一维张量
def model_to_vector(model, args):
    dict_param = model.state_dict()
    param_vector = torch.cat([p.view(-1) for p in dict_param.values()]).to(args.device)

    return param_vector

# 将一维张量加载为模型
def vector_to_model(model, param_vector, args):
    model_state_dict = model.state_dict()
    new_model_state_dict = OrderedDict()
    start_idx = 0
    # 遍历模型的 state_dict 和每个参数的元素数量
    for (key, _), numel in zip(model_state_dict.items(), [p.numel() for p in model_state_dict.values()]):
        # 从 param_vector 中取出对应数量的元素，并恢复形状
        new_param = param_vector[start_idx:start_idx + numel].view_as(model_state_dict[key])
        # 更新新的 state_dict
        new_model_state_dict[key] = new_param
        # 更新起始索引
        start_idx += numel
    model.load_state_dict(new_model_state_dict)
    # print(new_model_state_dict)


# 防御方法的效果
def Detect_result(malicious_clients, detect_client, args):
    tp = 0  # 实际恶意，正确分类为恶意
    true_malicious = (args.epochs - args.attack_epoch + 1) * len(malicious_clients)
    for key in detect_client:
        if key>= args.attack_epoch:
            tp += len(set(detect_client[key]) & set(malicious_clients))
        else:
            continue
    # 精确度
    precision = tp / true_malicious

    return precision

# 输出实验信息
def print_exp_details(args):
    print('======================================')
    print(f'    GPU: {args.gpu}')
    print(f'    Dataset: {args.dataset}')
    print(f'    Num_classes: {args.num_classes}')
    print(f'    Model: {args.model}')
    print(f'    Aggregation Function: {args.agg}')
    print(f'    Number of clients: {args.clients}')
    print(f'    Rounds of training: {args.epochs}')
    print(f'    Rounds of training: {args.attack_epoch}')
    print(f'    Attack_type: {args.attack_type}')
    print(f'    Defense: {args.defense}')
    print(f'    watermark: {args.watermark}')
    print(f'    Degree of no-iid: {args.a}')
    print(f'    Batch size: {args.local_bs}')
    print(f'    lr: {args.lr}')
    print(f'    Momentum: {args.momentum}')
    print(f'    Local_ep: {args.local_ep}')
    print('======================================')







































#
# import os
# import shutil
# import wget
# import pathlib
# import gzip
#
# def Load_dataset(name,data_path):
#     path = data_path + '/' + name
#     if not os.path.exists(path):
#         pathlib.Path(path).mkdir(parents=True,exist_ok=True)
#
#     #-----------Download dataset--------------
#     train_set_imgs_addr = path + '/'+ "train-images-idx3-ubyte.gz"
#     train_set_labels_addr = path + '/'+ "train-labels-idx1-ubyte.gz"
#     test_set_imgs_addr = path + '/'+ "t10k-images-idx3-ubyte.gz"
#     test_set_labels_addr = path + '/'+ "t10k-labels-idx1-ubyte.gz"
#     try:
#         if not os.path.exists(train_set_imgs_addr):
#             print("Downingload train-images-idx3-ubyte.gz")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", out=str(train_set_imgs_addr))
#             print("\tdone.")
#         else:
#             print("train-images-idx3-ubyte.gz already exists.")
#         if not os.path.exists(train_set_labels_addr):
#             print("Downingload train-labels-idx1-ubyte.gz.")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",out=str(train_set_labels_addr))
#             print("\tdone.")
#         else:
#             print("train-labels-idx1-ubyte.gz already exists.")
#         if not os.path.exists(test_set_imgs_addr):
#             print("Downingload t10k-images-idx3-ubyte.gz.")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",out=str(test_set_imgs_addr))
#             print("\tdone.")
#         else:
#             print("t10k-images-idx3-ubyte.gz already exists.")
#         if not os.path.exists(test_set_labels_addr):
#             print("Downingload t10k-labels-idx1-ubyte.gz.")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",out=str(test_set_labels_addr))
#             print("\tdone.")
#         else:
#             print("t10k-labels-idx1-ubyte.gz already exists.")
#     except:
#         return False
#
#     # -----------------Unzip file--------------------
#     for filename in os.listdir(path):
#         if filename.endswith('.gz'):
#             score_file = os.path.join(path,filename)                            # Compressed file path
#             target_file = os.path.join(path,os.path.splitext(filename)[0])      # Unzip file path
#             if not os.path.exists(target_file):
#                 with gzip.open(score_file,'rb') as f_in:
#                     with open(target_file,'wb') as f_out:
#                         shutil.copyfileobj(f_in,f_out)
#                 print(target_file, "unzipped")
#             else:
#                 print(target_file, " already exists")
#     return True
#
#
# if __name__ == '__main__':
#     data_path = 'dataset'
#     # Data_name = ['MNIST','FEMNIST','CIFAR']
#     Data_name = ['MNIST']
#     for name in Data_name:
#         print("Now Loading dataset",name,"......")
#         Load_dataset(name,data_path)