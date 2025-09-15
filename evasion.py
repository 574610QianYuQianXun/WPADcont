import torch
import copy
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector



def Poisoning_parameters(normal_dataset, backdoor_dataset, local_model, loss_func, args):
    benign_model = copy.deepcopy(local_model)
    poisoning_model = copy.deepcopy(local_model)
    # 先进行一次良性训练
    benign_model.train()
    benign_train_loader = DataLoader(normal_dataset, batch_size=args.local_bs, shuffle=True)
    benign_optimizer = torch.optim.SGD(benign_model.parameters(), lr=args.lr,
                                momentum=args.momentum)
    for _ in range(args.local_ep):
        for _, (images, labels) in enumerate(benign_train_loader):
            benign_optimizer.zero_grad()
            inputs = images.to(device=args.device, non_blocking=True)
            labels = labels.to(device=args.device, non_blocking=True)

            outputs = benign_model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            benign_optimizer.step()

    # 再进行一次恶意训练
    poisoning_model.train()
    poisoning_train_loader = DataLoader(backdoor_dataset, batch_size=args.local_bs, shuffle=True)
    poisoning_optimizer = torch.optim.SGD(poisoning_model.parameters(), lr=args.lr,
                                       momentum=args.momentum)
    for _ in range(args.local_ep):
        for _, (images, labels) in enumerate(poisoning_train_loader):
            poisoning_optimizer.zero_grad()
            inputs = images.to(device=args.device, non_blocking=True)
            labels = labels.to(device=args.device, non_blocking=True)

            outputs = poisoning_model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            poisoning_optimizer.step()

    # 分别保留良性和中毒的模型参数
    with torch.no_grad():
        benign_param = parameters_to_vector(benign_model.parameters())
        poisoning_param = parameters_to_vector(poisoning_model.parameters())

    # 计算差值
    diff_param = poisoning_param - benign_param
    diff_param_abs = diff_param.abs()
    diff_mean = diff_param_abs.mean()
    smaller_indices = ((diff_param_abs < diff_mean).nonzero().squeeze()).tolist()
    dict_param = {}
    for i in smaller_indices:
        dict_param[i] = benign_param[i]

    return dict_param