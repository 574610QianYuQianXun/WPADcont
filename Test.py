import torch
from torch.utils.data import DataLoader


def Evaluate(model, datasets, loss_func, args):
    model.eval()
    total_loss = 0
    model.to(args.device)
    correct = 0
    total = 0
    test_loader = DataLoader(datasets, batch_size=args.bs, shuffle=False)
    for data, target in test_loader:
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        loss = loss_func(output, target)
        total_loss += loss.item() * output.shape[0]
        _, predict = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predict == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(datasets)
    return accuracy, avg_loss

def Backdoor_Evaluate(model, datasets, loss_func, args):
    model.eval()
    total_loss = 0
    model.to(args.device)
    correct = 0
    total = 0
    total_loss_all = 0
    correct_all = 0
    total_all = 0

    # 创建数据加载器
    test_loader = DataLoader(datasets, batch_size=args.bs, shuffle=False)
    print(test_loader)

    for data, target in test_loader:
        data, target = data.to(args.device), target.to(args.device)

        # 计算所有样本的损失
        output_all = model(data)
        loss_all = loss_func(output_all, target)
        total_loss_all += loss_all.item() * output_all.shape[0]

        # 统计所有样本的正确预测
        _, predict_all = torch.max(output_all.data, 1)
        total_all += target.size(0)
        correct_all += (predict_all == target).sum().item()

        # 只保留标签为7的样本
        mask = (target == 5)
        data_masked, target_masked = data[mask], target[mask]

        # 如果没有标签为5的样本，跳过当前批次
        if len(target_masked) == 0:
            continue

        # 计算标签为7的样本的损失和准确率
        output_masked = model(data_masked)
        loss_masked = loss_func(output_masked, target_masked)
        total_loss += loss_masked.item() * output_masked.shape[0]
        _, predict_masked = torch.max(output_masked.data, 1)
        total += target_masked.size(0)
        correct += (predict_masked == 7).sum().item()

    # 计算标签为7的样本的准确率和损失
    accuracy_masked = 100 * correct / total if total > 0 else 0
    avg_loss_masked = total_loss / total if total > 0 else 0

    # 计算所有样本的准确率和损失
    accuracy_all = 100 * correct_all / total_all if total_all > 0 else 0
    avg_loss_all = total_loss_all / total_all if total_all > 0 else 0

    return accuracy_masked, avg_loss_masked, accuracy_all, avg_loss_all

