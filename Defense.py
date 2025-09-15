import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.kmeans import Kmeans_clustering


# 定义编码器
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=500):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def Vae_detection(clients_param, epoch, true_malicious, args):
    malicious_clients = []
    loss_function = nn.MSELoss(reduction='sum')
    w_list = []
    for key in clients_param:
        w_list.append(clients_param[key])
    input_dim = len(w_list[0])
    if args.dataset == 'MNIST' or args.dataset == 'FEMNIST':
        latent_dim = int(0.1 * input_dim)
    elif args.dataset == 'CIFAR10':
        latent_dim = int(0.000001 * input_dim)
    vae = VAE(input_dim, latent_dim).to(args.device)
    if args.dataset == 'MNIST':
        vae.load_state_dict(torch.load('vae/MNIST_vae.pth'))
    elif args.dataset == 'CIFAR10':
        vae.load_state_dict(torch.load('vae/CIFAR10_vae.pth'))
    elif args.dataset == 'FEMNIST':
        vae.load_state_dict(torch.load('vae/FEMNIST_vae.pth'))
    vae.eval()
    losses = []
    with torch.no_grad():
        for i in w_list:
            recon_data, mu, logvar = vae(i)
            mse_loss = loss_function(recon_data, i)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse_loss + kld_loss
            losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    for i in range(len(losses)):
        if losses[i] > mean_loss:
            malicious_clients.append(i)
    malicious_clients = torch.tensor(malicious_clients, device=args.device)
    # print(losses[9])
    # plt.figure(figsize=(30, 6))
    # plt.plot(losses, marker='')
    # highlight_index = 9
    # plt.scatter(highlight_index, losses[highlight_index], color='red', zorder=5)
    # # 确保横坐标完整显示
    # x_ticks = range(len(losses))  # 从 0 到 len(losses)-1
    # plt.xticks(x_ticks)
    # # 设置网格：仅显示竖线，覆盖完整范围
    # plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    # plt.show()
    intersection = torch.tensor([x for x in malicious_clients if x in true_malicious])       # 交集操作
    return intersection.tolist()


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


