# 目的：训练VAE模型
# 首先使用MNIST测试集训练CNN模型，保留CNN模型参数
# 用CNN模型参数训练VAE模型
from utils.Tool import FEMNIST
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from model.models import CNNMnist, LightweightResNet18, CNNFEMnist
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import pathlib
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader




# VAE模型
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Get_model_param(args, file_path):
    setup_seed(20240901)# 固定随机种子
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("\nArguments: %s " % args_str)                                            # 打印参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_set = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
    elif args.dataset == 'CIFAR10':
        Data_path = '../../FedNed-master/FedNed-master/data/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
    elif args.dataset == 'FEMNIST':
        Data_path = '../dataset/FEMNIST'
        if not os.path.exists(Data_path) or len(os.listdir(Data_path)) == 0:
            print("The FEMNIST dataset does not exist, please download it")
        test_set = FEMNIST(train=False)

    train_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    if args.model == 'CNNMnist':
        model = CNNMnist(args=args).to(args.device)
    if args.model == 'LightweightResNet18':
        model = LightweightResNet18(args=args).to(args.device)
    if args.model == 'CNNFEMnist':
        model = CNNFEMnist(args=args).to(args.device)

    print("\nStart training......\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)
    model.train()
    param_list = []
    for epoch in range(args.epochs):
        print(epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        param = model.state_dict()
        # param = torch.cat([p.view(-1) for p in param.values()]).tolist()
        param = parameters_to_vector(model.parameters()).detach()
        param_list.append(param)
    param_tensor = torch.stack(param_list)
    # param_tensor = torch.tensor(param_tensor).to(args.device)
    print('Finished Training, Get param')
    torch.save(param_tensor, file_path)
    return True

# VAE定义损失函数
loss_function = nn.MSELoss(reduction='sum')


def Train_VAE(args, file_path, model_path):
    setup_seed(20240901)
    print("train VAE")
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')  # GPU
    data = torch.load(file_path)
    batch_size = 1  # 设置批次大小
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    input_dim = len(data[0])
    latent_dim = int(0.1 * input_dim)            # mnist 0.1       # cifar10 0.000001
    vae = VAE(input_dim, latent_dim).to(args.device)
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)         #mnist 0.0001
    epoch = 100
    for epoch in range(epoch):
        vae.train()
        optimizer.zero_grad()
        for batch_data in data_loader:
            batch_data = batch_data[0].to(args.device)
            recon_batch, mu, logvar = vae(batch_data)
            mse_loss = loss_function(recon_batch, batch_data)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse_loss + kld_loss
            loss.backward()
            optimizer.step()
        print('Epoch: {} Loss: {:.4f}'.format(epoch, loss.item()))
    torch.save(vae.state_dict(), model_path)


if __name__ == '__main__':
    # 如果不存在vae文件夹，则创建
    if not os.path.exists('../vae'):
        os.makedirs('../vae')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='MNIST', help="name of dataset: MNIST, FEMNIST, CIFAR10")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--model', type=str, default='CNNMnist', help='LightweightResNet18, CNNMnist, CNNFEMnist')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")     # CIFAR10是彩色图像，通道3
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    args = parser.parse_args()
    param_path = 'vae/' + args.dataset +'_param.pth'
    model_path = 'vae/' + args.dataset +'_vae.pth'
    if os.path.exists(param_path):
        os.remove(param_path)
    if os.path.exists(model_path):
        os.remove(model_path)
    with open(param_path, 'w') as f:
        pass
    with open(model_path, 'w') as f:
        pass

    Get_model_param(args, param_path)
    Train_VAE(args, param_path, model_path)

