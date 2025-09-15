import copy
import torch
from utils import utils
import random
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from attack import How_backdoor
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        # self.embedding = nn.Embedding(input_dim, model_dim)               # 因为是纯数据，因此不需要词嵌入
        self.fc_in = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 分类器
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # 输入数据通过线性层映射到模型维度
        x = self.fc_in(x)

        # 调整输入形状为 (seq_len, batch_size, model_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)

        # 使用 TransformerEncoder 提取序列特征
        x = self.transformer_encoder(x)

        # 恢复形状为 (batch_size, seq_len, model_dim)
        x = x.permute(1, 0, 2)

        # 分类器输出类别 (batch_size, seq_len, output_dim)
        x = self.fc_out(x)

        return x


class Malicious_client():
    def __init__(self, _id, args, global_model, train_dataset=None, data_idxs=None):
        self.id = _id
        self.args = args
        self.global_model = global_model
        self.trans_model = None
        self.param_record = []
        self.mark = []
        self.task = False
        self.up = None
        self.down = None

        # 注入后门
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.backdoor_dataset = copy.deepcopy(self.normal_dataset)
        How_backdoor(self.backdoor_dataset)
        self.local_model = copy.deepcopy(global_model)
        self.benign_local_model = copy.deepcopy(global_model)
        self.train_loader = DataLoader(self.backdoor_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.benign_train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.n_data = len(self.normal_dataset)


    def local_train(self, loss_func, epoch, args, win=6):
        # 进行标记检测工作，如果上一轮进行了标记
        if self.task:
            next_param = parameters_to_vector(self.global_model.parameters()).detach()      # 这是要接受测试的数据
            test_data = utils.Extract_and_combine(next_param,self.mark)                     # 切片操作
            with torch.no_grad():
                test_data = test_data.unsqueeze(0).unsqueeze(-1)
                test_predict = self.trans_model(test_data)
                test_label = torch.argmax(test_predict, dim=-1)
                base_label = torch.zeros(len(next_param))
                base_label[self.up] = 1
                base_label[self.down] = 2
                slice_base_label = utils.Extract_and_combine(base_label, self.mark)         # 对基线标签进行切片操作
                acc = utils.Calculate_accuracy(test_label, slice_base_label)
                print(acc)
        print(epoch,":",parameters_to_vector(self.global_model.parameters()).detach()[7447])
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for _ in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                outputs = self.local_model(inputs)
                loss = loss_func(outputs, labels)     # 计算损失函数
                loss.backward()                       # 反向传播
                optimizer.step()                      # 更新模型参数
        with torch.no_grad():
            param = 2 * (parameters_to_vector(self.local_model.parameters()))
        # 这里进行标记设计
        if len(self.param_record) >= win:
            self.param_record.pop(0)
        self.param_record.append(parameters_to_vector(self.global_model.parameters()).detach())

        # 这里进行标记工作，从第2轮开始
        if epoch == 3:
            # 得到打标记的位置
            self.mark = utils.Find_mark_location(self.param_record, self.local_model, self.global_model)
            # 标记工作
            half_length = len(self.mark) // 2
            self.up = random.sample(self.mark, half_length)
            self.down = [x for x in self.mark if x not in self.up]
            param[self.up] += 1
            param[self.down] += 1
            print(self.mark)
        #     # 生成训练数据
        #     samples, labels = utils.Generate_train_data(self.param_record, param, self.mark, self.up, self.down, args)
        #     dataset = SequenceDataset(samples, labels)
        #     t_train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        #     # 训练Transformer
        #     self.trans_model = TransformerModel(1, 64, 4, 2, 3).to(args.device)
        #     criterion = nn.CrossEntropyLoss()  # 损失函数
        #     optimizer_trans = optim.Adam(self.trans_model.parameters(), lr=0.001)
        #     epochs = 100
        #     self.trans_model.train()
        #     for epoch in range(epochs):
        #         total_loss = 0
        #         for _, (batch_x, batch_y) in enumerate(t_train_loader):
        #             optimizer_trans.zero_grad()
        #             output = self.trans_model(batch_x)
        #             loss_tr = criterion(output.view(-1, 3), batch_y.view(-1))
        #             loss_tr.backward()
        #             optimizer_trans.step()
        #             total_loss += loss_tr.item()
        #         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(t_train_loader)}')
        #     # 已经完成了标记工作和transformer，进行工作标记，用来决定是否进行检测
        #     self.task = True

        return param, loss
