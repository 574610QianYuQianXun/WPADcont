import copy
import torch
from utils import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader


class Client():
    def __init__(self, _id, args, global_model, train_dataset=None, data_idxs=None):
        self.id = _id
        self.args = args
        self.global_model = global_model
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.local_model = None
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.n_data = len(self.normal_dataset)

    def local_train(self, loss_func, global_model, epoch, args, watermark_loa):

        self.local_model = copy.deepcopy(global_model)
        self.local_model.train()

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)

        for _ in range(self.args.local_ep):
            for _, (images, labels, _) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)

                outputs = self.local_model(inputs)
                loss = loss_func(outputs, labels)     # 计算损失函数

                loss.backward()                       # 反向传播
                optimizer.step()                      # 更新模型参数

        # with torch.no_grad():
            #     param = utils.model_to_vector(self.local_model, args).detach()

        return self.local_model, loss
