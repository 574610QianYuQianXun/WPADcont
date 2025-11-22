import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from attack import modify_labels
from clients.BaseClient import BaseClient
from torch.utils.data import DataLoader, random_split, Subset
import copy

from utils.utils import Inject_watermark_train


class Client(BaseClient):
    def __init__(self, client_id, params, global_model, train_dataset=None, data_idxs=None):
        super().__init__(client_id, params, global_model, train_dataset, data_idxs)
        # Inject_watermark_train(self.dataset)

        # # 从数据集中选一张图（比如第0张）
        # image = self.dataset.dataset.data[0]  # shape: [28, 28]
        # label = self.dataset.dataset.targets[0]
        #
        # # 如果是 Tensor，需要转换为 numpy 数组
        # if isinstance(image, torch.Tensor):
        #     image = image.numpy()
        #
        # # 可视化
        # plt.imshow(image, cmap='gray')
        # plt.title(f"Label: {label}")
        # plt.axis('off')
        # plt.show()


        self.train_loader = DataLoader(self.dataset, batch_size=self.params.local_bs, shuffle=True)
        # self.val_loader = DataLoader(self.dataset,batch_size=self.params.local_bs,shuffle=False)



        if self.params.agg== "FLShield":
            # 获取数据集的总大小
            # dataset_size = len(self.dataset)
            # val_size = int(0.2 * dataset_size)
            # train_size = dataset_size - val_size
            #
            # # 获取索引
            # indices = list(range(dataset_size))
            # train_indices, val_indices = indices[:train_size], indices[train_size:]
            # # 创建 Subset
            # val_dataset = Subset(self.dataset, val_indices)
            # # 创建 DataLoader
            # self.val_loader = DataLoader(val_dataset, batch_size=self.params.local_bs, shuffle=False)
            self.val_loader = DataLoader(self.dataset, batch_size=self.params.local_bs, shuffle=False)


    def local_train(self, loss_func,epoch,teacher_model=None,mask=None,pattern=None,delta_z=None,predicted_model=None):
        """
        Local training for benign client.
        """
        local_model = copy.deepcopy(self.global_model)
        local_model, last_loss = self.train_model(local_model, self.train_loader, loss_func,teacher_model=teacher_model,mask=mask,pattern=pattern,delta_z=delta_z,predicted_model=predicted_model)
        return local_model, last_loss