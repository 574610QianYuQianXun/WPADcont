import copy
import torch
from utils import utils
from detect import Detecting_disturbances
from attack import How_backdoor
from torch.utils.data import DataLoader
from Optimal_solution import Optimal_perturbation
import matplotlib.pyplot as plt


class Malicious_client():
    def __init__(self, _id, args, global_model, train_dataset=None, data_idxs=None):
        self.id = _id
        self.args = args
        self.global_model = global_model
        self.trans_model = None
        self.param_record = []
        self.watermark_loa = []
        self.watermark = False
        self.dist_factor = None
        self.dist_size = None
        self.anti_factor = None
        self.current_global_model = None
        self.last_param = None
        self.scaling_factor = 1        # 初始化缩放因子

        # 注入后门
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.backdoor_dataset = copy.deepcopy(self.normal_dataset)
        How_backdoor(self.backdoor_dataset)
        # 待训练模型初始化
        self.local_model = copy.deepcopy(global_model)
        self.benign_local_model = copy.deepcopy(global_model)

        self.train_loader = DataLoader(self.backdoor_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.benign_train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)

        self.n_data = len(self.normal_dataset)


    def local_train(self, loss_func, global_model, epoch, args, watermark_loa, win=6):
        self.local_model = copy.deepcopy(global_model)
        # 这里进行标记设计,先保存一部分的历史记录，用于选择标记
        if len(self.param_record) >= win:
            self.param_record.pop(0)
        self.param_record.append(utils.model_to_vector(self.global_model, args).detach())

        if epoch >= args.attack_epoch:
            # 这里是水印检测
            if self.watermark:
                if Detecting_disturbances(self.dist_factor, self.dist_size, self.global_model, self.current_global_model, self.watermark_loa):
                    print("攻击者没有被防御", self.id, self.scaling_factor)
                else:
                    print("攻击者已经被防御", self.id, self.scaling_factor)
                    self.scaling_factor -= 0.05
                    self.scaling_factor = max(self.scaling_factor, 1)

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
            # with torch.no_grad():
            #     # 这里我们还要获取一下哪些参数作为抗扰动参数
            #     diff = torch.abs(utils.model_to_vector(self.local_model, args).detach() - utils.model_to_vector(self.global_model, args).detach())
            #     _, robustness_mark = torch.topk(diff.view(-1), int(len(diff) * 0.1), largest=False)         # 1000
            #     values_list = torch.stack(list(watermark_loa.values()))
            #     watermark_value = utils.model_to_vector(self.local_model, args).detach()[values_list]       # 获得标记的原始值
            #     param = self.scaling_factor * utils.model_to_vector(self.local_model, args).detach()        # 得到缩放之后的恶意更新，对这个加水印
            #     param[values_list] = watermark_value                                                        # 还原标记之后的恶意更新，这是为了阈值计算

            # 这里进行扰动工作，从第3轮开始
            # if epoch >= 3 and args.watermark:
            #     # 水印位置
            #     self.watermark_loa = watermark_loa[self.id]
            #     # # 记录本轮全局模型，作为下一轮中的上一轮全局模型
            #     self.current_global_model = copy.deepcopy(utils.model_to_vector(self.global_model, args).detach())
            #     # 下面计算扰动和抗扰动的最优解，使用损失函数获得，损失函数值包括：隐蔽性（欧氏距离、余弦相似度）, 保留性（模拟多种聚合）
            #     # 返回最终扰动和抗扰动的幅度，或者直接返回模型参数
            #     self.dist_factor, self.anti_factor, self.dist_size, anti_size = Optimal_perturbation(self.param_record, param, self.watermark_loa, robustness_mark, self.current_global_model, self.local_model,
            #                          self.backdoor_dataset, loss_func, args)
            #
            #     param[self.watermark_loa] += self.dist_factor * self.dist_size     # 增加扰动
            #     for i in robustness_mark:                                 # 增加抗扰动
            #         param[i] += self.anti_factor * anti_size[i]
            #         param = param.detach()
            #     self.last_param = param
            #     self.watermark = True        # 水印标记为真，表示已经注入了水印
        else:

            self.local_model.train()
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
            for _ in range(self.args.local_ep):
                for _, (images, labels) in enumerate(self.benign_train_loader):
                    optimizer.zero_grad()
                    inputs = images.to(device=self.args.device, non_blocking=True)
                    labels = labels.to(device=self.args.device, non_blocking=True)
                    outputs = self.local_model(inputs)
                    loss = loss_func(outputs, labels)  # 计算损失函数
                    loss.backward()   # 反向传播
                    optimizer.step()  # 更新模型参数

            # with torch.no_grad():
            #     param = utils.model_to_vector(self.local_model, args).detach()

        return self.local_model, loss
