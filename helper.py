import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from attack import frequency_backdoor
from model import models
from clients.Client import Client
from clients.MaliciousClient import MaliciousClient
from utils import utils
from utils.ModelSaver import ModelSaver
from utils.parameters import Params
from defenses.fedavg import FedAvg as Defense
from utils.utils import Inject_trigger, Inject_watermark_test


logger = logging.getLogger('logger')
class Helper:
    params:Params=None
    defence:Defense=None

    def __init__(self,params):
        self.teacher_model = None
        self.params = Params(**params)
        self.client_loss = []
        self.malicious_clients = []
        self.clients = []
        self.global_model = None
        self.dict_users = None
        self.train_dataset = None
        self.test_dataset = None
        self.clients_model = {} # 存储各客户端本地训练后的模型参数
        self.clients_his_update={} # 存储各个客户端模型的历史参数更新量
        self.clients_update={} # 存储各个客户端模型的参数更新量

        self.model_saver = ModelSaver(self.params)# 模型保存
        self.loss_func = None

        self.mask=None # 用于逆向攻击的掩码
        self.pattern=None # 用于逆向攻击的模式
        self.delta_z=None # 用于逆向攻击的特征触发器

        # 新增：用于目标标签推断的平滑分数记录
        # 结构: {class_id: smoothed_score}
        self.target_label_inference_scores = {c: 0.0 for c in range(self.params.num_classes)}
        # 新增：平滑因子
        self.inference_smoothing_alpha = 0.4

        self.setup_seed(20250703)

        self.make_device()
        self.make_dataset()
        self.make_model()
        self.make_clients()
        self.make_loss()
        self.make_defense()
        self.make_attack()

    def make_device(self):
        self.params.device = torch.device(
            'cuda:{}'.format(self.params.gpu) if torch.cuda.is_available() and self.params.gpu != -1 else 'cpu'
        )
    def make_dataset(self):
        # 初始化数据集
        # 训练数据集  测试数据集  每个客户端对应的数据样本
        self.train_dataset, self.test_dataset, self.dict_users = utils.Download_data(self.params.dataset, 'dataset', self.params)
        # 如果攻击类型需要对测试集进行后门处理，则执行处理,加入触发器
        # Inject_watermark_test(self.test_dataset)
        if self.params.attack_type=='dct':
            # utils.Backdoor_process(self.test_dataset,self.params.origin_target,self.params.aim_target)
            # 对测试集所有样本植入相同触发器（保持原标签）
            frequency_backdoor(
                train_set=self.test_dataset,
                origin_target=self.params.origin_target,
                modify_label=False,  # 关键区别
                # strength=0.2
                # trigger_pattern=[0.5, -0.5]  # 可调整模式
            )

        # if self.params.poison_type==1 and self.params.attack_type in ['How_backdoor','dba']:
        #     utils.Backdoor_process(self.test_dataset,self.params.origin_target,self.params.aim_target)

            # # 从数据集中选一张图（比如第0张）
            # image = self.test_dataset.data[0]  # shape: [28, 28]
            # label = self.test_dataset.targets[0]
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

            # utils.Backdoor_process(self.test_dataset,self.params.test_target,self.params.aim_target)

    def make_model(self):
        # 初始化全局模型，并将模型放到对应设备上
        self.global_model = models.get_model(self.params).to(self.params.device)
        if self.params.model_is_pretrained:
            # 构造完整的文件名
            model_name = f"{self.params.dataset}_{self.params.model}"
            if self.params.is_back_model:
                model_name += "_backdoor"
            model_name += "_best.pth"

            # 构造完整路径
            model_path = os.path.join(self.params.folder_path, model_name)
            # 加载模型字典
            checkpoint = torch.load(model_path, map_location=self.params.device)
            # 从字典中提取 state_dict
            self.global_model.load_state_dict(checkpoint['state_dict'])
            # 如果需要其他信息（例如 epoch 或 val_loss），也可以加载
            epoch = checkpoint['epoch']
            val_loss = checkpoint['val_loss']
            # 打印加载的信息（可选）
            logger.info(f"Loaded best model from epoch {epoch} with val_loss {val_loss:.6f}")
            # self.global_model.load_state_dict(torch.load(f"{self.params.dataset}_{self.params.model}_best.pth", map_location=self.params.device))

        self.teacher_model=deepcopy(self.global_model)

    def make_clients(self):
        # 初始化客户端
        # 1.根据数据分布选择恶意客户端
        self.malicious_clients = utils.Choice_mali_clients(self.dict_users, self.train_dataset, self.params)
        print("恶意客户端：", sorted(self.malicious_clients))
        self.params.backdoor_clients=sorted(self.malicious_clients)

        # 2.根据客户端数目，创建正常客户端和恶意客户端
        for _id in range(self.params.clients):
            if _id in self.malicious_clients and self.params.attack_type in ['How_backdoor','dct','dba']:
                self.clients.append(MaliciousClient(_id, self.params, self.global_model, self.train_dataset, self.dict_users[_id]))
            else:
                self.clients.append(Client(_id, self.params, self.global_model, self.train_dataset, self.dict_users[_id]))
    def make_loss(self):
        self.loss_func = nn.CrossEntropyLoss().to(self.params.device)

    def make_defense(self):
        return

    def make_attack(self):
        return

    # def save_model(self,model, save_path):
    #     """
    #     保存 PyTorch 模型的状态字典 (state_dict) 到指定路径。
    #     :param model: 需要保存的 PyTorch 模型
    #     :param save_path: 保存文件的路径（包含文件名，如 'models/global_model.pth'）
    #     """
    #     # 确保目录存在
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     # 保存模型
    #     torch.save(model.state_dict(), save_path)
    #     print(f"模型已保存到: {save_path}")

    @staticmethod
    def setup_seed(seed=2025):
        """
        设置随机种子，保证实验可重复性
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True