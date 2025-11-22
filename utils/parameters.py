from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch


@dataclass
class Params:
    #任务名称
    task: str = 'MNIST'

    #当前时间
    current_time: str = field(default_factory=lambda: datetime.now().strftime('%b.%d_%H.%M.%S'))
    #显卡
    gpu: int = 0
    #设备
    device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #数据集
    dataset: str = 'MNIST'
    #分类任务的的总类别
    num_classes: int = 10

    #是否使用预训练模型
    model_is_pretrained: bool = False
    #模型选择
    model: str = 'CNNMnist'
    #恶意模型的比例
    malicious: float = 0.0
    #攻击开始轮次
    attack_epoch: float = 3.0
    #后门的触发器植入方式
    poison_type: int = 1
    #是否添加水印
    watermark: bool = False
    #是否开启防御
    defense: str = 'no'
    #攻击类型选择
    attack_type: str = 'How_backdoor'
    #聚合方式选择
    agg: str = 'FedAvg'
    #flshield的防御设置
    flshield_mode: str = 'bijective'

    #数据iid系数
    a: float = 0.5
    #客户端总数
    clients: int = 100
    #训练总轮次
    epochs: int = 100
    #本地的批处理大小
    local_bs: int = 64

    #批处理大小
    bs: int = 128
    #学习率
    lr: float = 1e-3
    #SGD的参数
    momentum: float = 0.5
    #客户端训练轮次
    local_ep: int = 1

    #是否保存模型
    save_model: bool = True
    #保存模型位置
    folder_path: str = 'trained_model'
    is_back_model:bool = False

    #后门反转目标
    origin_target: int = 1
    #后门改完后的标签
    aim_target: int = 8
    test_target: int = 5
    #防御策略
    purification_strategy: str = 'feature_unlearning'  # 强制使用特征解毒
    pur_lr: float = 0.001  # 解毒学习率
    pur_kd_lr: float = 1e-5  # 解毒蒸馏学习率
    pur_w: float = 0.275


    #恶意客户端
    backdoor_clients: list = None

    negative_distillation: bool = False  # 是否启用负蒸馏
    negative_lambda: float = 1.0  # 负蒸馏强度超参数
    negative_distill_epochs: int = 5  # 负蒸馏训练轮数
    negative_lr: float = 0.001  # 负蒸馏学习率
    negative_optimizer: str = "adam"  # 负蒸馏使用的优化器 (可选 adam / sgd)
    temperature: float = 2.0  # 蒸馏温度

    #转为字典
    def to_dict(self):
        return asdict(self)
