from attacks.attack import Attack
from utils.parameters import Params

class ReplaceAttack(Attack):
    def __init__(self, params: Params):
        super().__init__(params)
        self.target_client_indices = params.backdoor_clients  # 后门客户端列表

    # def perform_attack(self, helper) -> None:
    #     for idx in range(self.params.clients):
    #         if idx not in self.target_client_indices:
    #             continue  # 只处理后门客户端
    #
    #         client_state_dict = helper.clients_model[idx].state_dict()
    #         new_state_dict = {}
    #
    #         # 策略：idx偶数保留特征提取器，idx奇数保留分类器
    #         if idx % 2 == 0:
    #             # 保留后门模型的特征提取器，替换分类器
    #             for key in client_state_dict.keys():
    #                 if self.is_feature_extractor(key):
    #                     new_state_dict[key] = client_state_dict[key]  # 保留后门特征提取
    #                 else:
    #                     new_state_dict[key] = self.global_model_state_dict[key]  # 替换分类器
    #         else:
    #             # 保留后门模型的分类器，替换特征提取器
    #             for key in client_state_dict.keys():
    #                 if self.is_feature_extractor(key):
    #                     new_state_dict[key] = self.global_model_state_dict[key]  # 替换特征提取器
    #                 else:
    #                     new_state_dict[key] = client_state_dict[key]  # 保留后门分类器
    #
    #         # 加载新的参数
    #         helper.clients_model[idx].load_state_dict(new_state_dict)
    #
    # def is_feature_extractor(self, key: str) -> bool:
    #     """
    #     简单判断：根据key名字判断是特征提取器还是分类器
    #     """
    #     # 常见卷积、batchnorm、features等是特征提取器
    #     feature_keywords = ['conv', 'bn', 'features', 'layer']
    #     classifier_keywords = ['fc', 'fc2' , 'classifier', 'linear']
    #
    #     if any(fk in key.lower() for fk in feature_keywords):
    #         return True
    #     if any(ck in key.lower() for ck in classifier_keywords):
    #         return False
    #     # 默认当作特征提取器处理
    #     return True

    def perform_attack(self, idx, local_model, global_model_state_dict):
        """对训练后的 local_model 进行增强攻击操作"""
        if idx not in self.target_client_indices:
            return  # 不是后门客户端，不改

        client_state_dict = local_model.state_dict()
        new_state_dict = {}

        # 策略：idx 偶数保留特征提取器，替换分类器；idx 奇数保留分类器，替换特征提取器
        # if idx % 2 == 0:
        if False:
            # 保留特征提取器
            for key in client_state_dict.keys():
                if self.is_feature_extractor(key):
                    new_state_dict[key] = client_state_dict[key]
                else:
                    new_state_dict[key] = global_model_state_dict[key]
        else:
            # 保留分类器
            for key in client_state_dict.keys():
                if self.is_feature_extractor(key):
                    new_state_dict[key] = global_model_state_dict[key]
                else:
                    new_state_dict[key] = client_state_dict[key]

        # 加载新的参数到local_model中
        local_model.load_state_dict(new_state_dict)

    def is_feature_extractor(self, param_name):
        """判断当前参数名是不是属于特征提取器的"""
        # 针对MNIST（简单CNN）和CIFAR10（ResNet等）做区分
        feature_layers = ['conv1','conv2', 'bn', 'layer', 'features']  # 特征提取层常见命名
        classifier_layers = ['fc', 'fc2' , 'classifier', 'linear']  # 分类器层常见命名

        return (any(param_name.startswith(prefix) for prefix in feature_layers)
                and not any(param_name.startswith(prefix) for prefix in classifier_layers))
