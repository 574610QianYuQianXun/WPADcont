from torch.nn.utils import parameters_to_vector


# 检测函数
def Detecting_disturbances(dist_factor, dist_size, global_model, pre_global_nodel, mark):
    value = dist_size * dist_factor / 100
    next_param = parameters_to_vector(global_model.parameters()).detach() - pre_global_nodel
    # a = next_param[mark]
    # b = parameters_to_vector(global_model.parameters()).detach()[mark]
    # c = pre_global_nodel[mark]
    if next_param[mark] >= value - 0.001:
        return True
    else:
        return False


