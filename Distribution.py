import numpy as np


# 定义 Dirichlet 分布生成非独立同分布的数据分布
# def Generate_non_iid_indices(total_samples, num_clients, alpha):
#     proportions = np.random.dirichlet([alpha] * num_clients)
#     indices = np.arange(total_samples)
#     non_iid_indices = []
#     start = 0
#     for prop in proportions:
#         end = start + int(prop.sum() * total_samples)
#         sampled_indices = np.random.choice(indices[start:end], end - start, replace=False)
#         non_iid_indices.append(sampled_indices)
#         start = end
#     return non_iid_indices


def Generate_non_iid_indices(total_samples, num_clients, alpha):
    proportions = np.random.dirichlet([alpha] * num_clients)
    indices = np.arange(total_samples)

    non_iid_indices = []

    start = 0
    for i, prop in enumerate(proportions):
        if i == num_clients - 1:
            end = total_samples
        else:
            end = start + max(int(prop * total_samples), 1)
        sampled_indices = np.random.choice(indices[start:end], end - start, replace=False)
        non_iid_indices.append(sampled_indices)
        start = end

    remaining_indices = indices[start:]
    for i, idx in enumerate(remaining_indices):
        non_iid_indices[i % num_clients] = np.concatenate((non_iid_indices[i % num_clients], [idx]))

    return non_iid_indices



def NO_iid(train_set, num_clients, a):
    # 设置客户端数量和非独立同分布的数据分布
    total_samples = len(train_set)
    non_iid_indices = Generate_non_iid_indices(total_samples, num_clients, a)
    # 创建客户端数据字典
    client_data_indices = {}
    for i in range(num_clients):
        client_data_indices[i] = non_iid_indices[i]

    return client_data_indices                          # 返回客户端字典数据


