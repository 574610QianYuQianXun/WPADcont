import numpy as np
from sklearn.cluster import KMeans


# 计算分类
def Kmeans_clustering(tensor_data, n_clusters=2, random_state=42, max_iter=100):
    tensor_data = tensor_data.cpu()
    """
    对输入的张量数据进行 K-Means 聚类，并返回每个类别的下标。
    参数:
    tensor_data (torch.Tensor): 输入的 PyTorch 张量数据
    n_clusters (int): 聚类的数量，默认为 2
    random_state (int): 随机种子，默认为 42
    max_iter (int): 最大迭代次数，默认为 300

    返回:
    tuple: 两个元素的元组，第一个为类别 0 的下标，第二个为类别 1 的下标
    """
    # 将 PyTorch 张量转换为 NumPy 数组，KMeans需要NumPy数组输入
    data = tensor_data.reshape(-1, 1)  # 将一维数据转换为二维数据以符合KMeans的输入要求

    # 定义KMeans聚类模型，控制最大迭代次数
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)

    # 对数据进行聚类
    kmeans.fit(data)

    # 获取聚类结果
    labels = kmeans.labels_  # 每个数据点的聚类标签

    # 获取每个类别的下标
    class_0_indices = np.where(labels == 0)[0]  # 类别 0 的下标
    class_1_indices = np.where(labels == 1)[0]  # 类别 1 的下标

    return class_0_indices, class_1_indices  # 分开返回两个类别的下标

