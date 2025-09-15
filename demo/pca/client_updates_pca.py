import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Optional

def visualize_client_updates_pca(clients_update: Dict[int, Dict[str, torch.Tensor]],
                                 backdoor_clients: Optional[List[int]] = None,
                                 title: str = "Client Updates PCA Visualization"):
    """
    对联邦学习中的客户端更新进行 PCA 降维并可视化。
    恶意客户端（backdoor_clients）以红色标记，其余客户端以黑色标记。
    坐标轴比例与范围一致，类似数学中的直角坐标系。
    """

    client_vectors = []
    client_ids = sorted(clients_update.keys())
    layer_name = 'fc2'
    for idx in client_ids:
        update = clients_update[idx]
        # vector = torch.cat([param.flatten() for param in update.values()]).detach().cpu().numpy()
        vector = torch.cat([param.flatten() for name, param in update.items()
                            if name.endswith(f"{layer_name}.bias") or name.endswith(f"{layer_name}.weight")]).cpu().numpy()

        client_vectors.append(vector)

    X = np.stack(client_vectors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # # 计算坐标范围
    # x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
    # y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
    # range_min = min(x_min, y_min)
    # range_max = max(x_max, y_max)
    #
    # plt.figure(figsize=(8, 8))  # 保证图是正方形

    for i, (x, y) in enumerate(X_pca):
        client_id = client_ids[i]
        is_backdoor = backdoor_clients and client_id in backdoor_clients
        if is_backdoor:
            plt.scatter(x, y, color='red', s=30, alpha=1.0)
        else:
            plt.scatter(x, y, color='black', s=20, alpha=0.4)

    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title(title)
    plt.grid(True)

    # plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴等比例
    # plt.xlim(range_min, range_max)
    # plt.ylim(range_min, range_max)

    plt.show()
