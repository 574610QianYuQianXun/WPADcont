import torch



# 相似性检测，余弦相似度
def Cos_mean(clients_param, attack_index = 9):
    target_tensor = clients_param[attack_index]
    cosine_similarities = []
    for key, tensor in clients_param.items():
        if key not in target_tensor:
            similarity = torch.nn.functional.cosine_similarity(target_tensor, tensor, dim=0)
            cosine_similarities.append(similarity.item())
    avg_cos =  sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0

    return avg_cos


# 相似性检测，欧氏距离
def Euclidean_mean(clients_param, attack_index = 9):
    target_tensor = clients_param[attack_index]
    euclidean_distances = []
    for key, tensor in clients_param.items():
        if key not in target_tensor:
            distance = torch.norm(target_tensor - tensor)  # 计算欧氏距离
            euclidean_distances.append(distance.item())
    avg_eud =  sum(euclidean_distances) / len(euclidean_distances) if euclidean_distances else 0.0

    return avg_eud