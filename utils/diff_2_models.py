import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct


# -------------------------------
# 工具函数：模型参数展平
# -------------------------------
def flatten_model_params(model):
    """
    遍历模型，将每一层的参数展平为一维向量，并返回列表和对应层名称。
    """
    flat_params = []
    layer_names = []
    for name, param in model.named_parameters():
        flat = param.view(-1).cpu().detach().numpy()
        flat_params.append(flat)
        layer_names.append(name)
    return flat_params, layer_names


def pad_differences(diff_list):
    """
    将每一层的一维向量填充到相同长度，构成一个二维数组，
    使用 np.nan 填充不足部分，便于热力图绘制。
    """
    max_len = max(diff.shape[0] for diff in diff_list)
    padded = [np.pad(diff, (0, max_len - diff.shape[0]), mode='constant', constant_values=np.nan)
              for diff in diff_list]
    return np.vstack(padded)


# -------------------------------
# 变换函数：FFT 与 DCT
# -------------------------------
def compute_fft(params_list):
    """
    对列表中每一层展平的参数进行一维傅里叶变换，
    返回每一层的频域表示（幅值）。
    """
    fft_list = [np.abs(np.fft.fft(params)) for params in params_list]
    return fft_list


def compute_dct(params_list, norm='ortho'):
    """
    对列表中每一层展平的参数进行离散余弦变换（DCT），
    返回每一层的频域表示（幅值），参数 norm 设置为 'ortho' 保证正交性。
    """
    dct_list = [np.abs(dct(params, norm=norm)) for params in params_list]
    return dct_list


# -------------------------------
# 差异计算函数（在变换域上比较两个模型的差异）
# -------------------------------
def compute_transformed_diff(transform_func, model1_params, model2_params, metric="cosine", **kwargs):
    """
    对传入的 transform_func（如 compute_fft 或 compute_dct）分别对两个模型的参数进行变换，
    然后计算对应层变换结果之间的差异。

    参数:
      transform_func: 变换函数，例如 compute_fft 或 compute_dct
      model1_params: 模型1展平后的参数列表
      model2_params: 模型2展平后的参数列表
      metric: 指定计算差异的方式，取值可以为：
              "elementwise"：逐元素绝对差值（默认），输出与输入向量相同维度的差异向量；
              "euclidean"：欧氏距离，每层输出一个标量；
              "cosine"：余弦距离，每层输出一个标量（计算公式为 1 - cosine similarity）。
      **kwargs: 传递给 transform_func 的额外参数（例如 norm='ortho'）。

    返回:
      diff_list: 每一层变换后结果的差异列表。
                 若 metric=="elementwise"，每个元素为一维数组；
                 若 metric=="euclidean" 或 "cosine"，每个元素为单个标量（包装为一维数组，以便后续处理）。
    """
    trans1 = transform_func(model1_params, **kwargs)
    trans2 = transform_func(model2_params, **kwargs)
    diff_list = []

    for arr1, arr2 in zip(trans1, trans2):
        if metric == "elementwise":
            diff = np.abs(arr1 - arr2)
        elif metric == "euclidean":
            diff = np.linalg.norm(arr1 - arr2)
            diff = np.array([diff])  # 将标量包装为一维数组，便于后续处理
        elif metric == "cosine":
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            # 若任意向量的范数为 0，直接认为差异为最大值 1
            if norm1 == 0 or norm2 == 0:
                diff = 1.0
            else:
                cosine_similarity = np.dot(arr1, arr2) / (norm1 * norm2)
                diff = 1 - cosine_similarity
            diff = np.array([diff])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        diff_list.append(diff)

    return diff_list


# -------------------------------
# 通用绘图函数
# -------------------------------
def plot_heatmap(data_2d, layer_names, xlabel, ylabel, title, colorbar_label):
    """
    绘制二维热力图，数据中的 np.nan 部分不显示。
    """
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(data_2d, cmap='coolwarm', mask=np.isnan(data_2d),
                     cbar_kws={'label': colorbar_label})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(len(layer_names)) + 0.5)
    ax.set_yticklabels(layer_names, rotation=0)
    plt.title(title)
    plt.show()


# -------------------------------
# 主流程类：模型变换域差异分析器
# -------------------------------
class ModelTransformDiffAnalyzer:
    def __init__(self, model1, model2):
        # 分别获取两个模型展平后的参数及层名称（假设模型结构一致）
        self.params1, self.layer_names = flatten_model_params(model1)
        self.params2, _ = flatten_model_params(model2)

    def plot_spatial_diff(self):
        """
        在空间域直接比较模型参数差异热力图
        """
        # 计算绝对差异（直接在原始参数上比较）
        diff_list = [np.abs(p1 - p2) for p1, p2 in zip(self.params1, self.params2)]
        data_2d = pad_differences(diff_list)
        plot_heatmap(data_2d, self.layer_names,
                     xlabel='Flattened Parameter Index',
                     ylabel='Layer',
                     title='Model Parameter Differences (Spatial Domain)',
                     colorbar_label='Absolute Difference')

    def plot_fft_diff(self):
        """
        分别对两个模型进行 FFT 变换，再比较变换后的差异。
        """
        fft_diff_list = compute_transformed_diff(compute_fft, self.params1, self.params2)
        data_2d = pad_differences(fft_diff_list)
        plot_heatmap(data_2d, self.layer_names,
                     xlabel='Frequency Component Index',
                     ylabel='Layer',
                     title='Model Parameter Differences (FFT Domain)',
                     colorbar_label='FFT Absolute Difference')

    def plot_dct_diff(self):
        """
        分别对两个模型进行 DCT 变换，再比较变换后的差异。
        """
        dct_diff_list = compute_transformed_diff(compute_dct, self.params1, self.params2, norm='ortho')
        data_2d = pad_differences(dct_diff_list)
        plot_heatmap(data_2d, self.layer_names,
                     xlabel='DCT Coefficient Index',
                     ylabel='Layer',
                     title='Model Parameter Differences (DCT Domain)',
                     colorbar_label='DCT Absolute Difference')

# -------------------------------
# 使用示例（假设 benign_model 和 backdoor_model 已定义）
# -------------------------------
# analyzer = ModelTransformDiffAnalyzer(benign_model, backdoor_model)
# analyzer.plot_spatial_diff()
# analyzer.plot_fft_diff()
# analyzer.plot_dct_diff()
