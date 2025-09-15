import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# 定义数据
data = np.array([
    [0.01834, 0.00128, np.nan, -0.00256, -0.03101, -0.01805, -0.00370, 0.06242, np.nan, 0.02241],
    [0.03279, 0.01713, 0.02910, 0.00536, -0.04278, -0.03011, 0.00399, 0.02168, -0.00095, 0.00584],
    [0.01788, 0.02530, 0.03313, 0.00344, -0.03084, np.nan, -0.01019, 0.09091, 0.00178, 0.00221],  # 需要标红
    [0.02119, 0.00919, 0.04474, 0.01140, -0.03417, -0.01381, 0.00177, 0.03464, -0.00318, 0.01057],
    [0.03418, 0.01152, 0.04456, -0.00504, -0.03632, -0.02530, 0.00024, 0.02533, -0.00452, 0.01498],
    [0.11172, 0.00100, 0.16078, 0.00861, -0.00715, 0.03698, 0.01015, 0.00505, -0.00410, 0.00668],
    [0.06823, 0.03887, 0.06162, -0.00782, -0.02736, np.nan, -0.05770, 0.09148, -0.02656, 0.00152],  # 需要标红
    [0.02319, 0.02246, 0.04946, 0.00111, -0.02682, -0.03063, 0.00732, 0.01796, -0.00270, 0.00942],
    [0.01895, 0.01318, 0.05241, 0.00366, -0.03219, -0.02332, -0.00406, 0.02887, -0.00164, -0.00027],
    [0.02002, 0.01732, 0.01658, -0.00591, -0.03832, -0.02331, -0.00025, 0.02325, -0.01248, 0.00722]
])

# 处理 NaN 值，使用列均值填充
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data)

# PCA 降维到 2D
pca_2d = PCA(n_components=2)
data_pca_2d = pca_2d.fit_transform(data_imputed)

# PCA 降维到 1D
pca_1d = PCA(n_components=1)
data_pca_1d = pca_1d.fit_transform(data_imputed)

# 画布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2D 散点图
axes[0].set_title('PCA - 2D Projection')
for i in range(len(data_pca_2d)):
    color = 'red' if i == 2 or i == 6 else 'green'  # 第 3 行和第 7 行（索引 2 和 6）标红
    axes[0].scatter(data_pca_2d[i, 0], data_pca_2d[i, 1], color=color, alpha=0.7)
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].grid()

# 1D 投影（折线图）
axes[1].set_title('PCA - 1D Projection')
for i in range(len(data_pca_1d)):
    color = 'red' if i == 2 or i == 6 else 'green'
    axes[1].scatter(i, data_pca_1d[i, 0], color=color, alpha=0.7)
axes[1].plot(range(len(data_pca_1d)), data_pca_1d, linestyle='dashed', color='gray', alpha=0.5)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Principal Component 1 Value')
axes[1].grid()

# 显示
plt.tight_layout()
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.covariance import EllipticEnvelope
# from sklearn.decomposition import PCA
# from sklearn.experimental import enable_iterative_imputer  # 无此行报错
# from sklearn.impute import IterativeImputer, SimpleImputer
#
# # 你的原始数据
# data = np.array([
#     [0.01834, 0.00128, np.nan, -0.00256, -0.03101, -0.01805, -0.00370, 0.06242, np.nan, 0.02241],
#     [0.03279, 0.01713, 0.02910, 0.00536, -0.04278, -0.03011, 0.00399, 0.02168, -0.00095, 0.00584],
#     [0.01788, 0.02530, 0.03313, 0.00344, -0.03084, np.nan, -0.01019, 0.09091, 0.00178, 0.00221],
#     [0.02119, 0.00919, 0.04474, 0.01140, -0.03417, -0.01381, 0.00177, 0.03464, -0.00318, 0.01057],
#     [0.03418, 0.01152, 0.04456, -0.00504, -0.03632, -0.02530, 0.00024, 0.02533, -0.00452, 0.01498],
#     [0.11172, 0.00100, 0.16078, 0.00861, -0.00715, 0.03698, 0.01015, 0.00505, -0.00410, 0.00668],
#     [0.06823, 0.03887, 0.06162, -0.00782, -0.02736, np.nan, -0.05770, 0.09148, -0.02656, 0.00152],
#     [0.02319, 0.02246, 0.04946, 0.00111, -0.02682, -0.03063, 0.00732, 0.01796, -0.00270, 0.00942],
#     [0.01895, 0.01318, 0.05241, 0.00366, -0.03219, -0.02332, -0.00406, 0.02887, -0.00164, -0.00027],
#     [0.02002, 0.01732, 0.01658, -0.00591, -0.03832, -0.02331, -0.00025, 0.02325, -0.01248, 0.00722]
# ])
#
# # 1. 先用均值填充，避免IterativeImputer报错
# simple_imp = SimpleImputer(strategy="mean")
# data_filled = simple_imp.fit_transform(data)
#
# # # 1. 填充缺失值
# # imp = IterativeImputer()
# # filled_data = imp.fit_transform(data_filled)
#
# # 2. 使用 EllipticEnvelope 进行异常值检测
# detector = EllipticEnvelope(contamination=0.1)  # 设定污染率 10%
# inliers = detector.fit_predict(data_filled)  # 预测异常值（1：正常，-1：异常）
#
# # 3. 进行 PCA 降维
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(data_filled)
#
# # 4. 绘制 PCA 投影图，区分异常值和正常值
# plt.figure(figsize=(6, 5))
# plt.title("PCA Projection with Outlier Detection")
#
# for i in range(len(reduced_data)):
#     color = 'red' if inliers[i] == -1 else 'green'
#     plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=color)
#
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.grid(True)
# plt.show()
