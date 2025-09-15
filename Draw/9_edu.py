import matplotlib.pyplot as plt

# 数据
x = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
y1 = [7.17,7.24,7.33,7.44,7.56,7.69,7.83,7.98,8.10,8.22,8.31,8.40,8.48,8.56,8.64,8.73,8.81,8.89,8.98,9.06,9.15,9.24,9.32,9.42,9.51,9.60,9.69,9.79]  # 标记举例
y2 = [7.12,7.21,7.30,7.41,7.53,7.66,7.81,7.94,8.07,8.18,8.28,8.37,8.45,8.53,8.61,8.69,8.78,8.86,8.96,9.03,9.12,9.21,9.30,9.39,9.48,9.58,9.67,9.76]   # 未标记距离

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制两条线
plt.plot(x, y1, label="Line A (Watermark)", linestyle='-', marker='o', color='blue')  # 曲线A
plt.plot(x, y2, label="Line B (No Watermark)", linestyle='--', marker='s', color='orange')  # 曲线B

# 图例、标签和标题
plt.title("Comparison of Two Curves", fontsize=14)
plt.xlabel("Train_rounds", fontsize=12)
plt.ylabel("Euclidean distance", fontsize=12)
plt.legend(fontsize=10)

# 网格和布局
plt.grid(alpha=0.3)
plt.tight_layout()

# 显示图形
plt.show()
