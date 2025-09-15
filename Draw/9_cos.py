import matplotlib.pyplot as plt

# 数据
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
y1 = [0.999,0.999,0.998,0.998,0.998,0.998,0.997,0.997,0.997,0.996,0.996,0.996,0.997,0.997,0.997,0.998,0.998,0.998,0.999,0.998,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999]  # 标记余弦
y2 = [0.999,0.999,0.999,0.998,0.998,0.998,0.998,0.997,0.997,0.997,0.997,0.996,0.997,0.998,0.998,0.998,0.998,0.998,0.998,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999]   # 未标记余弦

# 创建图形
plt.figure(figsize=(8, 5))

# 绘制两条线
plt.plot(x, y1, label="Line A (Watermark)", linestyle='-', marker='o', color='blue')  # 曲线A
plt.plot(x, y2, label="Line B (No Watermark)", linestyle='--', marker='s', color='orange')  # 曲线B

# 图例、标签和标题
plt.title("Comparison of Two Curves", fontsize=14)
plt.xlabel("Train_rounds", fontsize=12)
plt.ylabel("Cosine similarity", fontsize=12)
plt.legend(fontsize=10)

# 网格和布局
plt.grid(alpha=0.3)
plt.tight_layout()

# 显示图形
plt.show()
