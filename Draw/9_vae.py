import matplotlib.pyplot as plt

# 数据
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
y1 = [50.21,50.21,192.30,203.25,209.12,216.75,192.37,226.29,228.81,242.14,189.87,252.25,264.21,264.11,276.44,282.48,287.35,292.29,303.53,309.21,314.75,314.69,324.28,273.62,338.35,346.07,356.56,362.50,371.41,380.27]
y2 = [50.21,50.21,191.05,202.54,208.38,216.24,191.69,225.55,228.01,241.33,189.19,251.50,263.41,264.21,275.61,281.53,286.42,291.37,302.63,308.34,313.79,313.70,323.10,272.72,337.45,345.08,355.41,361.54,370.37,379.19]
print(len(y1))
print(len(y2))
# 创建图形
plt.figure(figsize=(8, 5))

# 绘制两条线
plt.plot(x, y1, label="Line A (Watermark)", linestyle='-', marker='o', color='blue')  # 曲线A
plt.plot(x, y2, label="Line B (No Watermark)", linestyle='--', marker='s', color='orange')  # 曲线B

# 图例、标签和标题
plt.title("Comparison of Two Curves", fontsize=14)
plt.xlabel("Train_rounds", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(fontsize=10)

# 网格和布局
plt.grid(alpha=0.3)
plt.tight_layout()

# 显示图形
plt.show()
