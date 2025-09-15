import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

# 数据
asr = [0.00, 0.03, 0.10, 0.18, 0.24, 0.28, 0.36, 0.36, 0.42, 0.43, 0.45, 0.47,
       0.50, 0.51, 0.52, 0.15, 0.13, 0.13, 0.11, 0.11, 0.11, 0.10, 0.10, 0.10,
       0.10, 0.09, 0.09, 0.09, 0.09, 0.08]
highlight_index = 14

# 绘图
plt.figure(figsize=(9, 6))
plt.plot(asr, marker='', label='ASR')

# 确保横坐标从 1 开始
x_ticks = range(1, len(asr) + 1)  # 横坐标从 1 开始
plt.xticks(range(len(asr)), x_ticks)  # 设置横坐标标签为从 1 开始

plt.xlabel('Train Round', fontsize=20)  # 横坐标标签字体大小
plt.ylabel('ASR', fontsize=20)          # 纵坐标标签字体大小

# 设置网格：仅显示竖线，覆盖完整范围
# plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

# 高亮某一点
plt.scatter(highlight_index, asr[highlight_index], color='red', zorder=5)

# 去掉白边
plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.1)

# 显示图例
plt.legend(fontsize=20)

# 显示图表
plt.show()
