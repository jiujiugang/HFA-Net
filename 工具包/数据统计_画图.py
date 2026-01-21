import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 假设文件存放在一个目录下，这里给定路径
data_dir =r'D:\HTNet-master\NEW_MODEL\EDA\EDA_corp\1_1.txt'

# 用来存放每个信号的长度
signal_lengths = []

# 遍历文件夹中的所有txt文件
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)

        # 读取ECG信号数据
        with open(file_path, 'r') as f:
            signal = np.array([float(line.strip()) for line in f.readlines()])
            signal_lengths.append(len(signal))

# 统计信号长度
max_length = np.max(signal_lengths)
min_length = np.min(signal_lengths)
median_length = np.median(signal_lengths)

# 使用mode()正确获取众数
mode_result = stats.mode(signal_lengths)

# 确保正确获取众数，处理返回结果
mode_length = mode_result.mode.flatten()[0]  # flatten确保将其变为数组后取第一个元素

# 输出统计结果
print(f"最长信号的长度: {max_length}")
print(f"最短信号的长度: {min_length}")
print(f"信号长度的中位数: {median_length}")
print(f"信号长度的众数: {mode_length}")

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(signal_lengths, bins=10, edgecolor='black', alpha=0.7)
plt.axvline(max_length, color='r', linestyle='dashed', linewidth=2, label=f'Max length: {max_length}')
plt.axvline(min_length, color='g', linestyle='dashed', linewidth=2, label=f'Min length: {min_length}')
plt.axvline(median_length, color='b', linestyle='dashed', linewidth=2, label=f'Median length: {median_length}')
plt.axvline(mode_length, color='orange', linestyle='dashed', linewidth=2, label=f'Mode length: {mode_length}')
plt.title("ECG Signal Length Distribution")
plt.xlabel("Signal Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()
