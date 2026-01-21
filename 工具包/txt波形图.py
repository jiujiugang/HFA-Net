import os
import matplotlib.pyplot as plt
import numpy as np

# 定义输入和输出文件夹
input_folder = r'D:\HTNet-master\NEW_MODEL\其他模块\01_01.txt'
output_folder = r'D:\HTNet-master\NEW_MODEL\其他模块\图像'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有txt文件
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        # 构建文件的完整路径
        file_path = os.path.join(input_folder, filename)

        # 读取数据
        with open(file_path, 'r') as file:
            data = [float(line.strip()) for line in file if line.strip()]

        # 创建时间轴（假设采样率为1）
        time_axis = np.arange(len(data))

        # 设置图形
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, data, linewidth=1.5, color='blue', alpha=0.8)

        # 添加标题和标签
        plt.title(f'Waveform from {filename}', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)

        # 添加网格
        plt.grid(True, alpha=0.3)

        # 设置坐标轴范围
        plt.xlim(0, len(data))
        plt.ylim(min(data) - 0.1, max(data) + 0.1)

        # 添加零线
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # 保存图像
        output_path = os.path.join(output_folder, f'{filename.replace(".txt", ".png")}')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f'已保存图像: {output_path}')

# 打印完成信息
print("所有文件已转换为图像并保存。")
