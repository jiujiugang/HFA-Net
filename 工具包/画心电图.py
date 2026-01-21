import numpy as np
import matplotlib.pyplot as plt
import os

# 处理txt文件并保存图像
def process_and_save_images(data_dir, output_dir):
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)

            # 读取ECG信号数据
            with open(file_path, 'r') as f:
                signal = np.array([float(line.strip()) for line in f.readlines()])  # 读取每行数据，并转化为数组

            # 绘制信号图像
            plt.figure(figsize=(12, 6))
            plt.plot(signal, label='ECG Signal', alpha=0.6)
            plt.axis('off')  # 关闭坐标轴

            # 生成保存路径，保持原文件名并更改扩展名为.png
            output_image_path = os.path.join(output_dir, f"{filename.replace('.txt', '.png')}")

            # 保存图像
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
            plt.close()  # 关闭图像以便保存下一个

# 输入文件夹路径和输出文件夹路径
data_dir = r'D:\HTNet-master\NEW_MODEL\EDA_W_Denosied'  # 输入文件夹路径
output_dir = r'D:\HTNet-master\NEW_MODEL\EDA-TU'   # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 调用函数处理并保存图像
process_and_save_images(data_dir, output_dir)
