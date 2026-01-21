import numpy as np
import matplotlib.pyplot as plt
import os

# 处理txt文件并保存图像
def process_and_save_images(data_dir, output_dir, image_size=224):
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)

            # 读取ECG信号数据
            with open(file_path, 'r') as f:
                signal = np.array([float(line.strip()) for line in f.readlines()])  # 读取每行数据，并转化为数组

            # 创建一个224x224的图像
            fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=130)  # 设置为224x224图像，dpi=100
            ax.plot(signal, label='ECG Signal', alpha=0.6)
            ax.axis('off')  # 关闭坐标轴

            # 生成保存路径，保持原文件名并更改扩展名为.png
            output_image_path = os.path.join(output_dir, f"{filename.replace('.txt', '.png')}")

            # 保存图像
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0,transparent=False, format='png', dpi=130)
            # 将图像转换为RGB（如果使用的是灰度图，matplotlib会自动转换为RGB）
            img = plt.imread(output_image_path)
            # 转换为3通道RGB图像，假如原图是灰度图
            if len(img.shape) == 2:  # 如果是灰度图（1通道）
                img_rgb = np.stack([img] * 3, axis=-1)  # 将灰度图重复三次以生成RGB

                # 使用PIL保存为PNG图像
                from PIL import Image
                img_pil = Image.fromarray(img_rgb)
                img_pil.save(output_image_path)
            else:
                # 如果本身就是RGB图像，直接保存
                plt.close(fig)  # 关闭图像以便保存下一个

# 输入文件夹路径和输出文件夹路径
data_dir = r'D:\HTNet-master\NEW_MODEL\PPG_denosied'  # 输入文件夹路径
output_dir = r'D:\HTNet-master\NEW_MODEL\PPG_TU'   # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 调用函数处理并保存图像
process_and_save_images(data_dir, output_dir, image_size=224)
