import numpy as np
import pywt
import matplotlib.pyplot as plt
import os


# DWT小波去噪函数
def DWT(signal, wavelet='db4'):
    # 小波变换
    coeffs = pywt.wavedec(data=signal, wavelet=wavelet, level=8)  # 进行8层小波变换
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs  # 提取近似系数（cA8）和细节系数（cD8到cD1）

    # 自适应阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))  # 计算阈值
    cD1.fill(0)  # 将细节系数cD1清零
    cD2.fill(0)  # 将细节系数cD2清零
    cA8.fill(0)  # 将近似系数cA8清零
    for i in range(0, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)  # 对其他细节系数进行去噪处理

    # 小波反变换，获取去噪后的信号
    denoised_signal = pywt.waverec(coeffs=coeffs, wavelet=wavelet)
    return denoised_signal


# 处理txt文件并进行DWT去噪
def process_txt_files(data_dir, output_dir, wavelet='db4'):
    # 用来存放去噪后的信号
    denoised_signals = []

    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)

            # 读取ECG信号数据
            with open(file_path, 'r') as f:
                signal = np.array([float(line.strip()) for line in f.readlines()])  # 读取每行数据，并转化为数组

            # 对读取的信号进行小波去噪
            denoised_signal = DWT(signal, wavelet)
            denoised_signals.append(denoised_signal)

            # 只显示前3个信号的去噪前后对比
            if len(denoised_signals) <= 3:  # 只显示前3个信号
                plt.figure(figsize=(12, 6))
                plt.plot(signal, label='Original Signal', alpha=0.6)
                plt.plot(denoised_signal, label='Denoised Signal', alpha=0.6)
                plt.title(f'Original and Denoised Signal: {filename}')
                plt.xlabel('Sample Index')
                plt.ylabel('Signal Value')
                plt.legend()
                plt.show()

            # 保存去噪后的信号到output_dir
            output_file = os.path.join(output_dir, f"{filename}")
            np.savetxt(output_file, denoised_signal)

    return denoised_signals


# 目录路径：将此路径替换为存储txt文件的文件夹路径
data_dir = r'D:\HTNet-master\NEW_MODEL\PPG'  # 输入文件夹路径
output_dir = r'D:\HTNet-master\NEW_MODEL\PPG_denosied'  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 调用处理函数
denoised_signals = process_txt_files(data_dir, output_dir, wavelet='db4')

# 如果你需要进一步操作，或导出其他文件，也可以在这里进行扩展
