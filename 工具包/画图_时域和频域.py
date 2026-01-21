import numpy as np
import matplotlib.pyplot as plt

# 读取 ECG 数据的文件路径
def load_ecg_data(file_path):
    try:
        # 读取文件，假设数据是单列格式
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# 设置你的 ECG 数据文件路径（修改为你的实际路径）
file_path = r"D:\HTNet-master\NEW_MODEL\ECG_Denosied\DWT_Denosied\01_03.txt"  # 请替换成你的 ECG 文件路径

data = load_ecg_data(file_path)

if data is not None:
    # 绘制时域信号
    plt.figure(figsize=(10, 4))
    plt.plot(data, label="ECG Signal")
    plt.title("ECG Signal (Time Domain)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
else:
    print("No data to plot.")

from scipy.fftpack import fft

# 计算 FFT
N = len(data)
ecg_freq = fft(data)
freqs = np.fft.fftfreq(N, d=1/200)  # 假设采样率为 200Hz

# 画出频谱
plt.figure(figsize=(10,4))
plt.plot(freqs[:N//2], np.abs(ecg_freq[:N//2]))  # 只看正频率部分
plt.title("ECG Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()
