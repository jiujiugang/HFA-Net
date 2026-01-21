import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, lfilter
from scipy.ndimage import zoom

# ==========================================================
# 图像保存函数
# ==========================================================
def save_no_axis_cwt_image(power, extent, save_path):
    """保存无坐标图"""
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(power, extent=extent, aspect='auto', cmap='jet', origin='lower')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==========================================================
# Morlet 小波变换与保存
# ==========================================================
def crop_and_save_cwt(signal, fs, signal_type, subject_id, save_dir_no_axis):
    """使用 Morlet 小波计算 CWT 并保存"""
    signal = signal - np.mean(signal)
    signal = signal / (np.std(signal) + 1e-8)

    # 关键频域
    if signal_type.upper() == 'RESP':
        f_min, f_max, num_scales = 0.05, 0.4, 256
    elif signal_type.upper() == 'ECG':
        f_min, f_max, num_scales = 0.5, 45, 256
    else:
        raise ValueError("signal_type must be 'RESP' or 'ECG'")

    freqs = np.linspace(f_min, f_max, num_scales)
    fc = pywt.central_frequency('morl')
    scales = fc * fs / freqs

    coef, _ = pywt.cwt(signal, scales, 'morl', sampling_period=1 / fs)
    power = np.abs(coef) ** 2
    power -= np.min(power)
    if np.max(power) != 0:
        power /= np.max(power)
    power = np.power(power, 0.8)  # γ增强

    time = np.linspace(0, len(signal) / fs, len(signal))
    extent = [time[0], time[-1], freqs.min(), freqs.max()]

    # 生成时频图并保存
    save_path = os.path.join(save_dir_no_axis, f"{subject_id}_{signal_type}.png")
    save_no_axis_cwt_image(power, extent, save_path)

    print(f"✅ 已保存 {signal_type} 时频图 | 文件: {subject_id}_{signal_type}.png")

# ==========================================================
# 主处理流程
# ==========================================================
base_data_dir = r"D:/HTNet-master/NEW_MODEL/ECG/ECG_crop/ECG100C_Crop"
output_path = r"D:\HTNet-master\NEW_MODEL\ECG\ECG-shiyu"

# 设定原始信号的采样频率
fs_original = 700  # 假设信号采样频率为700 Hz，根据实际数据调整

# 获取所有的.txt文件
txt_files = [f for f in os.listdir(base_data_dir) if f.endswith('.txt')]

# 读取每个文件并生成时频图
for file_name in txt_files:
    try:
        file_path = os.path.join(base_data_dir, file_name)

        # 读取ECG信号
        ecg = np.loadtxt(file_path)
        subject_id = file_name.split('.')[0]  # 使用文件名作为 subject_id

        # 信号已经是截取好的片段，直接使用
        signal = ecg

        # 生成时频图
        crop_and_save_cwt(signal, fs_original, 'ECG', subject_id, output_path)
        print(f"🎯 完成 {file_name}")

    except Exception as e:
        print(f"❌ 错误：{file_name}, 信息：{e}")

print("✅ 所有时频图生成完毕！")
