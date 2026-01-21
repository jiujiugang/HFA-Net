import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, lfilter, resample
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================================
# 滤波器定义
# ==========================================================
def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=4):
    """低通滤波器（RESP）"""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)


def signal_filter(data, frequency, highpass=50, lowpass=0.5):
    """带通滤波器（ECG）"""
    b, a = butter(3, [lowpass / frequency * 2, highpass / frequency * 2], 'bandpass')
    return lfilter(b, a, data)


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
def crop_and_save_cwt(signal, fs, signal_type, subject_id, emotion, segment_idx, time_range, save_dir_no_axis):
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

    os.makedirs(os.path.join(save_dir_no_axis, emotion), exist_ok=True)
    save_path = os.path.join(save_dir_no_axis, emotion,
                             f"{subject_id}_{emotion}_seg{segment_idx}_{signal_type}.png")
    save_no_axis_cwt_image(power, extent, save_path)

    print(f"✅ 已保存 {signal_type} 时频图: seg{segment_idx} | 时间段: {time_range[0]:.2f}–{time_range[1]:.2f} s")


# ==========================================================
# 主处理流程
# ==========================================================
base_data_dir = "D:/文献复现/WESAD"
emotion_folders = {
    "Baseline": os.path.join(base_data_dir, "Baseline"),
    "Stress": os.path.join(base_data_dir, "Stress"),
    "Happy": os.path.join(base_data_dir, "Happy"),
}

resp_output_path = r"E:\三个数据库处理的数据\时频图Morlet反推\RESP"
ecg_output_path = r"E:\三个数据库处理的数据\时频图Morlet反推\ECG"

fs_original = 700
fs_target_ecg = 360
fs_target_resp = 100
segment_duration = 60  # 每段 60 s

stats_dict = {}

for emotion, folder_path in emotion_folders.items():
    print(f"\n📂 正在处理情绪：{emotion}")
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    for file_name in file_list:
        try:
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_pickle(file_path)
            subject_id = file_name.split('_')[0]

            # ECG 滤波 + 重采样
            ecg = data['ECG'].values.flatten()
            ecg_filtered = signal_filter(ecg, frequency=fs_original)
            ecg_resampled = resample(ecg_filtered, int(len(ecg_filtered) * fs_target_ecg / fs_original))

            # RESP 滤波 + 重采样
            resp = data['RESP'].values.flatten()
            resp_filtered = butter_lowpass_filter(resp, cutoff_freq=0.4, sampling_rate=fs_original)
            resp_resampled = resample(resp_filtered, int(len(resp_filtered) * fs_target_resp / fs_original))

            # ================= 从后往前分段 =================
            seg_len_ecg = int(segment_duration * fs_target_ecg)
            seg_len_resp = int(segment_duration * fs_target_resp)

            num_seg_ecg = len(ecg_resampled) // seg_len_ecg
            num_seg_resp = len(resp_resampled) // seg_len_resp

            ecg_total_time = len(ecg_resampled) / fs_target_ecg
            resp_total_time = len(resp_resampled) / fs_target_resp

            ecg_count, resp_count = 0, 0

            # ECG: 从后往前取
            for i in range(num_seg_ecg):
                end = ecg_total_time - i * segment_duration
                start = end - segment_duration
                start_idx = int(start * fs_target_ecg)
                end_idx = int(end * fs_target_ecg)

                segment = ecg_resampled[start_idx:end_idx]
                if len(segment) == seg_len_ecg:
                    crop_and_save_cwt(segment, fs_target_ecg, 'ECG',
                                      subject_id, emotion, i + 1, (start, end), ecg_output_path)
                    ecg_count += 1

            # RESP: 从后往前取
            for i in range(num_seg_resp):
                end = resp_total_time - i * segment_duration
                start = end - segment_duration
                start_idx = int(start * fs_target_resp)
                end_idx = int(end * fs_target_resp)

                segment = resp_resampled[start_idx:end_idx]
                if len(segment) == seg_len_resp:
                    crop_and_save_cwt(segment, fs_target_resp, 'RESP',
                                      subject_id, emotion, i + 1, (start, end), resp_output_path)
                    resp_count += 1

            # 统计
            if subject_id not in stats_dict:
                stats_dict[subject_id] = {'ECG': 0, 'RESP': 0}
            stats_dict[subject_id]['ECG'] += ecg_count
            stats_dict[subject_id]['RESP'] += resp_count

            print(f"🎯 完成 {file_name} | ECG 段数={ecg_count}, RESP 段数={resp_count}")

        except Exception as e:
            print(f"❌ 错误：{file_name}, 信息：{e}")


# ==========================================================
# 汇总统计
# ==========================================================
print("\n📊 各被试生成的时频图数量：")
for sid, counts in sorted(stats_dict.items()):
    print(f"  {sid}: ECG={counts['ECG']} 张, RESP={counts['RESP']} 张")

total_ecg = sum(v['ECG'] for v in stats_dict.values())
total_resp = sum(v['RESP'] for v in stats_dict.values())
print(f"\n✅ 总计: ECG={total_ecg} 张, RESP={total_resp} 张")

