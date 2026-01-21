import pandas as pd
import matplotlib.pyplot as plt


# 读取 ECG 数据
file_path = r'D:\HTNet-master\NEW_MODEL\ECG\ECG_crop\ECG100C_Crop\01_01.txt'  # 替换为你的路径
data = pd.read_csv(file_path, header=0, names=["ECG"])  # 确保第一行是列名

# 画图
plt.figure(figsize=(10, 6))
plt.plot(data["ECG"], label='ECG Signal')
plt.title('ECG Signal')
plt.xlabel('Time (samples)')  # 如果有时间数据，可以替换成 'Time (s)'
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
