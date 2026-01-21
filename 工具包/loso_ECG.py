import os
import shutil
import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(r'D:\HTNet-master\NEW_MODEL\CASME3_updated1.xlsx')

# 确保 'image_name' 列的数据是字符串类型，并处理缺失值
df['image_name'] = df['image_name'].fillna('').astype(str)

# 确保 'sub' 列的数据是字符串类型，并处理缺失值
df['sub'] = df['sub'].fillna('').astype(str)

# 确保 'filename' 列的数据是字符串类型，并处理缺失值
df['filename'] = df['filename'].fillna('').astype(str)

# 指定原始 ECG 文件夹路径
original_ecg_dir = r'D:\HTNet-master\NEW_MODEL\ECG_NO—DE\ECG_de_224'  # ECG信号文件夹路径

# 指定新数据集的文件夹路径
base_dir = r'D:\HTNet-master\NEW_MODEL\ECG_NO—DE\loso_de'

# 遍历每个唯一的 subject
for subject in df['sub'].unique():
    if not subject:  # 跳过空的 subject
        continue

    # 确保 subject 为2位数字格式
    formatted_subject = subject.zfill(2)

    subject_df = df[df['sub'] == subject]

    # 为每个 subject 创建主文件夹
    subject_dir = os.path.join(base_dir, formatted_subject)
    os.makedirs(subject_dir, exist_ok=True)

    # 创建 u_test 和 u_train 文件夹
    u_test_dir = os.path.join(subject_dir, 'u_test')
    os.makedirs(u_test_dir, exist_ok=True)
    u_train_dir = os.path.join(subject_dir, 'u_train')
    os.makedirs(u_train_dir, exist_ok=True)

    # 在 u_test 中创建 label 文件夹
    for label in subject_df['label'].unique():
        os.makedirs(os.path.join(u_test_dir, str(label)), exist_ok=True)

    # 处理每个 subject 的数据
    for _, row in subject_df.iterrows():
        # 获取 ECG 信号文件名
        img_file = row['image_name']
        if not img_file:  # 跳过空的 img 文件名
            continue

        ecg_file = img_file.replace('.jpg', '.txt')  # 将 .jpg 替换为 .txt
        src_ecg_file = os.path.join(original_ecg_dir, ecg_file)  # 直接使用 ECG 信号名称
        label = row['label']

        # 移动到 u_test 对应的 label 文件夹
        dst_dir = os.path.join(u_test_dir, str(label))
        os.makedirs(dst_dir, exist_ok=True)
        dst_ecg_file = os.path.join(dst_dir, ecg_file)

        # 处理 ECG 信号文件是否存在
        if os.path.isfile(src_ecg_file):
            shutil.copy(src_ecg_file, dst_ecg_file)  # 使用 shutil.move 如果需要移动文件而非复制
        else:
            print(f"ECG 文件 {src_ecg_file} 不存在")

    # 处理 u_train 文件夹中的数据
    other_subjects_df = df[df['sub'] != subject]

    for label in other_subjects_df['label'].unique():
        os.makedirs(os.path.join(u_train_dir, str(label)), exist_ok=True)

    for _, row in other_subjects_df.iterrows():
        # 获取 ECG 信号文件名
        img_file = row['image_name']
        if not img_file:  # 跳过空的 img 文件名
            continue

        ecg_file = img_file.replace('.jpg', '.txt')  # 将 .jpg 替换为 .txt
        src_ecg_file = os.path.join(original_ecg_dir, ecg_file)  # 直接使用 ECG 信号名称
        label = row['label']

        # 移动到 u_train 对应的 label 文件夹
        dst_dir = os.path.join(u_train_dir, str(label))
        os.makedirs(dst_dir, exist_ok=True)
        dst_ecg_file = os.path.join(dst_dir, ecg_file)

        # 处理 ECG 信号文件是否存在
        if os.path.isfile(src_ecg_file):
            shutil.copy(src_ecg_file, dst_ecg_file)  # 使用 shutil.move 如果需要移动文件而非复制
        else:
            print(f"ECG 文件 {src_ecg_file} 不存在")
