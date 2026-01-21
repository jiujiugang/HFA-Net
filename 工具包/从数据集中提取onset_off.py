import os
import shutil
import pandas as pd

# 设置路径
DATASET_PATH = r"D:\WSA\c3\part_C\RGB_Depth\part_C\part_C"  # 数据集根目录
OUTPUT_PATH = r"D:\ZP\CASME_3"  # 目标目录
TABLE_PATH = r"D:\WSA\c3\CAS(ME)3_part_C_ME.xlsx"  # Excel 表格文件

# 读取 Excel 表格并根据受试者 ID 分组
annotations = pd.read_excel(TABLE_PATH, engine="openpyxl")
annotations_grouped = annotations.groupby('sub')

# 遍历所有受试者文件夹
for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue  # 跳过非文件夹

    # 进入子文件夹 `a`（自动检测）
    sub_folders = [f for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))]
    if len(sub_folders) == 1:  # 只有一个子文件夹（即 `a`）
        subject_path = os.path.join(subject_path, sub_folders[0])  # 更新路径
    else:
        print(f"Skipping {subject}, unexpected folder structure: {sub_folders}")
        continue  # 避免错误

    # 解析受试者 ID，确保匹配表格数据
    try:
        subject_id = int(subject)  # 假设文件夹名称为 "01", "02"...
    except ValueError:
        print(f"Skipping invalid subject folder: {subject}")
        continue

    # 获取该受试者的 Onset 和 Offset 数据
    if subject_id not in annotations_grouped.groups:
        print(f"Skipping subject {subject_id}, no data found in the table.")
        continue

    subject_data = annotations_grouped.get_group(subject_id)

    # 遍历当前受试者的数据行
    for idx, row in subject_data.iterrows():
        # 确保 Onset 和 Offset 有效
        if pd.isna(row['onset']) or pd.isna(row['offset']):
            print(f"Skipping subject {subject_id} due to missing Onset/Offset.")
            continue

        onset, offset = int(row['onset']), int(row['offset'])
        count = int(row['count'])  # 获取 count 列的值

        for mode in ['color', 'depth']:
            input_folder = os.path.join(subject_path, mode)  # `a/color` 或 `a/depth`

            # 根据 count 列来创建目标文件夹路径
            output_folder_base = os.path.join(OUTPUT_PATH, f"sub{subject_id}", mode)
            os.makedirs(output_folder_base, exist_ok=True)

            # 创建每个 count 对应的子文件夹，避免重复生成
            for sample_idx in range(1, count + 1):
                output_folder_name = f"{subject_id}_{count}"  # 使用 subject_id 和 count 作为文件夹名
                output_folder = os.path.join(output_folder_base, output_folder_name)

                # 如果文件夹不存在则创建
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # 计算每个子文件夹需要的帧范围（依据当前的 onset 和 offset）
                for frame_num in range(onset, offset + 1):
                    frame_name = f"{frame_num}.jpg" if mode == "color" else f"{frame_num}.png"
                    src_path = os.path.join(input_folder, frame_name)
                    dst_path = os.path.join(output_folder, frame_name)

                    # 如果源文件存在，则复制到目标文件夹
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
                    else:
                        print(f"Warning: {src_path} not found")

print("Frame extraction complete!")
