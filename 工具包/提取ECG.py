import os
import pandas as pd

# 设置文件路径
xlsx_path = R"D:\HTNet-master\NEW_MODEL\CAS(ME)3_part_C_ME.xlsx"  # xlsx 文件路径
source_folder = r"D:\HTNet-master\NEW_MODEL\EDA\EDA_corp"  # 存放原始数据的文件夹
output_folder = r"D:\HTNet-master\NEW_MODEL\PPG"  # 存放ECG数据的文件夹

# 读取 xlsx 文件
excel_data = pd.read_excel(xlsx_path)
phy_column = "PHY"  # xlsx 里存储文件名的列名

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历 xlsx 中的文件名
for file_name in excel_data[phy_column]:
    input_file = os.path.join(source_folder, f"{file_name}.txt")
    output_file = os.path.join(output_folder, f"{file_name}.txt")

    if os.path.exists(input_file):
        # 读取文件并提取 ECG100C 列
        df = pd.read_csv(input_file, delim_whitespace=True)  # 按空格分隔
        if "PPG" in df.columns:
            df_ecg = df[["PPG"]]
            df_ecg.to_csv(output_file, index=False, sep='\t')
            print(f"Processed: {file_name}.txt")
        else:
            print(f"Warning: 'EDA' column not found in {file_name}.txt")
    else:
        print(f"Warning: File {file_name}.txt not found.")

print("Processing complete.")