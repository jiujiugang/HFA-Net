import os
import pandas as pd

# 配置路径
txt_folder = r'D:\HTNet-master\NEW_MODEL\PPG'  # 存放 .txt 文件的目录
xlsx_path = r'D:\HTNet-master\NEW_MODEL\CAS(ME)3_part_C_ME.xlsx'  # xlsx 文件路径

# 读取 xlsx 文件，获取 PHY 列（假设包含文件名）
df = pd.read_excel(xlsx_path, engine='openpyxl')

# 遍历 PHY 列中的文件名
for phy in df["PHY"]:
    txt_file = os.path.join(txt_folder, f"{phy}.txt")  # 生成 .txt 文件完整路径

    if os.path.exists(txt_file):
        # 读取 .txt 文件（跳过第一行）
        data = pd.read_csv(txt_file, skiprows=1, header=None)  # `skiprows=1` 跳过第一行

        # 重新保存去掉第一行的内容，覆盖原文件
        data.to_csv(txt_file, index=False, header=False)  # 不保存索引和列名
        print(f"处理完成: {txt_file}")
    else:
        print(f"文件不存在: {txt_file}")

print("所有文件处理完成！")
