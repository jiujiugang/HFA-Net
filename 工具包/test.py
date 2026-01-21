import os
import pandas as pd

# 读取 .csv 文件
csv_file_path = r"C:\Users\24291\Desktop\combined_3_class2_for_optical_flow.csv"  # 请根据实际路径修改
df = pd.read_csv(csv_file_path)

# 定义图片文件夹路径（请根据实际路径修改）
image_folder_path = r"C:\Users\24291\Desktop\total_offset-apex" # 修改为实际文件夹路径

# 遍历每一行，重命名图片
for index, row in df.iterrows():
    # 获取原始图片的编号（这里是数字，不带.jpg）
    image_xulie = str(row['image_xulie'])

    # 获取新的图片名称（imagename列）
    new_image_name = str(row['imagename']).replace('.jpg', '.png')  # 确保后缀是 .png

    # 构造图片的原始文件路径和新文件路径
    original_image_path = os.path.join(image_folder_path, f"{image_xulie}.png")  # 图片文件名包含.png
    new_image_path = os.path.join(image_folder_path, new_image_name)

    # 调试：打印原始路径以检查是否正确
    print(f"Checking: {original_image_path}")

    # 如果文件存在，执行重命名
    if os.path.exists(original_image_path):
        os.rename(original_image_path, new_image_path)
        print(f"Renamed: {original_image_path} -> {new_image_path}")
    else:
        print(f"Image {original_image_path} not found.")

print("Renaming process completed.")
