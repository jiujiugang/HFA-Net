import os


def rename_png_to_jpg_in_place(folder):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder):
        # 检查文件是否为 PNG 格式
        if file_name.endswith(".png"):
            # 获取文件的完整路径
            png_path = os.path.join(folder, file_name)

            # 生成新的 JPG 文件名
            jpg_name = file_name.replace(".png", ".jpg")
            jpg_path = os.path.join(folder, jpg_name)

            # 重命名文件
            os.rename(png_path, jpg_path)
            print(f"Renamed {file_name} to {jpg_name}")


# 设置文件夹路径
folder = r"D:\HTNet-master\NEW_MODEL\PPG_TU_224"  # 替换为你的文件夹路径

# 执行重命名
rename_png_to_jpg_in_place(folder)
