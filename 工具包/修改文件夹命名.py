import os

# 文本文件夹路径
text_dir = r'D:\HTNet-master\NEW_MODEL\ECG100C_Denoised'  # 修改为你的文件夹路径

# 遍历文件夹中的所有 .txt 文件
for filename in os.listdir(text_dir):
    if filename.endswith('.txt'):  # 只处理以 .txt 结尾的文件
        # 去掉 .txt 后缀
        filename_without_extension = filename[:-4]

        # 通过下划线分割文件名
        parts = filename_without_extension.split('_')

        if len(parts) == 2:  # 确保文件名格式正确，只有两个部分
            # 对两个部分进行格式化，确保每个部分都是两位数
            new_filename = f"{int(parts[0]):02}_{int(parts[1]):02}.txt"

            old_filepath = os.path.join(text_dir, filename)
            new_filepath = os.path.join(text_dir, new_filename)

            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {old_filepath} to {new_filepath}")

print("Renaming completed.")
