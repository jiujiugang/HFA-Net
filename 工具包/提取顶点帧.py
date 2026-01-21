import os
import shutil
import pandas as pd

# 读取 Excel 文件
xlsx_path = r"D:\HTNet-master\NEW_MODEL\CASME3_updated1.xlsx"
df = pd.read_excel(xlsx_path)

# 设置原始数据集路径和目标存放路径
source_root = r"D:\HTNet-master\NEW_MODEL\CASME_3"
dest_root = r"D:\HTNet-master\NEW_MODEL\CASME_3_APEX"

# 确保目标目录存在
os.makedirs(dest_root, exist_ok=True)

# 遍历每一行数据
for _, row in df.iterrows():
    sub = f"{int(row['sub']):02d}"  # 确保受试者编号为两位数
    count = str(row['count']).zfill(2)  # 保证 count 始终为两位字符串
    apex = row['apex']
    image_name = row['image_name']

    # 受试者 color 目录
    color_dir = os.path.join(source_root, sub, "color")

    # 计算对应的文件夹名称
    folder_name = f"{int(sub)}_{int(count)}"  # 例如受试者 02，count 01 时文件夹为 2_1

    # 查找对应的文件夹
    possible_dirs = [d for d in os.listdir(color_dir) if d == folder_name]

    if not possible_dirs:
        print(f"未找到 count 目录: {color_dir}/{folder_name}")
        continue

    # 取第一个匹配的文件夹
    source_dir = os.path.join(color_dir, possible_dirs[0])

    # 构造源图片路径
    source_image = os.path.join(source_dir, f"{apex}.jpg")

    # 输出调试信息
    print(f"处理受试者 {sub}, count {count}, apex {apex}")
    print(f"源目录: {source_dir}")
    print(f"目标图片: {source_image}")

    # 构造目标图片路径
    dest_image = os.path.join(dest_root, image_name)

    # 复制并重命名图片
    if os.path.exists(source_image):
        shutil.copy2(source_image, dest_image)
        print(f"已复制: {source_image} -> {dest_image}")
    else:
        print(f"未找到: {source_image}")

print("处理完成！")
