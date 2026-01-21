import pandas as pd

# 读取 Excel 文件
file_path = r"D:\HTNet-master\NEW_MODEL\CASME3_updated.xlsx"
df = pd.read_excel(file_path)

# 确保 'sub' 和 'count' 列为字符串，两位数格式
df['sub'] = df['sub'].apply(lambda x: f"{int(x):02d}")
df['count'] = df['count'].apply(lambda x: f"{int(x):02d}")

# 创建 'filename' 列，格式为 'sub_count'
df['filename'] = df['sub'] + "_" + df['count']

# 创建 'label' 列，映射 emotion 到数字
label_mapping = {'negative': 0, 'positive': 1, 'surprise': 2}
df['label'] = df['emotion'].map(label_mapping).fillna(3).astype(int)

# 保存新文件
output_path = r"D:\HTNet-master\NEW_MODEL\CASME3_updated2.xlsx"
df.to_excel(output_path, index=False)
output_path
