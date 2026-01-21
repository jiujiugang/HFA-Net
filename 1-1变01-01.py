from pathlib import Path
import re

# 改成你的目录（Windows 路径建议用原始字符串 r"..."）
folder = Path(r"D:\HTNet-master\NEW_MODEL\PPG_TU_224")

# 同时匹配连字符或下划线；支持 jpg/jpeg/png；忽略大小写
pat = re.compile(r'^(\d+)[-_](\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)

renamed = skipped = unmatched = 0

for p in sorted(folder.iterdir()):
    if not p.is_file():
        continue
    m = pat.match(p.name)
    if not m:
        unmatched += 1
        # 如需查看哪些没匹配到，可取消下一行注释
        # print("不匹配：", p.name)
        continue

    a, b, ext = m.groups()
    new_name = f"{int(a):02d}_{int(b):02d}.{ext.lower()}"
    dest = p.with_name(new_name)

    if p.name == new_name:
        skipped += 1
        # print("已是目标格式：", p.name)
        continue
    if dest.exists():
        skipped += 1
        print(f"跳过（目标已存在）：{p.name} -> {new_name}")
        continue

    p.rename(dest)
    print(f"重命名：{p.name} -> {new_name}")
    renamed += 1

print(f"完成：重命名 {renamed} 个，跳过 {skipped} 个，不匹配 {unmatched} 个。")
