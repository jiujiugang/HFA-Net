import torch
import numpy as np
from facenet_pytorch import MTCNN


# 提取人脸关键点
def extract_face_landmarks(image):
    mtcnn = MTCNN(keep_all=True)  # MTCNN 实例化
    faces = mtcnn.detect(image)  # 检测人脸

    if faces[0] is not None and len(faces[0]) > 0:
        # 获取第一个人脸的关键点（landmarks）
        return faces[1][0]  # 假设 faces[1][0] 是关键点数据
    else:
        # 没有检测到人脸
        return None


# 生成注意力掩码
def generate_attention_mask(face_landmarks, image_size=(224, 224)):
    # 检查 face_landmarks 是否有效
    if face_landmarks is None or not face_landmarks:
            # print("No face landmarks detected, using default attention mask.")
        return torch.ones(image_size[0], image_size[1]).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    mask = torch.zeros(image_size, dtype=torch.float32)

    # 假设 face_landmarks 是一个形如 [(x1, y1), (x2, y2), ...] 的列表
    for (x, y) in face_landmarks:
        if x is not None and y is not None:  # 确保坐标是有效的
            mask[max(0, int(y) - 5):min(image_size[1], int(y) + 5),
            max(0, int(x) - 5):min(image_size[0], int(x) + 5)] = 1.0

    return mask.unsqueeze(0).unsqueeze(0)  # 增加批次和通道维度

