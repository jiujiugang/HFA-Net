import os
import sys
import cv2
import numpy as np
import pandas as pd
import face_recognition.api as face_recognition
import matplotlib.pyplot as plt
from tqdm import tqdm

data_root = r'D:\HTNet-master\NEW_MODEL\CASME_3'                                      # 数据集所在的根目录 为SMIC数据集裁剪后的图片的顶点帧检测
annotation_file = r'/NEW_MODEL/CAS(ME)3_part_C_ME.xlsx'  # 数据集的注释文件名称
label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}                  # 定义类别与数字编码之间的对应关系,用一个字典表示,关键字是类别名称,值是对应的数字编码。


def get_clip_frame_paths(subject, count, onset, offset):
    """
    获取从onset到offset之间的所有帧路径（不保证连续）
    :param subject: 受试者编号（如1, 2等）
    :param count: 微表情序列编号（如1, 2, 3等）
    :param onset: 起始帧编号
    :param offset: 结束帧编号
    :return: 帧路径列表
    """
    frame_paths = []
    subject_dir = os.path.join(data_root, f"{int(subject):02d}")  # 受试者文件夹路径，如01, 02等
    clip_dir = os.path.join(subject_dir, 'color', f"{int(subject):02d}_{count}")  # 微表情序列文件夹路径，如01/color/1_1

    # 遍历从onset到offset的所有帧
    for idx in range(onset, offset + 1):
        frame_path = os.path.join(clip_dir, f"{idx}.jpg")  # 帧文件命名规则为373.jpg等
        if os.path.exists(frame_path):  # 只添加存在的帧
            frame_paths.append(frame_path)
        else:
            print(f'Warning: Frame {frame_path} does not exist, skipping.')
    return frame_paths


# 用于检测图片中的唯一一张人脸关键点
def detect_lmks(frame):
    """
    检测图片中的唯一一张人脸关键点
    :param frame: 输入的图像帧
    :return: 人脸关键点字典，如果未检测到人脸则返回 None
    """
    try:
        lmks = face_recognition.face_landmarks(frame)
        if not lmks:  # 如果未检测到人脸
            print('Warning: No face detected in the frame.')
            return None
        return lmks[0]  # 返回第一张人脸的关键点
    except Exception as e:
        print(f'Error detecting landmarks: {e}')
        return None

# 从图像中裁剪出特定区域的cell
def get_cell(img, cell_location):
# img:原始图像
# cell_location：细胞区域的左上和右下角坐标点组成的元组
    point1, point2 = cell_location
    cell = img[point1[1]:point2[1], point1[0]:point2[0]]
    return cell

# 根据给定的人脸关键点坐标landmarks,计算出各个细胞区域的坐标框
def get_cell_locations(lmks):
    # 定义get_rect函数,可以传入中心点和宽度计算坐标框 输出为中心点坐标减去一半宽度的左坐标 和  中心点坐标加上一半宽度的右坐标
    def get_rect(center, width):
        point1 = np.array(center) - int(width / 2)
        point2 = np.array(center) + int(width / 2)
        return tuple(point1), tuple(point2)
    # 创建一个空的字典cells来存储提取的细胞区域坐标。
    # 计算上嘴唇区域的宽度作为细胞的默认宽度cell_width。
    cells = {}
    cell_width = int((lmks['top_lip'][6][0] - lmks['top_lip'][0][0]) / 2)

    key = 'top_lip'# 上嘴唇
    points = np.array(lmks[key])
    left_lip_rect = get_rect(points[0], cell_width)
    right_lip_rect = get_rect(points[6], cell_width)
    cells['left_lip'] = left_lip_rect
    cells['right_lip'] = right_lip_rect
    # 将得到的左右细胞坐标框存入cells字典中。

    key = 'chin' # 下巴
    point = lmks[key][int(len(lmks[key]) / 2)]
    rect_point1 = (point[0] - int(cell_width / 2), point[1] - cell_width)
    rect_point2 = (point[0] + int(cell_width / 2), point[1])
    chin_rect = (rect_point1, rect_point2)
    # 将提取的下巴细胞坐标框存入cells字典
    cells['chin_rect'] = chin_rect

    key = 'nose_tip'
    point = lmks[key][0]
    left_nose_rect_point1 = (point[0] - cell_width, left_lip_rect[0][1] - cell_width)
    left_nose_rect_point2 = (point[0], left_lip_rect[0][1])
    left_nose_rect = (left_nose_rect_point1, left_nose_rect_point2)
    cells['left_nose'] = left_nose_rect

    point = lmks[key][4]
    right_nose_rect_point1 = (point[0], right_lip_rect[0][1] - cell_width)
    right_nose_rect_point2 = (point[0] + cell_width, right_lip_rect[0][1])
    right_nose_rect = (right_nose_rect_point1, right_nose_rect_point2)
    cells['right_nose'] = right_nose_rect

    key = 'left_eye'
    point = lmks[key][0]
    left_eye_rect_point1 = (point[0] - cell_width, int(point[1] - cell_width / 2))
    left_eye_rect_point2 = (point[0], int(point[1] + cell_width / 2))
    left_eye_rect = (left_eye_rect_point1, left_eye_rect_point2)
    cells['left_eye'] = left_eye_rect

    key = 'right_eye'
    point = lmks[key][3]
    right_eye_rect_point1 = (point[0], int(point[1] - cell_width / 2))
    right_eye_rect_point2 = (point[0] + cell_width, int(point[1] + cell_width / 2))
    right_eye_rect = (right_eye_rect_point1, right_eye_rect_point2)
    cells['right_eye'] = right_eye_rect

    left_point = lmks['left_eyebrow'][2]
    right_point = lmks['right_eyebrow'][2]
    center_point = (int((left_point[0] + right_point[0]) / 2),
                    int((left_point[1] + right_point[1]) / 2))

    center_eyebrow_rect = get_rect(center_point, cell_width)
    cells['center_eyebrow'] = center_eyebrow_rect

    left_rect_point1 = (int(center_point[0] - cell_width * 3 / 2),
                        int(center_point[1] - cell_width / 2))
    left_rect_point2 = (int(center_point[0] - cell_width * 1 / 2),
                        int(center_point[1] + cell_width / 2))
    left_eyebrow_rect = (left_rect_point1, left_rect_point2)
    cells['left_eyebrow'] = left_eyebrow_rect

    right_rect_point1 = (int(center_point[0] + cell_width * 1 / 2),
                         int(center_point[1] - cell_width / 2))
    right_rect_point2 = (int(center_point[0] + cell_width * 3 / 2),
                         int(center_point[1] + cell_width / 2))
    right_eyebrow_rect = (right_rect_point1, right_rect_point2)
    cells['right_eyebrow'] = right_eyebrow_rect

    return cells, cell_width

# 输入参数为细胞在当前时刻t的值cell_t,起始时刻onset的值cell_onset,结束时刻offset的值cell_offset,以及一个衰减常数cell_epsilon
import os


def get_clip_frame_paths(subject, count, onset, offset, data_root):
    """
    获取从onset到offset之间的所有帧路径（不保证连续）
    :param subject: 受试者编号（如1, 2等）
    :param count: 微表情序列编号（如1, 2, 3等）
    :param onset: 起始帧编号
    :param offset: 结束帧编号
    :param data_root: 数据根路径
    :return: 帧路径列表
    """
    frame_paths = []
    subject_str = f"{int(subject):02d}"  # 将subject格式化为两位数字，如01, 10, 31
    subject_dir = os.path.join(data_root, subject_str)  # 受试者文件夹路径，如01, 10, 31

    # 生成正确的微表情序列文件夹名称，count保持原始数值
    clip_dir = os.path.join(subject_dir, 'color', f"{subject}_{count}")

    # 遍历从onset到offset的所有帧
    for idx in range(onset, offset + 1):
        frame_path = os.path.join(clip_dir, f"{idx}.jpg")  # 帧文件命名规则为373.jpg等
        if os.path.exists(frame_path):  # 只添加存在的帧
            frame_paths.append(frame_path)
        else:
            print(f'Warning: Frame {frame_path} does not exist, skipping.')

    return frame_paths


def compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon):
    """
    计算当前帧与起始帧、结束帧以及前一帧之间的差异
    :param cell_t: 当前帧的细胞区域
    :param cell_onset: 起始帧的细胞区域
    :param cell_offset: 结束帧的细胞区域
    :param cell_epsilon: 前一帧的细胞区域
    :return: 差异值（标量）
    """
    # 计算当前帧与起始帧的差异
    diff_onset = np.abs(cell_t - cell_onset)
    # 计算当前帧与结束帧的差异
    diff_offset = np.abs(cell_t - cell_offset)
    # 计算当前帧与前一帧的差异
    diff_epsilon = np.abs(cell_t - cell_epsilon)

    # 综合差异值（可以根据需要调整权重）
    difference = (diff_onset.mean() + diff_offset.mean()) / (diff_epsilon.mean() + 1e-6)  # 避免除以0
    return difference

def compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon):
    """
    计算当前帧的特征
    :param frame_t: 当前帧
    :param on_frame: 起始帧
    :param off_frame: 结束帧
    :param frame_epsilon: 前一帧
    :return: 特征字典，如果未检测到人脸则返回 None
    """
    lmks = detect_lmks(frame_t)  # 检测当前帧的关键点
    if lmks is None:  # 如果未检测到人脸
        return None

    try:
        cell_locations, cell_width = get_cell_locations(lmks)
        cell_differences = {}
        frame_t = frame_t.astype(np.float32)
        on_frame = on_frame.astype(np.float32)
        off_frame = off_frame.astype(np.float32)
        frame_epsilon = frame_epsilon.astype(np.float32)

        for key in cell_locations:
            cell_location = cell_locations[key]
            cell_t = get_cell(frame_t, cell_location)
            cell_onset = get_cell(on_frame, cell_location)
            cell_offset = get_cell(off_frame, cell_location)
            cell_epsilon = get_cell(frame_epsilon, cell_location)

            cell_difference = compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon)
            cell_differences[key] = cell_difference
        return cell_differences
    except Exception as e:
        print(f'Error computing cell features: {e}')
        return None

# 在一个视频片段的所有帧上,提取每帧的特征,并找到特征峰值最大的帧,作为顶点帧
def find_apex_frame_of_clip(frame_paths):
    """
    在一个视频片段的所有帧上，提取每帧的特征，并找到特征峰值最大的帧，作为顶点帧
    :param frame_paths: 帧路径列表
    :return: 顶点帧路径、特征值列表、顶点帧的相对索引
    """
    epsilon = 1  # 用于计算特征的帧间隔

    # 读取起始帧和结束帧
    on_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    off_frame = cv2.imread(frame_paths[-1], cv2.IMREAD_GRAYSCALE)

    features = []

    # 遍历所有帧，计算特征
    for i in range(epsilon, len(frame_paths)):
        frame_t = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
        frame_epsilon = cv2.imread(frame_paths[i - epsilon], cv2.IMREAD_GRAYSCALE)

        # 计算特征
        current_features = compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon)
        if current_features is None:  # 如果未检测到人脸或计算特征失败
            features.append(0.0)  # 添加默认特征值
            continue

        feature = sum(current_features.values()) / len(current_features)  # 计算平均特征值
        features.append(feature)

    # 找到特征峰值最大的帧
    padding = [0.0] * epsilon
    features = np.array(padding + features)
    apex_frame_idx = features.argmax()
    apex_frame_path = frame_paths[apex_frame_idx]

    return apex_frame_path, features, apex_frame_idx


def draw_avg_plot(features, pred_apex_idx, data, clip_name):
    x = list(range(len(features)))
    plt.plot(x, features)
    plt.axvline(x=pred_apex_idx, label='pred apex idx at={}'.format(pred_apex_idx), c='red')
    plt.legend()
    plt.savefig('plots/{}/{}.png'.format(data, clip_name))
    plt.clf()
    plt.cla()
    plt.close()


def on_all_clips():
    """
    处理所有微表情序列，找到顶点帧
    """
    # 定义数据根目录路径
    data_root = r"D:\HTNet-master\NEW_MODEL\CASME_3"  # 设置你的数据存放根目录路径

    # 读取注释文件
    df = pd.read_excel(annotation_file)
    data_list = []  # 使用字典列表代替多个独立列表

    # 遍历每一行数据
    with tqdm(total=len(df)) as progress_bar:
        for _, row in df.iterrows():
            subject = row['sub']  # 受试者编号
            count = row['count']  # 微表情序列编号
            onset = row['onset']  # 起始帧
            offset = row['offset']  # 结束帧
            emotion = row['emotion']  # 微表情类别

            # 获取帧路径，传递 data_root 参数
            clip_frame_paths = get_clip_frame_paths(subject, count, onset, offset, data_root)

            # 如果帧路径为空，跳过该序列
            if not clip_frame_paths:
                print(f'Warning: No frames found for subject {subject}, count {count}. Skipping.')
                progress_bar.update(1)  # 仍然更新进度条
                continue

            # 找到顶点帧
            apex_frame_path, features, apex_relative_idx = find_apex_frame_of_clip(clip_frame_paths)

            # 提取顶点帧编号
            apex_frame_idx = int(os.path.basename(apex_frame_path).split('.')[0])  # 从文件名中提取帧编号

            # 保存结果到字典
            data_list.append({
                'data': 'CAS(ME)3',
                'subject': subject,
                'count': count,
                'label': label_dict.get(emotion, -1),
                'onset_frame': onset,
                'apex_frame': apex_frame_idx,
                'offset_frame': offset,
                'onset_frame_path': clip_frame_paths[0],
                'apex_frame_path': apex_frame_path,
                'off_frame_path': clip_frame_paths[-1]
            })

            progress_bar.update(1)

    # 保存结果到CSV文件
    result_df = pd.DataFrame(data_list)
    result_df.to_csv('CAS(ME)3_apex_.csv', header=True, index=None)



import os
import matplotlib.pyplot as plt

def draw_avg_plot(features, pred_apex_idx, data, clip_name):
    if not features:
        print("⚠️ Features list is empty! Skipping plot.")
        return

    x = list(range(len(features)))

    # 创建目标目录
    save_dir = r'D:\HTNet-master\NEW_MODEL\plots'
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path = os.path.join(save_dir, '{}.png'.format(clip_name))

    # 调试信息
    print("==== Debug Info ====")
    print(f"Features length: {len(features)}")
    print(f"Pred Apex Index: {pred_apex_idx}")
    print(f"Save path: {save_path}")
    print(f"Is save directory exists? {os.path.exists(save_dir)}")
    print("====================")

    # 画图
    print(f"📊 Drawing plot for: {clip_name}")
    print(f"🔹 Feature length: {len(features)} | Pred Apex Index: {pred_apex_idx}")
    print(f"💾 Save path: {save_path}")

    plt.plot(x, features, label="Feature Curve")

    # 检查 pred_apex_idx 是否在合法范围内
    if 0 <= pred_apex_idx < len(features):
        plt.axvline(x=pred_apex_idx, label=f'Predicted Apex @ {pred_apex_idx}', c='red')
    else:
        print(f"⚠️ Warning: pred_apex_idx ({pred_apex_idx}) is out of range!")

    plt.legend()

    # 先显示图像，检查是否正确
    plt.show()

    try:
        plt.savefig(save_path)
        print(f"✅ Plot saved at: {save_path}")
    except Exception as e:
        print(f"❌ Error saving plot: {e}")

    plt.clf()
    plt.cla()
    plt.close()



if __name__ == '__main__':

    on_all_clips()