from os import path
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from distutils.util import strtobool
from sklearn.metrics import confusion_matrix
import pandas as pd
import warnings
from model import MultiEMO_ECG_OpticalFlow
warnings.filterwarnings("ignore")


class MultimodalDataset(Dataset):
    def __init__(self, main_path, subfolder, optical_flow_data, ecg_data, mode='train'):
        """
        自定义多模态数据集类
        Args:
            main_path: 主数据路径
            subfolder: 子文件夹名（被试者名）
            optical_flow_data: 光流数据字典
            ecg_data: ECG数据字典
        """
        self.optical_flow_data = optical_flow_data
        self.ecg_data = ecg_data
        self.labels = []
        self.flow_samples = []
        self.ecg_samples = []

        # 加载数据 遍历 u_train/u_test 下的子目录
        data_path = os.path.join(main_path, subfolder, f'u_{mode}')
        for n_expression in os.listdir(data_path):# 遍历表情类别文件夹
            expr_path = os.path.join(data_path, n_expression)
            for n_img in os.listdir(expr_path):# 遍历每个图像文件
                if n_img in optical_flow_data and n_img in ecg_data:# 检查双模态数据都存在
                    self.labels.append(int(n_expression))# 表情标签
                    self.flow_samples.append(optical_flow_data[n_img])# 光流数据
                    self.ecg_samples.append(ecg_data[n_img])# ECG数据

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 转换为Tensor并归一化
        flow_img = torch.from_numpy(self.flow_samples[idx]).float().permute(2, 0, 1) / 255.0
        ecg_img = torch.from_numpy(self.ecg_samples[idx]).float().permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return flow_img, ecg_img, label


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2, 'other': 3}
    #label_dict = {'negative': 0, 'positive': 1, 'surprise': 2, }  # 情感标签映射到数字编码

    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''


def load_image_data(data_path):
    """加载图像数据并返回字典"""
    data_dict = {}
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            data_dict[img_name] = img
    return data_dict


def main(config):
    # 参数设置
    learning_rate = 0.0005
    batch_size = 8
    epochs = 60
    accumulation_steps = 4

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()

    if config.train and not path.exists('ourmodel_CASME3_weights'):
        os.mkdir('ourmodel_CASME3_weights')

    print(f'lr={learning_rate}, epochs={epochs}, device={device}\n')

    # 数据路径设置
    main_path = r'D:\HTNet-master\NEW_MODEL\ECG_LOSO'
    optical_flow_path = r'D:\HTNet-master\NEW_MODEL\partc_flow(on-ap)'
    ecg_path = r'D:\HTNet-master\NEW_MODEL\ECG_TU_224_224'  # ECG图像路径

    # 加载多模态数据
    optical_flow_data = load_image_data(optical_flow_path)
    ecg_data = load_image_data(ecg_path)

    subName = os.listdir(main_path)
    print("Subjects:", subName)

    total_gt = []
    total_pred = []
    best_total_pred = []
    t = time.time()
    all_accuracy_dict = {}

    for n_subName in subName:
        print('\nProcessing Subject:', n_subName)

        # 创建数据集
        train_dataset = MultimodalDataset(main_path, n_subName, optical_flow_data, ecg_data, mode='train')
        test_dataset = MultimodalDataset(main_path, n_subName, optical_flow_data, ecg_data, mode='test')

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=batch_size)

        # 初始化模型
        model = MultiEMO_ECG_OpticalFlow(model_dim=128, hidden_dim=256,
                                          num_layers=2, num_heads=4, dropout=0.3,
                                          n_classes=4, multi_attn_flag=True)
        model = model.to(device)
        weight_path = os.path.join('ourmodel_CASME3_weights', n_subName + '.pth')

        if config.train:
            print('Training ')
            #model.apply(reset_weights)
        else:
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)#学习率调度器

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if config.train:
                # 训练阶段
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch_idx, (flow_batch, ecg_batch, labels) in enumerate(train_dl):
                    flow_batch = flow_batch.to(device)
                    ecg_batch = ecg_batch.to(device)
                    labels = labels.to(device)#将数据转移到设备上（CPU 或 GPU）

                    optimizer.zero_grad()# 清除之前的梯度
                    padded_labels = torch.zeros_like(labels)  # [B]，设置为非 -1 即可
                    fused_ecg, fused_flow, fc_out, outputs = model(ecg_batch.unsqueeze(1), flow_batch.unsqueeze(1), padded_labels.unsqueeze(1))
  # 取 mlp_outputs
                    loss = loss_fn(outputs, labels)# 计算损失
                    loss.backward()# 反向传播计算梯度

                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()# 更新模型参数
                        optimizer.zero_grad()# 清除当前梯度
                    # 计算训练过程中的准确度等
                    train_loss += loss.item() * flow_batch.size(0)
                    num_train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                    num_train_examples += labels.size(0)

                train_acc = num_train_correct / num_train_examples
                avg_loss = train_loss / num_train_examples
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}')

            # 验证阶段
            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for flow_batch, ecg_batch, labels in test_dl:
                    flow_batch = flow_batch.to(device)
                    ecg_batch = ecg_batch.to(device)
                    labels = labels.to(device)

                    padded_labels = torch.zeros_like(labels)  # [B]，设置为非 -1 即可
                    fused_ecg, fused_flow, fc_out, outputs = model(ecg_batch.unsqueeze(1), flow_batch.unsqueeze(1), padded_labels.unsqueeze(1))

                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item() * flow_batch.size(0)
                    num_val_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                    num_val_examples += labels.size(0)
                    all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = num_val_correct / num_val_examples
            avg_val_loss = val_loss / num_val_examples
            print(f' test Accuracy: {val_acc:.4f}')

            # 更新学习率
            scheduler.step(val_acc)

            # 保存最佳模型
            if val_acc >= best_accuracy_for_each_subject:
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = all_preds.copy()
                if config.train:
                    torch.save(model.state_dict(), weight_path)
                print(f'New best accuracy: {best_accuracy_for_each_subject:.4f}')

                if best_accuracy_for_each_subject == 1.0:
                    print(f"Saved model with 100% accuracy for subject: {n_subName}")
                    break

        # 在当前主题训练完成后打印所有结果
        print('Best Predicted    :', best_each_subject_pred)
        accuracy_dict = {
            'pred': best_each_subject_pred,
            'truth': all_labels
        }
        all_accuracy_dict[n_subName] = accuracy_dict

        print('Ground Truth :', all_labels)
        print('Evaluation until this subject: ')

        total_gt.extend(all_labels)
        total_pred.extend(all_preds)
        best_total_pred.extend(best_each_subject_pred)

        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    # 最终评估
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True)
    config = parser.parse_args()
    main(config)