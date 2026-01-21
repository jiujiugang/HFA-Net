import os
import sys
import torch
import numpy as np
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from distutils.util import strtobool
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from PIL import Image
import cv2
import time

import warnings  # 新增
# ======== 警告过滤设置 ========
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="pandas")
warnings.filterwarnings("ignore", module="mmcv")
warnings.filterwarnings("ignore", message="torch.meshgrid")
# ============================
# 屏蔽特定的警告信息
warnings.filterwarnings("ignore", message="If for semantic segmentation, please install mmsegmentation first")
warnings.filterwarnings("ignore", message="If for detection, please install mmdetection first")
# 重定向标准输出和标准错误输出
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
from model_dual_eda_flow import DualInception
from att_mask import extract_face_landmarks, generate_attention_mask
# 恢复标准输出和标准错误输出
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
warnings.filterwarnings("ignore")
class MultimodalDataset(Dataset):
    def __init__(self, main_path, subfolder, optical_flow_data, ecg_data, mode='train', transform=None):
        """
        自定义多模态数据集类
        Args:
            main_path: 主数据路径
            subfolder: 子文件夹名（被试者名）
            optical_flow_data: 光流数据字典
            ecg_data: ECG数据字典
            mode: 'train' 或 'test'
            transform: 图像变换
        """
        self.optical_flow_data = optical_flow_data
        self.ecg_data = ecg_data
        self.transform = transform
        self.labels = []
        self.flow_samples = []
        self.ecg_samples = []

        # 加载数据 遍历 u_train/u_test 下的子目录
        data_path = os.path.join(main_path, subfolder, f'u_{mode}')
        for n_expression in os.listdir(data_path):  # 遍历表情类别文件夹
            expr_path = os.path.join(data_path, n_expression)
            for n_img in os.listdir(expr_path):  # 遍历每个图像文件
                if n_img in optical_flow_data and n_img in ecg_data:  # 检查双模态数据都存在
                    self.labels.append(int(n_expression))  # 表情标签
                    self.flow_samples.append(optical_flow_data[n_img])  # 光流数据
                    self.ecg_samples.append(ecg_data[n_img])  # ECG数据

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取光流图像并应用变换
        flow_img = self.flow_samples[idx]
        if self.transform:
            flow_img = self.transform(Image.fromarray(flow_img))

        # 获取ECG图像并应用变换
        ecg_img = self.ecg_samples[idx]
        if self.transform:
            ecg_img = self.transform(Image.fromarray(ecg_img))

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return flow_img, ecg_img, label


def load_image_data(data_path):
    """加载图像数据并返回字典"""
    data_dict = {}
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            data_dict[img_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    return data_dict


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def confusionMatrix(gt, pred):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples if num_samples > 0 else 0
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred):
    label_dict = {'negative': 0, 'others': 1, 'positive': 2, 'surprise': 3}
    f1_list = []
    ar_list = []
    for emotion, emotion_index in label_dict.items():
        gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
        pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
        try:
            f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
            f1_list.append(f1_recog)
            ar_list.append(ar_recog)
        except Exception:
            pass
    UF1 = np.mean(f1_list)
    UAR = np.mean(ar_list)
    return UF1, UAR

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', beta=1.0, class_counts=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.beta = beta
        self.class_counts = class_counts

        # 计算alpha权重
        if alpha is not None:
            self.alpha = alpha
        elif class_counts is not None:
            total_samples = sum(class_counts)
            # 方法1：逆频率权重
            self.alpha = [total_samples / count for count in class_counts]
            # 或者方法3：基于beta的权重
            # self.alpha = [1 / (1 + beta * (count / total_samples)) for count in class_counts]
            self.alpha = torch.tensor(self.alpha).float()
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(self.alpha).to(inputs.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True, help='Train or use pre-trained weight for prediction')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--flow-data-path', type=str, default=r'D:\HTNet-master\NEW_MODEL\partc_flow(of-ap)',
                        help='Path to the flow dataset')
    parser.add_argument('--ecg-data-path', type=str, default=r'D:\HTNet-master\NEW_MODEL\ECG_TU_224_224',
                        help='Path to the ECG dataset')
    parser.add_argument('--loso-data-path', type=str, default=r'D:\HTNet-master\NEW_MODEL\LOSO\on_ap',
                        help='Path for leave-one-subject-out cross-validation')
    config = parser.parse_args()
    return config


def main(config):
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    epochs = config.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Device: {device}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载多模态数据
    optical_flow_data = load_image_data(config.flow_data_path)
    ecg_data = load_image_data(config.ecg_data_path)

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}
    t = time.time()

    loso_path = config.loso_data_path
    subNames = os.listdir(loso_path)
    print(f"Subjects: {subNames}")

    for subName in subNames:
        print(f'\nSubject: {subName}')

        # 创建多模态数据集
        train_dataset = MultimodalDataset(loso_path, subName, optical_flow_data, ecg_data, mode='train',
                                          transform=transform)
        test_dataset = MultimodalDataset(loso_path, subName, optical_flow_data, ecg_data, mode='test',
                                         transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        weight_path = f'ourmodel_threedatasets_weights/{subName}.pth'
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)

        # 初始化模型 - 需要修改为支持双模态输入的模型
        model = DualInception(num_classes=4).to(device)  # 需要确保模型能处理双输入

        if config.train:
            print('Training mode enabled')
        else:
            print('Loading pre-trained weights')
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        class_counts = [99, 20, 16, 30]
        total_samples = sum(class_counts)
        alpha_weights = [total_samples / count for count in class_counts]

        loss_fn = FocalLoss(
            alpha=alpha_weights,  # 传入新权重
            gamma=2,
            reduction='mean',
        )

        best_accuracy = 0
        best_predictions = []

        for epoch in range(1, epochs + 1):
            if config.train:
                model.train()
                train_loss, correct, total = 0.0, 0, 0

                for flow_batch, ecg_batch, labels in train_loader:
                    flow_batch, ecg_batch = flow_batch.to(device), ecg_batch.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    # 修改模型前向传播以接受双输入
                    outputs = model(x=flow_batch, y=ecg_batch)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * flow_batch.size(0)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)

                train_accuracy = correct / total
                print(
                    f"Epoch {epoch}/{epochs}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Accuracy: {train_accuracy:.4f}")

            # 验证阶段
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            predictions = []

            with torch.no_grad():
                for flow_batch, ecg_batch, labels in test_loader:
                    flow_batch, ecg_batch = flow_batch.to(device), ecg_batch.to(device)
                    labels = labels.to(device)

                    outputs = model(x=flow_batch, y=ecg_batch)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item() * flow_batch.size(0)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
                    predictions.extend(outputs.argmax(1).cpu().tolist())

            val_accuracy = correct / total
            print(
                f"Epoch {epoch}/{epochs}, Val Loss: {val_loss / len(test_loader.dataset):.4f}, Val Accuracy: {val_accuracy:.4f}")

            # 更新学习率
            scheduler.step(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_predictions = predictions.copy()
                if config.train:
                    torch.save(model.state_dict(), weight_path)

        # 记录当前被试结果
        total_gt.extend(test_dataset.labels)
        total_pred.extend(predictions)
        best_total_pred.extend(best_predictions)

        accuracy_dict = {'pred': best_predictions, 'truth': test_dataset.labels}
        all_accuracy_dict[subName] = accuracy_dict

        print(f"Subject {subName} Best Accuracy: {best_accuracy:.4f}")
        print('Best Predicted:', best_predictions)
        print('Ground Truth:', test_dataset.labels)

    # 最终评估
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred)

    print(f"\nTotal Time: {time.time() - t:.2f}s")
    print(f"Final UF1: {UF1:.4f}, Final UAR: {UAR:.4f}")
    print(f"Best UF1: {best_UF1:.4f}, Best UAR: {best_UAR:.4f}")
    print('All Accuracy Dict:', all_accuracy_dict)




if __name__ == '__main__':
    config = parse_option()
    main(config)