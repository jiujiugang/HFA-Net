import os
import torch
import numpy as np
import argparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from distutils.util import strtobool
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from PIL import Image
from model_dual_eda_flow import DualInception
from att_mask import extract_face_landmarks, generate_attention_mask


def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# 混淆矩阵计算函数
def confusionMatrix(gt, pred):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples if num_samples > 0 else 0
    return f1_score, average_recall

# 模型评估函数
def recognition_evaluation(final_gt, final_pred):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
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


# 加载数据集
def load_dataset(data_path, transform):
    images = []
    labels = []
    classes = os.listdir(data_path)
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
                labels.append(int(cls))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        if self.verbose:
            print(f"Validation loss decreased, saving model...")
        self.best_model_wts = model.state_dict()

    def load_checkpoint(self, model):
        model.load_state_dict(self.best_model_wts)


# 焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', beta=1.0, class_counts=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.beta = beta
        self.class_counts = class_counts
        if alpha is not None:
            self.alpha = alpha
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
            if self.class_counts is None:
                raise ValueError("class_counts must be provided if alpha is not provided.")
            total_samples = sum(self.class_counts)
            weights = 1 / (1 + self.beta * (torch.tensor(self.class_counts).to(inputs.device) / total_samples))
            alpha_t = weights[targets]

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
    parser.add_argument('--flow-data-path', type=str, default=r'D:\HTNet-master\NEW_MODEL\partc_flow(on-ap)', help='Path to the flow dataset for model input')
    parser.add_argument('--loso-data-path', type=str, default=r'D:\ZP\HFA\data\fuse_PH_LOSO', help='Path to the dataset for leave-one-subject-out cross-validation')
    config = parser.parse_args()
    return config


def main(config):
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    epochs = config.epochs

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if is_cuda else torch.device('cpu')

    print(f'Learning rate: {learning_rate}, Epochs: {epochs}, Device: {device}')

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}

    loso_path = config.loso_data_path
    subNames = os.listdir(loso_path)

    transform = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for subName in subNames:
        print(f'Subject: {subName}')

        train_path = os.path.join(loso_path, subName, 'u_train')
        test_path = os.path.join(loso_path, subName, 'u_test')

        train_images, train_labels = load_dataset(train_path, transform)
        test_images, test_labels = load_dataset(test_path, transform)

        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        weight_path = f'ourmodel_threedatasets_weights/{subName}.pth'

        os.makedirs(os.path.dirname(weight_path), exist_ok=True)

        # 初始化模型
        model =DualInception().to(device)

        if config.train:
            print('Training mode enabled')
        else:
            print('Loading pre-trained weights')
            model.load_state_dict(torch.load(weight_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        class_weights = [0.158, 0.362, 0.479]
        loss_fn = FocalLoss(alpha=class_weights, gamma=2, reduction='mean', beta=1.0)

        best_accuracy = 0
        best_predictions = []

        for epoch in range(1, epochs + 1):
            if config.train:
                model.train()
                train_loss, correct, total = 0.0, 0, 0

                for x, y in train_loader:
                    optimizer.zero_grad()
                    x, y = x.to(device), y.to(device)

                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x.size(0)
                    correct += (y_pred.argmax(1) == y).sum().item()
                    total += y.size(0)

                train_accuracy = correct / total
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Accuracy: {train_accuracy:.4f}")

            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            predictions = []

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)

                    val_loss += loss.item() * x.size(0)
                    correct += (y_pred.argmax(1) == y).sum().item()
                    total += y.size(0)
                    predictions.extend(y_pred.argmax(1).cpu().tolist())

            val_accuracy = correct / total
            print(f"Epoch {epoch}/{epochs}, Val Loss: {val_loss / len(test_loader.dataset):.4f}, Val Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_predictions = predictions
                if config.train:
                    torch.save(model.state_dict(), weight_path)

        total_gt.extend(test_labels.tolist())
        total_pred.extend(predictions)
        best_total_pred.extend(best_predictions)

    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred)

    print(f"Final UF1: {UF1:.4f}, Final UAR: {UAR:.4f}")
    print(f"Best UF1: {best_UF1:.4f}, Best UAR: {best_UAR:.4f}")


if __name__ == '__main__':
    config = parse_option()
    main(config)
