import torch
import torch.nn as nn
from NaiveTransformerExpr.EncoderMultiStream import Encoder as TransformerEncoder


def aug_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    std_features = torch.std(data, dim=aug_dim)
    max_features, _ = torch.max(data, dim=aug_dim)
    min_features, _ = torch.min(data, dim=aug_dim)
    union_feature = torch.cat((mean_features, std_features, min_features, max_features), dim=-1)
    return union_feature


def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features


class ModelSimilarity(nn.Module):
    def __init__(self, d_img, d_inner, layers, n_head, dropout, d_out, feature_aug, feature_compose, add_sa, num_class,
                 n_position=30):
        super(ModelSimilarity, self).__init__()
        self.feature_aug, self.feature_compose = feature_aug, feature_compose

        # 双流 Transformer 编码器 (ECG + 光流)
        self.tf_encoder_img1_img2 = TransformerEncoder(
            d_img, d_img, layers, d_inner, n_head, d_k=None, d_v=None, dropout=dropout, n_position=n_position,
            add_sa=add_sa
        )

        # 2D 卷积处理 ECG 和 光流图像
        self.conv_img1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_img2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.bn_img1 = nn.BatchNorm2d(32)
        self.bn_img2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 特征提取线性层
        self.fc_img1 = nn.Linear(32 * 224 * 224, d_out)
        self.fc_img2 = nn.Linear(32 * 224 * 224, d_out)

        # 分类器
        if self.feature_aug == 'aug':
            fc_dimension = d_out * 4
        elif self.feature_aug == 'mean':
            fc_dimension = d_out
        else:
            raise NotImplementedError

        if self.feature_compose == 'cat':
            fc_dimension = fc_dimension * 2
        elif self.feature_compose == 'mean' or self.feature_compose == 'sum':
            fc_dimension = fc_dimension
        else:
            raise NotImplementedError

        self.classifier = nn.Sequential(
            nn.Linear(fc_dimension, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_class)
        )

    def forward(self, img1, img2):
        # 图像流处理 (ECG 和 光流)
        img1 = self.relu(self.bn_img1(self.conv_img1(img1)))
        img2 = self.relu(self.bn_img2(self.conv_img2(img2)))

        # 展平并通过线性层
        img1 = img1.view(img1.size(0), -1)
        img2 = img2.view(img2.size(0), -1)
        img1 = self.fc_img1(img1)
        img2 = self.fc_img2(img2)

        # Transformer 编码器
        img1, img2 = self.tf_encoder_img1_img2(img1.unsqueeze(1), img2.unsqueeze(1))

        # 特征增强
        if self.feature_aug == 'aug':
            img1, img2 = aug_temporal(img1, 1), aug_temporal(img2, 1)
        elif self.feature_aug == 'mean':
            img1, img2 = mean_temporal(img1, 1), mean_temporal(img2, 1)

        # 特征组合
        if self.feature_compose == 'sum':
            features = img1 + img2
        elif self.feature_compose == 'cat':
            features = torch.cat([img1, img2], dim=-1)
        elif self.feature_compose == 'mean':
            features = (img1 + img2) / 2

        # 分类
        result = self.classifier(features)
        return result


if __name__ == '__main__':
    model = ModelSimilarity(
        d_img=32, d_inner=512, layers=2, n_head=4, dropout=0.5, d_out=128,
        feature_aug='mean', feature_compose='cat', num_class=4, add_sa=True
    )

    # 测试输入：ECG + 光流，形状为 (B, 3, 224, 224)
    ecg = torch.randn(8, 3, 224, 224)
    flow = torch.randn(8, 3, 224, 224)
    result = model(ecg, flow)
    print(result.shape)
