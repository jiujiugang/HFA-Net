import torch
import torch.nn as nn
import torchvision.models as models
from MLP import MLP  # 你已有的模块
from MultiAttn import MultiAttnModel  # 你已有的模块


class MultiEMO_ECG_OpticalFlow(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_layers, num_heads, dropout,
                 n_classes, multi_attn_flag=True):
        super().__init__()
        self.multi_attn_flag = multi_attn_flag

        # ECG 图像特征提取器
        resnet_ecg = models.resnet18(pretrained=True)
        resnet_ecg.fc = nn.Identity()
        self.ecg_cnn = nn.Sequential(
            resnet_ecg,
            nn.Linear(512, model_dim)
        )

        # 光流图像特征提取器
        resnet_flow = models.resnet18(pretrained=True)
        resnet_flow.fc = nn.Identity()
        self.flow_cnn = nn.Sequential(
            resnet_flow,
            nn.Linear(512, model_dim)
        )

        # 多模态注意力融合模块
        self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout)

        # 融合后投影层
        self.fc = nn.Linear(model_dim * 2, model_dim)

        # 最终分类器
        self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)

    def forward(self, ecg_images, flow_images, padded_labels):
        """
        Args:
            ecg_images: Tensor [B, T, 3, 224, 224]
            flow_images: Tensor [B, T, 3, 224, 224]
            padded_labels: Tensor [B, T]
        """
        B, T, C, H, W = ecg_images.size()

        # 合并 B*T -> [B*T, 3, 224, 224]
        ecg_images = ecg_images.view(B * T, C, H, W)
        flow_images = flow_images.view(B * T, C, H, W)

        # 特征提取后恢复 [B, T, D]
        ecg_features = self.ecg_cnn(ecg_images).view(B, T, -1)
        flow_features = self.flow_cnn(flow_images).view(B, T, -1)

        # 转换为 [T, B, D] 以适配注意力机制
        ecg_features = ecg_features.transpose(0, 1)
        flow_features = flow_features.transpose(0, 1)

        # 多模态注意力融合
        if self.multi_attn_flag:
            fused_ecg, fused_flow = self.multiattn(ecg_features, flow_features)
        else:
            fused_ecg, fused_flow = ecg_features, flow_features

        # 拉平后掩码掉 padded 的样本
        fused_ecg = fused_ecg.reshape(-1, fused_ecg.shape[-1])
        fused_flow = fused_flow.reshape(-1, fused_flow.shape[-1])
        valid_mask = padded_labels.view(-1) != -1  # 修正掩码维度

        fused_ecg = fused_ecg[valid_mask]
        fused_flow = fused_flow[valid_mask]

        # 拼接后送入分类网络
        fused = torch.cat((fused_ecg, fused_flow), dim=-1)
        fc_out = self.fc(fused)
        mlp_out = self.mlp(fc_out)

        return fused_ecg, fused_flow, fc_out, mlp_out
