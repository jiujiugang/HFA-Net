from HFA_Net.model.repvit_re import repvit_m0_6
from HFA_Net.mul_block import MSA
from module.MLFA import MLFA
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from module.AAFF import AAFF
from model.swim_transformer import SwinTransformerFeatures
from model.Swim_mona import SwinTransformer_mona_features
import torch.nn.functional as F
class CustomLoss(nn.Module):
    def __init__(self, lambda_swim=1.0, lambda_msa=1.0, lambda_mlfa=1.0,
                 lambda_swim_factor=1.0, lambda_msa_factor=1.0):
        super(CustomLoss, self).__init__()
        self.lambda_swim = lambda_swim  # 光流支路损失的权重
        self.lambda_msa = lambda_msa  # ECG支路损失的权重
        self.lambda_mlfa = lambda_mlfa  # 融合特征的损失权重
        self.lambda_swim_factor = lambda_swim_factor  # 光流支路的系数
        self.lambda_msa_factor = lambda_msa_factor  # ECG支路的系数
        self.ce_loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, outputs, labels, m1, m2, m3, m4, t1, t2, t3, t4, fused_features):
        # 1. 计算光流支路的损失，使用 t4 作为光流支路的输出
        swim_loss = self.ce_loss(t4, labels)  # 使用 t4（光流支路的输出）计算损失

        # 2. 计算ECG支路的损失，使用 m4 作为ECG支路的输出
        msa_loss = self.ce_loss(m4, labels)  # 使用 m4（ECG支路的输出）计算损失

        # 3. 计算融合特征的损失，使用融合后的特征计算损失
        mlfa_loss = 0.0
        for stage_output in [m1, m2, m3, m4]:  # 在不同阶段计算损失并平均化
            mlfa_loss += self.ce_loss(stage_output, labels)
        mlfa_loss /= 4  # 平均化各阶段损失

        # 4. 计算总损失，应用各部分损失的加权和
        total_loss = (self.lambda_swim_factor * self.lambda_swim * swim_loss) + \
                     (self.lambda_msa_factor * self.lambda_msa * msa_loss) + \
                     (self.lambda_mlfa * mlfa_loss)

        return total_loss


class DualInception(nn.Module):#损失函数
    def __init__(self, num_classes =4):
        super(DualInception, self).__init__()
        self.msa = MSA()
        #self.rmt = RMT_modify()
        self.vis = repvit_m0_6()
        self.swim = SwinTransformerFeatures()
        self.swimmona = SwinTransformer_mona_features()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlfa = MLFA(inter_dim=1024, level=4, channel=[128, 256, 512,1024])
        self.aaff = AAFF(1024, 512)

        self.head = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
        self.custom_loss = CustomLoss(lambda_swim_factor=0.5, lambda_msa_factor=0.5)  # 设置初始系数

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 定义了一个辅助方法 _init_weights，用于初始化模型的权重。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x, y, ):


        m1, m2, m3, m4 = self.msa(y)# m1, m2, m3, m4: Local branch 各 stage 输出
        #print(f"m1 shape: {m1.shape}")

        #print("m1:",m1)
        #t1, t2, t3, t4 = self.rmt(x)# t1, t2, t3, t4 = self.rmt(x)#原始网络
        t1, t2, t3, t4 = self.swimmona(x)
        #print(f"t1 shape: {t1.shape}")



        mt1 = torch.cat((m1, t1), dim=1)#特征拼接
        mt2 = torch.cat((m2 ,t2) ,dim=1)
        mt3 = torch.cat((m3, t3), dim=1)
        mt4 = torch.cat((m4, t4), dim=1)
        mt = self.mlfa(mt1, mt2, mt3, mt4) # 将 4 个 stage 的 mt1~mt4 输入到 MLFA
        output = self.aaff(mt4, mt) # AAFF 接收 mt4（高层次特征） 和 mt（融合的全尺度特征）
        output = self.avgpool(output)  # B C 1
        output = torch.flatten(output, 1)
        output = self.head(output)
        """
        if labels is not None:
            print(f"output shape: {output.shape}, labels shape: {labels.shape}")  # 检查形状
            loss = self.custom_loss(output, labels, m1, m2, m3, m4, t1, t2, t3, t4, mt)
            return output, loss
         """
        return output

    # Assuming the necessary packages and classes are correctly imported

if __name__ == "__main__":
    # Instantiate the model
    model = DualInception(num_classes=3)

    # Create a sample input tensor (batch_size, channels, height, width)
    input_tensor = torch.rand(64, 3, 224, 224)  # Example input (8 samples, 3 channels, 224x224 images)

    # Forward pass
    output = model(input_tensor)

    # Print the output shape
    print(f"Output shape: {output.shape}")  # Should be [batch_size, num_classes] -> [8, 4]