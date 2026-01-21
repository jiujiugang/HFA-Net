from HFA_Net.model.repvit_re import repvit_m0_6
from HFA_Net.mul_block import MSA
from module.MLFA import MLFA
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from module.AAFF import AAFF
from model.swim_transformer import SwinTransformerFeatures
from model.Swim_mona import SwinTransformer_mona_features
from utils.tensor_shaper import tensor_shaper
class DualInception(nn.Module):
    def __init__(self, classes=4,
                 imgs=3,
                 input_type=None,
                 **kwargs):
        super(DualInception, self).__init__()
        self.input_type = input_type or "flow0+ECG"
        # 添加检查：确保 input_type 是 "optical"、"depths" 或 "delta" 中的一个
        assert self.input_type in ["flow0+ECG"], \
            f"错误的输入类型: {self.input_type}. 正确类型：'flow0+ECG'."
        # 如果存在额外的参数，可以通过 kwargs 访问
        self.imgs = kwargs.get('imgs', imgs)
        self.classes = kwargs.get('classes', classes)
        self.pretrained_weights_path = kwargs.get('pretrained_weights_path', None)
        self.subName = kwargs.get('subName', None)

        if self.pretrained_weights_path and self.subName:
            print("预训练权重路径", self.pretrained_weights_path)
            weight_path = f"{self.pretrained_weights_path}/{self.subName}.pth"
            self._initialize_modules()
            self._load_pretrained_weights(weight_path)
        else:
            print("未提供预训练权重，正在从零开始初始化模型")
            self._initialize_modules()

    def _initialize_modules(self):
        self.msa = MSA()
        # self.rmt = RMT_modify()
        #self.vis = repvit_m0_6()
        #self.swim = SwinTransformerFeatures()
        self.swimmona = SwinTransformer_mona_features()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlfa = MLFA(inter_dim=1024, level=4, channel=[128, 256, 512, 1024])
        self.aaff = AAFF(1024, 512)

        self.head = nn.Linear(512, self.classes) if self.classes > 0 else nn.Identity()


    def forward(self, x):
        x = tensor_shaper(x, self.imgs)
        if self.input_type == "flow0+ECG" and self.imgs == 3:
            x,y= x[0],x[2]  # 取出 光流

        else:
            raise ValueError(
                f"SB 的'{self.input_type}' 或 '{self.imgs}'错误.")

        """
        x: Input image for MSA module (e.g., from one path)
        y: Input image for SwinTransformer_mona_features module (e.g., from another path)
        """
        # Process the input for MSA
        m1, m2, m3, m4 = self.msa(y)  # m1, m2, m3, m4: Local branch output

        # Process the input for SwinTransformer_mona_features
        t1, t2, t3, t4 = self.swimmona(x)  # t1, t2, t3, t4: Outputs for another branch

        # Concatenate features from both paths
        mt1 = torch.cat((m1, t1), dim=1)
        mt2 = torch.cat((m2, t2), dim=1)
        mt3 = torch.cat((m3, t3), dim=1)
        mt4 = torch.cat((m4, t4), dim=1)

        # Multi-level feature aggregation (MLFA)
        mt = self.mlfa(mt1, mt2, mt3, mt4)

        # AAFF processing
        output = self.aaff(mt4, mt)

        # Adaptive pooling and final classification
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.head(output)

        return output

# 3通道 光流
def test(**kwargs):
    model = DualInception(
        input_type="flow0+ECG",

        **kwargs  # 将额外的参数传递给 SB
    )
    return model

if __name__ == '__main__':

    # config.imgs=3
    x = torch.randn(1, 3, 3, 224, 224) # delta

    # 创建模型时通过**kwargs传递额外的参数
    model_kwargs = {}
    classes=5
    if classes>0 :
        model_kwargs['classes'] = 5
        #model_kwargs['imgs'] = 2
    else:
        print("未传入classes参数，默认四分类任务")

    model = test(**model_kwargs)
    #print(model)
    x = model(x)
    print(f"output shape: {x.shape}")
