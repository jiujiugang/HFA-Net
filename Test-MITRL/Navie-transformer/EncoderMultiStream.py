from torch import nn
import torch

from .EncodingLayerMultiStream import EncoderLayer as EncoderLayerNoSA
from .EncodingLayerMultiStreamWithSA import EncoderLayer as EncoderLayerAddSA
from .PositionEncoding import PositionEncoding


class Encoder(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, n_layers, d_inner, n_head, d_k=None, d_v=None, dropout=0.1, n_position=2048,
                 add_sa=False):
        super(Encoder, self).__init__()

        # 可选的自注意力 Transformer 编码器（带或不带SA）
        self.position_enc1 = PositionEncoding(d_emb_1, n_position=n_position, )
        self.position_enc2 = PositionEncoding(d_emb_2, n_position=n_position, )
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_emb_1, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_emb_2, eps=1e-6)

        if not add_sa:
            self.layer_stack1 = nn.ModuleList([
                EncoderLayerNoSA(n_head, d_emb_1, d_emb_2, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.layer_stack2 = nn.ModuleList([
                EncoderLayerNoSA(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            self.layer_stack1 = nn.ModuleList([
                EncoderLayerAddSA(n_head, d_emb_1, d_emb_2, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.layer_stack2 = nn.ModuleList([
                EncoderLayerAddSA(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ])

    def forward(self, seq1, seq2, src_mask1=None, src_mask2=None, return_attns=False):
        """
        seq1: (B, L1, D1)
        seq2: (B, L2, D2)
        """
        enc_slf_attn_list1, enc_slf_attn_list2 = [], []

        # 位置编码 + Dropout + LayerNorm
        seq1 = self.layer_norm1(self.dropout1(self.position_enc1(seq1)))
        seq2 = self.layer_norm2(self.dropout2(self.position_enc2(seq2)))

        # 确保没有 NaN 值
        seq1 = seq1.masked_fill(torch.isnan(seq1), 0.0)
        seq2 = seq2.masked_fill(torch.isnan(seq2), 0.0)

        # Transformer 层叠
        for enc_layer1, enc_layer2 in zip(self.layer_stack1, self.layer_stack2):
            temp_seq1, temp_seq2 = seq1, seq2
            seq1, attn1 = enc_layer1(temp_seq1, temp_seq2, temp_seq2, slf_attn_mask1=src_mask1,
                                     slf_attn_mask2=src_mask2)
            seq2, attn2 = enc_layer2(temp_seq2, temp_seq1, temp_seq1, slf_attn_mask1=src_mask2,
                                     slf_attn_mask2=src_mask1)

            if return_attns:
                enc_slf_attn_list1.append(attn1)
                enc_slf_attn_list2.append(attn2)

        if return_attns:
            return seq1, seq2, enc_slf_attn_list1, enc_slf_attn_list2
        return seq1, seq2


def make_mask(feature):
    """
    生成掩码：假设全零位置为掩码（掩码为 True）
    """
    return torch.sum(torch.abs(feature), dim=-1) == 0


if __name__ == '__main__':
    # 测试双流 Transformer 编码器
    encoder = Encoder(d_emb_1=128, d_emb_2=128, n_layers=2, d_inner=512, n_head=4, dropout=0.1, add_sa=True)

    # 模拟的双流输入 (ECG + 光流)
    ecg = torch.randn(4, 32, 128)  # (B, L, D)
    flow = torch.randn(4, 32, 128)  # (B, L, D)

    # 掩码 (可选)
    src_mask1 = make_mask(ecg)
    src_mask2 = make_mask(flow)

    # 前向传播
    output1, output2 = encoder(ecg, flow, src_mask1=src_mask1, src_mask2=src_mask2)
    print("Output Shape (Stream 1):", output1.shape)
    print("Output Shape (Stream 2):", output2.shape)
