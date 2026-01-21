import torch
import torch.nn as nn
import torch.nn.functional as F
#from TFN import TFN  # 如果不使用可以注释掉
from .ScaledDotProductAttention import ScaledDotProductAttention

class FBP(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, fbp_hid, fbp_k, dropout):
        super(FBP, self).__init__()
        # 原始的融合矩阵 + 池化结构不变
        self.fusion_1_matrix = nn.Linear(d_emb_1, fbp_hid * fbp_k, bias=False)
        self.fusion_2_matrix = nn.Linear(d_emb_2, fbp_hid * fbp_k, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_pooling = nn.AvgPool1d(kernel_size=fbp_k)
        self.fbp_k = fbp_k

    def forward(self, seq1, seq2):
        """
        seq1: (B, D_k)
        seq2: (B, D_k)
        returns: (B, fbp_hid)
        """
        # 融合
        x1 = self.fusion_1_matrix(seq1)  # (B, fbp_hid*fbp_k)
        x2 = self.fusion_2_matrix(seq2)  # (B, fbp_hid*fbp_k)
        fused = x1 * x2                  # (B, fbp_hid*fbp_k)
        fused = fused.unsqueeze(1) if fused.dim() == 2 else fused  # (B,1,fbp_hid*fbp_k)
        fused = self.fusion_dropout(fused)
        fused = self.fusion_pooling(fused).squeeze(1) * self.fbp_k  # (B, fbp_hid)
        fused = F.normalize(fused, p=2, dim=-1)
        return fused  # (B, fbp_hid)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_emb_q, d_emb_v, d_k=None, d_v=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_emb_q = d_emb_q
        self.d_emb_v = d_emb_v
        self.d_k = d_k or d_emb_q
        self.d_v = d_v or d_emb_v

        assert self.d_k % n_head == 0, "d_k must be divisible by n_head"
        assert self.d_v % n_head == 0, "d_v must be divisible by n_head"

        # 线性映射
        self.w_q = nn.Linear(d_emb_q, self.d_k, bias=False)
        self.w_k = nn.Linear(d_emb_v, self.d_k, bias=False)
        self.w_v = nn.Linear(d_emb_v, self.d_v, bias=False)
        self.fc = nn.Linear(self.d_v, d_emb_q, bias=False)

        # FBP 融合门控
        self.fbp = FBP(self.d_k, self.d_k, fbp_hid=32, fbp_k=2, dropout=dropout)
        self.fc_gate = nn.Linear(32, 1)
        self.gate_activate = nn.Tanh()

        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_emb_q, eps=1e-6)

    def forward(self, q, k, v, mask1=None, mask2=None):
        """
        q: (B, L_q, D_q)
        k: (B, L_k, D_v)
        v: (B, L_v, D_v)
        mask1: (B, L_q, L_q) or None
        mask2: (B, L_k, L_k) or None
        returns:
          - output: (B, L_q, D_q)
          - attn:  (B, n_head, L_q, L_k)
        """
        B, L_q, _ = q.size()
        L_k = k.size(1)
        L_v = v.size(1)
        residual = q

        # 线性投影并分头
        q_proj = self.w_q(q).view(B, L_q, self.n_head, self.d_k // self.n_head)
        k_proj = self.w_k(k).view(B, L_k, self.n_head, self.d_k // self.n_head)
        v_proj = self.w_v(v).view(B, L_v, self.n_head, self.d_v // self.n_head)

        # 计算门控
        q_flat = q_proj.view(B, L_q, self.d_k)
        k_flat = k_proj.view(B, L_k, self.d_k)
        gate = self.fbp(q_flat.mean(1), k_flat.mean(1))           # (B, 32)
        gate = self.gate_activate(self.fc_gate(gate))            # (B, 1)
        gate = (torch.sign(gate) + gate.abs()) / 2.              # 二值化门控

        # 转换为 (B, n_head, L, d_head)
        qh = q_proj.transpose(1, 2)  # (B, n_head, L_q, d_k/n_head)
        kh = k_proj.transpose(1, 2)  # (B, n_head, L_k, d_k/n_head)
        vh = v_proj.transpose(1, 2)  # (B, n_head, L_v, d_v/n_head)

        # 广播掩码
        if mask1 is not None:
            mask1 = mask1.unsqueeze(1).unsqueeze(-1)  # (B,1,L_q,1)
        if mask2 is not None:
            mask2 = mask2.unsqueeze(1).unsqueeze(2)   # (B,1,1,L_k)

        # 缩放点积注意力
        attn_out, attn_map = self.attention(qh, kh, vh, mask1=mask1, mask2=mask2)
        # 合并头部
        out = attn_out.transpose(1, 2).contiguous().view(B, L_q, self.d_v)  # (B, L_q, D_v)

        # 最终线性 + 门控 + 残差 + LayerNorm
        out = self.dropout(self.fc(out))                                      # (B, L_q, D_q)
        out = out * gate.unsqueeze(-1)                                        # 应用门控
        out = out + residual                                                  # 残差
        out = self.layer_norm(out)                                            # 归一化
        out = out.masked_fill(torch.isnan(out), 0.0)

        return out, attn_map
