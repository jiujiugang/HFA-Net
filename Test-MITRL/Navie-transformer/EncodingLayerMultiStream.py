import torch
import torch.nn as nn
from .MultiHeadAttentionMultiStream import MultiHeadAttention
from .PositionwiseFeedForward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    双流 Transformer 编码层：
    - Q (Query): 主流输入
    - K (Key): 辅流输入
    - V (Value): 辅流输入
    """
    def __init__(self, n_head, d_emb_q, d_emb_v, d_inner, d_k=None, d_v=None, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 多头注意力层，双流输入 (Q, K, V)
        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_emb_q=d_emb_q,
            d_emb_v=d_emb_v,
            d_k=d_k or d_emb_q // n_head,
            d_v=d_v or d_emb_v // n_head,
            dropout=dropout
        )

        # 位置前馈神经网络 (Feed-Forward Network)
        self.pos_ffn = PositionwiseFeedForward(d_emb_q, d_inner, dropout=dropout)

        # 层归一化和 Dropout
        self.layer_norm1 = nn.LayerNorm(d_emb_q, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_emb_q, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, slf_attn_mask1=None, slf_attn_mask2=None):
        """
        前向传播：
        - q: Query 序列 (B, L_q, D_q)
        - k: Key 序列 (B, L_k, D_v)
        - v: Value 序列 (B, L_v, D_v)
        - slf_attn_mask1: Q 流的掩码 (B, L_q, L_q)
        - slf_attn_mask2: K, V 流的掩码 (B, L_k, L_k)
        """
        # 残差连接 + LayerNorm (注意力层)
        residual = q
        q = self.layer_norm1(q)
        attn_output, attn_map = self.slf_attn(q, k, v, mask1=slf_attn_mask1, mask2=slf_attn_mask2)
        attn_output = self.dropout(attn_output) + residual

        # 残差连接 + LayerNorm (前馈层)
        residual = attn_output
        attn_output = self.layer_norm2(attn_output)
        ff_output = self.pos_ffn(attn_output)
        ff_output = self.dropout(ff_output) + residual

        return ff_output, attn_map
