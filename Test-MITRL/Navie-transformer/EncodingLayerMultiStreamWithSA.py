import torch
import torch.nn as nn
from .MultiHeadAttentionMultiStream import MultiHeadAttention
from .PositionwiseFeedForward import PositionwiseFeedForward


import torch
import torch.nn as nn
from .MultiHeadAttentionMultiStream import MultiHeadAttention
from .PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    双流 + 自注意力的 Transformer 编码层
      - Cross‑Attention : Q(主流) 对 K/V(辅流)
      - Self‑Attention  : Q 流自身
      - Feed‑Forward    : 逐位置前馈
    """
    def __init__(self,
                 n_head: int,
                 d_emb_q: int,
                 d_emb_v: int,
                 d_inner: int,
                 d_k: int = None,
                 d_v: int = None,
                 dropout: float = 0.1):
        super().__init__()

        # 若未显式指定 d_k / d_v，直接使用嵌入维度，保证能被 n_head 整除
        d_k = d_k or d_emb_q
        d_v = d_v or d_emb_v

        # ① Cross‑Attention (双流)
        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_emb_q=d_emb_q,
            d_emb_v=d_emb_v,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        # ② Self‑Attention (Query 自身)
        self.slf_attn_sa = MultiHeadAttention(
            n_head=n_head,
            d_emb_q=d_emb_q,
            d_emb_v=d_emb_q,   # Self: K/V 与 Q 同维
            d_k=d_k,
            d_v=d_k,
            dropout=dropout
        )

        # ③ Position‑wise Feed‑Forward
        self.pos_ffn = PositionwiseFeedForward(d_emb_q, d_inner, dropout=dropout)

        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_emb_q, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_emb_q, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                slf_attn_mask1: torch.Tensor = None,
                slf_attn_mask2: torch.Tensor = None):
        """
        Args:
            q: (B, L_q, D_q)  主流 (e.g. optical flow)
            k: (B, L_k, D_v)  辅流
            v: (B, L_v, D_v)
            slf_attn_mask1: (B, L_q, L_q)  Q 流 mask
            slf_attn_mask2: (B, L_k, L_k)  K/V 流 mask
        """
        # ---- 1) Cross‑Attention ----
        res = q
        q_norm = self.norm1(q)
        attn_out, attn_map = self.slf_attn(q_norm, k, v,
                                           mask1=slf_attn_mask1,
                                           mask2=slf_attn_mask2)
        q = res + self.dropout(attn_out)

        # ---- 2) Self‑Attention ----
        res = q
        self_out, _ = self.slf_attn_sa(q, q, q, mask1=slf_attn_mask1)
        q = res + self.dropout(self_out)

        # ---- 3) Position‑wise FFN ----
        res = q
        ff_out = self.pos_ffn(self.norm2(q))
        q = res + self.dropout(ff_out)

        return q, attn_map

