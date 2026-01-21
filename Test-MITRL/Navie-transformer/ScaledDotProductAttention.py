import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask1=None, mask2=None):
        """
        q, k, v: (B, n_head, L_q/k/v, d_k/d_v)
        mask*:   bool, True 表示要 mask 的位置
        """
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))  # (B, n_head, L_q, L_k)

        # 独立处理两个 mask，允许只传其中之一
        if mask1 is not None:
            attn = attn.masked_fill(mask1, float('-inf'))
        if mask2 is not None:
            attn = attn.masked_fill(mask2, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)  # (B, n_head, L_q, d_v)
        return output, attn



if __name__ == '__main__':
    batch_size, n_head = 16, 8
    n_hid_k, n_hid_v = 128, 200
    seq_len_q, seq_len_k = 10, 15
    q_ = torch.randn(batch_size, n_head, seq_len_q, n_hid_k)
    k_ = torch.randn(batch_size, n_head, seq_len_k, n_hid_k)
    v_ = torch.randn(batch_size, n_head, seq_len_k, n_hid_v)

    # 示例掩码 (B, 1, L_q, L_k)
    mask1 = torch.ones(batch_size, 1, seq_len_q, seq_len_k).bool()
    mask2 = torch.ones(batch_size, 1, seq_len_q, seq_len_k).bool()

    scaledDotProductAttention = ScaledDotProductAttention(temperature=1)
    result, attention = scaledDotProductAttention(q_, k_, v_, mask1, mask2)
    print("Result:", result.shape)
    print("Attention:", attention.shape)
