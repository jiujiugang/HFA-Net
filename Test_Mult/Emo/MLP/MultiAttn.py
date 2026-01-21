import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Bidirectional cross-attention layer.
'''
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()
        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)

    def bidirectional_scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1] ** 0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim=-1), V)
        return attention

    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        return self.bidirectional_scaled_dot_product_attention(Q, K, V)


'''
Multi-head bidirectional cross-attention.
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim)
            for _ in range(num_heads)
        ])
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)

    def forward(self, query, key, value):
        heads = [head(query, key, value) for head in self.attention_heads]
        multihead_output = torch.cat(heads, dim=-1)
        return self.projection_matrix(multihead_output)


'''
Feedforward layer with residual and dropout.
'''
class Feedforward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))


'''
Residual connection + LayerNorm.
'''
class AddNorm(nn.Module):
    def __init__(self, model_dim, dropout_rate):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


'''
A single layer of multimodal attention fusion.
'''
class MultiAttnLayer(nn.Module):
    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()
        QKV_dim = model_dim // num_heads
        self.attn_1 = MultiHeadAttention(num_heads, model_dim, QKV_dim, QKV_dim, QKV_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.attn_2 = MultiHeadAttention(num_heads, model_dim, QKV_dim, QKV_dim, QKV_dim)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)

    def forward(self, query_modality, modality_A, modality_B):
        out_1 = self.add_norm_1(query_modality, lambda x: self.attn_1(x, modality_A, modality_A))
        out_2 = self.add_norm_2(out_1, lambda x: self.attn_2(x, modality_B, modality_B))
        return self.add_norm_3(out_2, self.ff)


'''
Stacks of MultiAttnLayer.
'''
class MultiAttn(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, query_modality, modality_A, modality_B):
        for layer in self.layers:
            query_modality = layer(query_modality, modality_A, modality_B)
        return query_modality


'''
Double-modality attention fusion module (for ECG + Optical Flow).
'''
class MultiAttnModel(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.multiattn_ecg = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_flow = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)

    def forward(self, ecg_features, flow_features):
        fused_ecg = self.multiattn_ecg(ecg_features, flow_features, flow_features)
        fused_flow = self.multiattn_flow(flow_features, ecg_features, ecg_features)
        return fused_ecg, fused_flow
