import torch.nn as nn
class ImagePatcher(nn.Module):
    def __init__(self, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = in_channels * patch_size * patch_size

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, self.embed_dim)  # [B, N, D]
        return x
