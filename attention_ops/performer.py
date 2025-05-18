import torch
import torch.nn as nn
import math

def orthogonal_random_features(x, proj_matrix):
    # x: [B, H, N, D], proj_matrix: [D, r]
    # 返回: [B, H, N, r]
    x_proj = torch.einsum('b h n d, d r -> b h n r', x, proj_matrix)
    x_proj = x_proj / (x.shape[-1] ** 0.5)  # 缩放
    x_proj = torch.clamp(x_proj, min=-10, max=10)  # 防止exp溢出
    return torch.exp(x_proj)  # 近似 softmax kernel

class PerformerAttention(nn.Module):
    def __init__(self, dim, n_heads, feature_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.feature_dim = feature_dim
        self.proj_matrix = nn.Parameter(torch.randn(self.head_dim, feature_dim))

    def forward(self, Q, K, V):
        # Q, K, V: [B, H, N, D]
        Q_prime = orthogonal_random_features(Q, self.proj_matrix)  # [B, H, N, r]
        K_prime = orthogonal_random_features(K, self.proj_matrix)  # [B, H, N, r]
        # 计算 K' 与 V 的乘积
        KV_accum = torch.einsum('b h n r, b h n d -> b h r d', K_prime, V)
        # 计算 Q' 与上述结果相乘
        out = torch.einsum('b h n r, b h r d -> b h n d', Q_prime, KV_accum)
        # 归一化
        normalizer = torch.einsum('b h n r, b h r -> b h n', Q_prime, K_prime.sum(dim=2))
        normalizer = torch.clamp(normalizer, min=1e-6)  # 防止除0
        out = out / normalizer.unsqueeze(-1)
        return out 