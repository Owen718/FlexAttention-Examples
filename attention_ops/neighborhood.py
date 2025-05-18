import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, n_heads, H_img, W_img, radius=2):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.H_img = H_img
        self.W_img = W_img
        self.radius = radius
        self.head_dim = dim // n_heads

    def neighborhood_mask(self, b, h, q_idx, kv_idx):
        W_img = self.W_img
        r = self.radius
        q_row = q_idx // W_img
        q_col = q_idx % W_img
        k_row = kv_idx // W_img
        k_col = kv_idx % W_img
        return torch.logical_and(torch.abs(q_row - k_row) <= r, torch.abs(q_col - k_col) <= r)

    def forward(self, Q, K, V):
        # Q, K, V: [B, H, N, D], N = H_img*W_img
        N = Q.shape[2]
        device = Q.device
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: self.neighborhood_mask(b, h, q_idx, kv_idx),
            B=Q.shape[0], H=Q.shape[1], Q_LEN=N, KV_LEN=N, device=device
        )
        out = flex_attention(Q, K, V, block_mask=block_mask)
        return out 