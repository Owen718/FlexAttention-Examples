import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class SwinAttention(nn.Module):
    def __init__(self, dim, n_heads, H_img, W_img, window_size=7, shift_size=0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.H_img = H_img
        self.W_img = W_img
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // n_heads

    def window_mask(self, b, h, q_idx, kv_idx):
        W_img = self.W_img
        M = self.window_size
        q_row = q_idx // W_img
        q_col = q_idx % W_img
        k_row = kv_idx // W_img
        k_col = kv_idx % W_img
        return torch.logical_and((q_row // M == k_row // M), (q_col // M == k_col // M))

    def cyclic_shift(self, t: torch.Tensor) -> torch.Tensor:
        B, H, N, D_head = t.shape
        # 将输入张量重塑为 (B, H, H_img, W_img, D_head) 以便在空间维度上进行 roll 操作
        x_reshaped = t.view(B, H, self.H_img, self.W_img, D_head)
        # 执行循环移位
        x_rolled = torch.roll(x_reshaped, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        # 将形状恢复为 (B, H, N, D_head)
        return x_rolled.view(B, H, N, D_head)

    def forward(self, Q, K, V):
        # Q, K, V: [B, H, N, D], N = H_img*W_img
        N = Q.shape[2]
        device = Q.device
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: self.window_mask(b, h, q_idx, kv_idx),
            B=Q.shape[0], H=Q.shape[1], Q_LEN=N, KV_LEN=N, device=device
        )
        if self.shift_size > 0:
            Q = self.cyclic_shift(Q)
            K = self.cyclic_shift(K)
            V = self.cyclic_shift(V)
        out = flex_attention(Q, K, V, block_mask=block_mask)
        return out 