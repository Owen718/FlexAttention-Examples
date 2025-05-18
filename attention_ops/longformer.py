import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
class LongformerAttention(nn.Module):
    def __init__(self, window_size, global_positions=None):
        super().__init__()
        self.window_size = window_size
        self.global_positions = global_positions if global_positions is not None else set()

    def longformer_mask(self, b, h, q_idx, kv_idx):
        if q_idx in self.global_positions or kv_idx in self.global_positions:
            return True
        else:
            return abs(q_idx - kv_idx) <= self.window_size

    def forward(self, Q, K, V):
        # Q, K, V: [B, H, N, D]
        N = Q.shape[2]
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: self.longformer_mask(b, h, q_idx, kv_idx),
            B=Q.shape[0],
            H=Q.shape[1],
            Q_LEN=N,
            KV_LEN=N,
            device=Q.device
        )
        out = flex_attention(Q, K, V, block_mask=block_mask)
        return out 