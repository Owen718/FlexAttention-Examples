import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class SlidingWindowCausalAttention(nn.Module):
    def __init__(self, Q_LEN: int, KV_LEN: int, window_size: int, B: int = None, H: int = None):
        super().__init__()
        self.window_size = window_size
        # Precompute block_mask 
        self.block_mask = create_block_mask(
            self.sliding_window_causal_mask_mod, 
            B=B, 
            H=H, 
            Q_LEN=Q_LEN, 
            KV_LEN=KV_LEN
        )

    def sliding_window_causal_mask_mod(self, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        causal_mask = q_idx >= kv_idx
        window_mask = (q_idx - kv_idx) <= self.window_size 
        return causal_mask & window_mask

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N, D]
        """
        # Ensure block_mask is on the same device as inputs
        output = flex_attention(Q, K, V, block_mask=self.block_mask.to(Q.device))
        return output 