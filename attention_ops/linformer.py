import torch
import torch.nn as nn
# from flexattention import flex_attention
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class LinformerAttention(nn.Module):
    def __init__(self, dim, n_heads, seq_len, proj_len, shared_proj=True):
        super().__init__()
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.proj_len = proj_len
        self.head_dim = dim // n_heads
        self.shared_proj = shared_proj
        if shared_proj:
            self.E = nn.Parameter(torch.randn(seq_len, proj_len))
        else:
            self.E_K = nn.Parameter(torch.randn(seq_len, proj_len))
            self.E_V = nn.Parameter(torch.randn(seq_len, proj_len))

    def forward(self, Q, K, V):
        # Q: [B, H, N, D], K: [B, H, N, D], V: [B, H, N, D]
        if self.shared_proj:
            K_proj = torch.einsum('b h n d, n m -> b h m d', K, self.E)  # [B, H, M, D]
            V_proj = torch.einsum('b h n d, n m -> b h m d', V, self.E)  # [B, H, M, D]
        else:
            K_proj = torch.einsum('b h n d, n m -> b h m d', K, self.E_K)
            V_proj = torch.einsum('b h n d, n m -> b h m d', V, self.E_V)
        out = flex_attention(Q, K_proj, V_proj)
        return out 