import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

class RelativePositionalAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def score_mod(self, score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        """
        Adds relative positional encoding to the score.
        score: Scalar tensor representing the dot product of a query token and a key token.
        b: Current element in batch.
        h: Current head.
        q_idx: Position in query.
        kv_idx: Position in key/value tensors.
        """
        return score + (q_idx - kv_idx)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N, D]
        """
        output = flex_attention(Q, K, V, score_mod=self.score_mod)
        return output 