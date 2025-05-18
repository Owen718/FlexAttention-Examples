import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

class SoftCapAttention(nn.Module):
    def __init__(self, softcap_value: float = 20.0, use_tanh_approximation: bool = False):
        super().__init__()
        self.softcap_value = softcap_value
        self.use_tanh_approximation = use_tanh_approximation

    def score_mod(self, score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        """
        Applies soft-capping to the attention score.
        """
        score = score / self.softcap_value
        if self.use_tanh_approximation:
            # Using the approximation: tanh(x) approx x - x^3/3 for small x or other fast approximations
            # For simplicity, sticking to torch.tanh unless a specific approximation is provided/required.
            # PyTorch blog mentions: "we likely want to use a tanh approximation in this case for performance reasons"
            # A common approximation for tanh(x) is x / (1 + |x|) or variants for speed.
            # Sticking to torch.tanh for now for correctness, can be swapped for an approx.
            score = torch.tanh(score) 
        else:
            score = torch.tanh(score)
        score = score * self.softcap_value
        return score

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N, D]
        """
        output = flex_attention(Q, K, V, score_mod=self.score_mod)
        return output 