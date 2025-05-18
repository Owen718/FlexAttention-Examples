import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Copied from https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742"""
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    import math
    if math.log2(n_heads).is_integer():
        slopes = torch.Tensor(get_slopes_power_of_2(n_heads))
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        slopes = torch.Tensor(get_slopes_power_of_2(closest_power_of_2))
        additional_slopes = torch.Tensor(get_slopes_power_of_2(2*closest_power_of_2))[0::2][:n_heads-closest_power_of_2]
        slopes = torch.cat([slopes, additional_slopes])
    return slopes

class AlibiAttention(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        # The alibi_bias tensor will be registered as a buffer
        self.register_buffer('alibi_bias_slopes', get_alibi_slopes(n_heads))

    def score_mod(self, score: torch.Tensor, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        """
        Applies ALiBi bias to the attention score.
        """
        # Ensure h is an integer for indexing
        # The alibi_bias_slopes is of shape [num_heads]
        # h is a tensor, needs to be converted to an int index or use gather
        # For simplicity in score_mod, assuming h can be used to index if it's scalar-like
        # or we ensure alibi_bias_slopes is on the correct device and use it directly.
        # FlexAttention's score_mod receives h as a scalar tensor (i32[])
        bias = self.alibi_bias_slopes[h] * (q_idx - kv_idx)
        return score + bias

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N, D]
        H (number of heads) in Q, K, V must match n_heads used in __init__
        """
        if Q.shape[1] != self.alibi_bias_slopes.shape[0]:
            raise ValueError(f"Number of heads in Q ({Q.shape[1]}) must match n_heads ({self.alibi_bias_slopes.shape[0]}) used for ALiBi initialization.")
        output = flex_attention(Q, K, V, score_mod=self.score_mod)
        return output 