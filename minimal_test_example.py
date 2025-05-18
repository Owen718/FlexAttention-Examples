from torch.nn.attention.flex_attention import flex_attention
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, H, N, D = 8, 32, 1024, 128
Q = torch.randn(B, H, N, D, device=device).requires_grad_()
K = torch.randn(B, H, N, D, device=device).requires_grad_()
V = torch.randn(B, H, N, D, device=device).requires_grad_()
def noop(score, b,h, q_idx, kv_idx):
    return score
out = flex_attention(Q, K, V, score_mod=noop) # full attention
out.sum().backward()
# print(out)

def relative_positional(score,b,h,q_idx,kv_idx):
    return score +(q_idx-kv_idx)
out = flex_attention(Q, K, V, score_mod=relative_positional) # relative positional attention
out.sum().backward()

window_size = 1024
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
def sliding_window_causal_mask_mod( b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
    causal_mask = q_idx >= kv_idx
    window_mask = (q_idx - kv_idx) <= window_size 
    return causal_mask & window_mask

block_mask = create_block_mask(
            sliding_window_causal_mask_mod, 
            B=B, 
            H=H, 
            Q_LEN=N, 
            KV_LEN=N
        )
out = flex_attention(Q, K, V, block_mask=block_mask) # sliding window attention
out.sum().backward()
# print(out)