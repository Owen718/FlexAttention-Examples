import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class CausalAttention(nn.Module):
    def __init__(self, Q_LEN: int, KV_LEN: int, B: int = None, H: int = None):
        super().__init__()
        # Precompute block_mask if dimensions are static
        # If B or H are None, they will be broadcasted
        self.block_mask = create_block_mask(
            self.causal_mask_mod, 
            B=B, 
            H=H, 
            Q_LEN=Q_LEN, 
            KV_LEN=KV_LEN
        )

    def causal_mask_mod(self, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns True if the position should participate (q_idx >= kv_idx).
        """
        return q_idx >= kv_idx

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N, D]
        Ensure Q_LEN and KV_LEN used for block_mask match N from Q, K, V.
        If B and H were specified for block_mask, they must match B and H from Q, K, V.
        """
        # Dynamically create block_mask if Q_LEN/KV_LEN or B/H change often or weren't known at init.
        # For this example, assuming block_mask is precomputed or Q_LEN/KV_LEN are fixed for the model.
        # A more robust implementation might re-create block_mask if shapes mismatch and it was created with specific B, H.
        
        # Check if the Q_LEN and KV_LEN match the input tensor's sequence length N
        # The block_mask internally stores Q_LEN and KV_LEN from its creation.
        # flex_attention will handle mismatches if the mask is incompatible.
        # For simplicity, we assume the user ensures Q_LEN, KV_LEN used for CausalAttention
        # instantiation match the N dimension of Q, K, V tensors passed to forward.

        # If block_mask was created with specific B and H (not None)
        # then input Q must match those B and H.
        # The current block_mask creation in __init__ uses Q_LEN, KV_LEN passed to init.
        # And B, H if they are also passed (otherwise None for broadcasting).

        current_B, current_H, current_Q_N, _ = Q.shape
        _, _, current_KV_N, _ = K.shape

        # This check is a bit simplistic. `create_block_mask` has its own shape logic.
        # What's important is that the block_mask passed to flex_attention is compatible.
        # If self.block_mask was created with B=None, H=None, it should broadcast.
        # If created with specific B, H, Q_LEN, KV_LEN, those must match.
        # We are assuming Q_LEN and KV_LEN in __init__ match N for Q and K respectively.

        output = flex_attention(Q, K, V, block_mask=self.block_mask.to(Q.device))
        return output 