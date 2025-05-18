import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Helper for causal mask part of PrefixLM
def causal_mask_mod_func(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
    return q_idx >= kv_idx

class PrefixLMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # No block_mask precomputation here as prefix_length can vary per batch/input

    def prefix_lm_mask_mod(self, b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, prefix_length_for_item: int) -> torch.Tensor:
        """
        prefix_length_for_item: scalar int for the current batch item b
        """
        is_prefix_related = kv_idx < prefix_length_for_item # Full attention for keys within prefix
        is_causal_after_prefix = (q_idx >= kv_idx) # Causal for all
        
        # If query is in prefix, it can attend to full prefix and causally to itself and subsequent tokens in prefix.
        # If query is after prefix, it can attend to full prefix and causally to tokens after prefix up to itself.
        # The blog uses or_masks(prefix_mask, causal_mask)
        # prefix_mask: kv_idx <= prefix_length[b]
        # causal_mask: q_idx >= kv_idx

        # Simplified: If kv is in prefix, allow. Else, apply causal.
        # This is equivalent to or_masks logic if prefix_length means context length
        mask = (kv_idx < prefix_length_for_item) | (q_idx >= kv_idx)
        return mask

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, prefix_lengths: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N, D]
        prefix_lengths: [B], tensor of integers indicating the prefix length for each item in the batch.
        """
        B, H, N, D = Q.shape
        
        # According to the blog, block_mask needs to be recomputed if sparsity pattern changes per input.
        # For PrefixLM, prefix_lengths can change, so the mask changes.
        # We need to create a mask_mod that captures prefix_lengths.
        
        # This implementation will create ONE block_mask for the whole batch.
        # This means the prefix_lm_mask_mod needs to handle the b index correctly with prefix_lengths[b].
        # create_block_mask will iterate b from 0 to B-1.
        
        # The mask_mod passed to create_block_mask should not take extra args beyond b, h, q_idx, kv_idx.
        # So, we need to define it within forward where prefix_lengths is available.
        # Or, if prefix_lengths is fixed for an instance, it can be part of __init__.
        # Given the dynamic nature, let's define it here.
        
        # This is tricky because create_block_mask expects a function with a fixed signature.
        # We use a lambda that captures `prefix_lengths` from the outer scope.
        # The `b` argument in the lambda is the batch index passed by create_block_mask.
        
        # The blog states: "block_mask = create_block_mask(prefix_lm_causal, B=B, H=None, S, S)"
        # where prefix_lm_causal itself depends on prefix_length[b].

        def current_prefix_lm_mask_mod(batch_idx: torch.Tensor, head_idx: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
            # batch_idx is a scalar tensor, convert to int for indexing
            # We assume prefix_lengths is on the same device or can be accessed.
            # Ensure prefix_lengths is on the same device as q_idx for direct indexing.
            pl = prefix_lengths[batch_idx.item()] # .item() to get Python int
            
            # Original blog logic for prefix_lm_causal = or_masks(prefix_mask, causal_mask)
            # prefix_mask: kv_idx <= prefix_length[b]
            # causal_mask: q_idx >= kv_idx
            
            # Mask for the prefix part: Key is within the prefix length for this batch item
            # Note: blog uses kv_idx <= prefix_length[b]. Using < for 0-indexed length.
            # E.g. if prefix_length is 2, indices 0, 1 are prefix. kv_idx < 2.
            prefix_m = kv_idx < pl
            
            # Standard causal mask
            causal_m = q_idx >= kv_idx
            
            return prefix_m | causal_m

        block_mask_for_batch = create_block_mask(
            current_prefix_lm_mask_mod, 
            B=B, 
            H=None, # Broadcast over heads as per blog example
            Q_LEN=N, 
            KV_LEN=N,
            device=Q.device # Ensure mask is created on the correct device
        )
        
        output = flex_attention(Q, K, V, block_mask=block_mask_for_batch.to(Q.device))
        return output 