import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class DocumentMaskAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Mask is created dynamically in forward based on document_id

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, document_id: torch.Tensor) -> torch.Tensor:
        """
        Q, K, V: [B, H, N_total, D] where N_total is the flattened sequence length.
                 It's assumed B=1 for typical packed sequence use case, or document_id handles batching.
                 If B > 1, document_id should distinguish documents across batches or be shaped [B, N_total].
                 The blog implies document_id is [SEQ_LEN] for a single flattened sequence.
                 Let's assume B=1 for this implementation for simplicity with document_id: [N_total]
        document_id: [N_total] tensor of integers, indicating the document for each token.
        """
        B, H, N_total, D = Q.shape

        if B != 1 and not (document_id.ndim == 2 and document_id.shape[0] == B):
            # For B > 1, document_id needs to be [B, N_total] or handled appropriately.
            # This basic implementation assumes B=1 or a pre-batched document_id for simplicity.
            # Or, the mask_mod should take `b` into account for document_id[b, q_idx] etc.
            # The blog example `document_id: [SEQ_LEN]` implies a single sequence context (B=1, effectively).
            # Let's stick to the blog's direct interpretation for the mask_mod for now.
            # This implies that if B > 1, each item in batch has its own unrelated document_id array of shape [N_total].
            # Which means the mask_mod must use document_id[b, q_idx] and document_id[b, kv_idx].
            # For now, let's assume document_id is prepared such that it can be indexed by q_idx, kv_idx globally
            # or that we are effectively in a B=1 scenario for packing.
            # If B > 1, and document_id is [N_total], it means all batch items share the same doc structure - unlikely.
            # Let's assume for testing, B=1 will be used or document_id correctly handles batch index `b` inside mask_mod.
            pass # Allow B > 1 if user prepares document_id to be [B, N_total]

        def doc_masking_mod(batch_idx: torch.Tensor, head_idx: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
            # Assuming document_id is [N_total] and broadcasted/repeated for batch if B > 1,
            # or more correctly, document_id should be [B, N_total] and indexed by batch_idx.
            # If document_id is [N_total] (shared across batch):
            #   return document_id[q_idx] == document_id[kv_idx]
            # If document_id is [B, N_total]:
            #   return document_id[batch_idx.item(), q_idx] == document_id[batch_idx.item(), kv_idx]
            # For this example, let's assume document_id is [N_total] as in blog, and it works if B=1 or if the same structure is repeated.
            # Or, it could be document_id for the *specific* batch item if create_block_mask calls with b.
            # The blog example `document_id: [SEQ_LEN]` and `document_id[q_idx] == document_id[kv_idx]`
            # suggests `document_id` is 1D. create_block_mask will pass `b` from 0 to B-1.
            # So, if `document_id` is truly per-batch item, it should be `document_ids[b]`, where `document_ids` is a list of tensors,
            # or `document_id` is a [B, N_total] tensor.

            # Let's assume document_id is [N_total] and applies to all batch items if B > 1 (less common),
            # or this module is intended for B=1 usage with packed sequences.
            # For a more robust solution with B > 1, document_id should be [B, N_total].
            if document_id.ndim == 1:
                 # Shared document_id across batch, or B=1
                return document_id[q_idx] == document_id[kv_idx]
            elif document_id.ndim == 2:
                 # document_id is [B, N_total]
                return document_id[batch_idx.item(), q_idx] == document_id[batch_idx.item(), kv_idx]
            else:
                raise ValueError("document_id must be 1D [N_total] or 2D [B, N_total]")

        # B for create_block_mask should be B from input Q if document_id is [B, N_total]
        # If document_id is [N_total] (shared), B for create_block_mask can be 1 or B (mask is same for all items)
        # Blog: create_block_mask(document_masking, B=None, H=None, S, S) if mask is same for batch/head
        # If document_id is [B, N_total], then B for create_block_mask should be actual B.
        mask_B = B if document_id.ndim == 2 else None

        block_mask_for_batch = create_block_mask(
            doc_masking_mod, 
            B=mask_B, 
            H=None, # Assume document structure is head-independent
            Q_LEN=N_total, 
            KV_LEN=N_total,
            device=Q.device
        )
        
        output = flex_attention(Q, K, V, block_mask=block_mask_for_batch.to(Q.device))
        return output

# Higher-level modification function from the blog
# This function itself is not an nn.Module, but a generator for mask_mod functions.
def generate_doc_mask_mod(inner_mask_mod, document_id: torch.Tensor):
    # Ensure document_id is on CPU for unique_consecutive, or adapt as needed.
    # Or, ensure it's on the same device as where computations will occur.
    # Let's assume document_id is already on the correct device.
    unique_docs, counts = torch.unique_consecutive(document_id, return_counts=True)
    offsets = torch.cat([torch.tensor([0], device=document_id.device, dtype=counts.dtype), counts.cumsum(0)[:-1]])

    # Map original document IDs to zero-based indices for accessing offsets, if necessary
    # This is only needed if document_ids are not already 0, 1, 2...
    # For simplicity, assume document_id values are direct indices or can be mapped.
    # A robust way: create a mapping from unique_docs to 0..len(unique_docs)-1
    doc_to_offset_idx = {doc_val.item(): i for i, doc_val in enumerate(unique_docs)}

    def doc_mask_wrapper(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor):
        # This wrapper assumes document_id is 1D [N_total] and applies across batch items if B > 1,
        # or that this combined mask_mod will be used in a context where `b` is handled (e.g. B=1, or document_id itself is [B, N_total]).
        # If document_id is [B, N_total], then document_id[q_idx] needs to be document_id[b, q_idx], etc.
        # Let's assume 1D document_id for this generic wrapper, consistent with blog's `generate_doc_mask_mod` example structure.

        q_doc_original_id = document_id[q_idx]
        kv_doc_original_id = document_id[kv_idx]
        same_doc = (q_doc_original_id == kv_doc_original_id)
        
        if not same_doc.item(): # If not same doc, mask out immediately.
            return False      # Return scalar boolean as per mask_mod contract.

        # If same_doc, then proceed with inner_mask_mod using logical indices.
        # Need to get the 0-based index for the offset array.
        # This assumes q_doc_original_id.item() is a key in doc_to_offset_idx
        q_doc_offset_idx = doc_to_offset_idx[q_doc_original_id.item()]
        kv_doc_offset_idx = doc_to_offset_idx[kv_doc_original_id.item()] # should be same as q_doc_offset_idx if same_doc

        q_logical = q_idx - offsets[q_doc_offset_idx]
        kv_logical = kv_idx - offsets[kv_doc_offset_idx]
        
        # Call the original mask_mod with logical indices
        # The `b` and `h` are passed through. If inner_mask_mod uses them, it should be fine.
        return inner_mask_mod(b, h, q_logical, kv_logical)
    
    return doc_mask_wrapper 