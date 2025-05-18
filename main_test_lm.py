import torch
import unittest
from attention_ops.linformer import LinformerAttention
from attention_ops.performer import PerformerAttention
from attention_ops.longformer import LongformerAttention
from attention_ops.relative_positional import RelativePositionalAttention
from attention_ops.alibi import AlibiAttention
from attention_ops.soft_cap import SoftCapAttention
from attention_ops.causal import CausalAttention
from attention_ops.sliding_window_causal import SlidingWindowCausalAttention
from attention_ops.prefix_lm import PrefixLMAttention
from attention_ops.document_mask import DocumentMaskAttention, generate_doc_mask_mod

LLM_SHAPES = [
    (8, 32, 1024, 128),   # B, H, N, D
    (4, 40, 2048, 64),
    (2, 16, 4096, 80),
]

class TestAttentionOps(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_linformer_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            M = max(64, N // 8)  # 降维后长度
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            attn = LinformerAttention(dim=H*D, n_heads=H, seq_len=N, proj_len=M).to(self.device)
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            self.assertFalse(torch.isnan(out).any())

    def test_performer_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            r = max(32, D)  # 随机特征维度
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            attn = PerformerAttention(dim=H*D, n_heads=H, feature_dim=r).to(self.device)
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            # print(out)
            self.assertFalse(torch.isnan(out).any())

    def test_longformer_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            # 选前2个为全局token
            global_pos = {0, 1}
            attn = LongformerAttention(window_size=32, global_positions=global_pos).to(self.device)
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            self.assertFalse(torch.isnan(out).any())
            self.assertTrue((out[:, :, list(global_pos), :].abs().sum() > 0).all())

    def test_relative_positional_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            attn = RelativePositionalAttention().to(self.device)
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            self.assertFalse(torch.isnan(out).any())

    def test_alibi_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            # n_heads for AlibiAttention is H from LLM_SHAPES
            attn = AlibiAttention(n_heads=H).to(self.device)
            # Ensure alibi_bias_slopes is on the same device as Q, K, V
            attn.alibi_bias_slopes = attn.alibi_bias_slopes.to(self.device)
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            self.assertFalse(torch.isnan(out).any())

    def test_soft_cap_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            attn_default = SoftCapAttention().to(self.device)
            out_default = attn_default(Q, K, V)
            self.assertEqual(out_default.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out_default))
            self.assertFalse(torch.isnan(out_default).any())

            attn_custom_cap = SoftCapAttention(softcap_value=30.0).to(self.device)
            out_custom_cap = attn_custom_cap(Q, K, V)
            self.assertEqual(out_custom_cap.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out_custom_cap))
            self.assertFalse(torch.isnan(out_custom_cap).any())

    def test_causal_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            # For CausalAttention, Q_LEN and KV_LEN are N
            # B and H are set to None in CausalAttention to allow broadcasting, matching blog example.
            attn = CausalAttention(Q_LEN=N, KV_LEN=N).to(self.device) # B and H are None by default
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            self.assertFalse(torch.isnan(out).any())
            # TODO: Add a check to verify causal masking behavior if possible, e.g., by checking specific elements

    def test_sliding_window_causal_llm_shapes(self):
        for B, H, N, D in LLM_SHAPES:
            Q = torch.randn(B, H, N, D, device=self.device)
            K = torch.randn(B, H, N, D, device=self.device)
            V = torch.randn(B, H, N, D, device=self.device)
            window_size = min(N // 4, 128) # Example window size
            if window_size == 0 and N > 0: window_size = N # handle small N
            if N == 0: window_size = 0 # handle N=0
            
            attn = SlidingWindowCausalAttention(Q_LEN=N, KV_LEN=N, window_size=window_size).to(self.device)
            out = attn(Q, K, V)
            self.assertEqual(out.shape, (B, H, N, D))
            self.assertTrue(torch.is_tensor(out))
            self.assertFalse(torch.isnan(out).any())
            # TODO: Specific checks for sliding window + causal behavior

    # def test_prefix_lm_llm_shapes(self):
    #     for B, H, N, D in LLM_SHAPES:
    #         if N == 0: continue # Prefix length makes no sense for N=0
    #         Q = torch.randn(B, H, N, D, device=self.device)
    #         K = torch.randn(B, H, N, D, device=self.device)
    #         V = torch.randn(B, H, N, D, device=self.device)
            
    #         # Example prefix_lengths: vary per batch item, ensure they are less than N
    #         prefix_lengths = torch.randint(1, N + 1, (B,), device=self.device)
    #         # Ensure prefix length is at least 1 if N > 0, and not greater than N.
    #         # For N=0 case, this test is skipped.
    #         # If N=1, prefix_lengths can be 1.
    #         # prefix_lengths = torch.clamp(prefix_lengths, 1, N)
    #         # Handle cases where N could be small, ensure prefix_lengths are valid.
    #         # Let's make prefix_lengths N // 2 for simplicity in test, or at least 1.
    #         test_prefix_lengths = torch.clamp(torch.full((B,), N // 2, device=self.device), min=1, max=N if N > 0 else 1)
    #         if N == 0: test_prefix_lengths = torch.zeros((B,), device=self.device, dtype=torch.long)

    #         attn = PrefixLMAttention().to(self.device)
    #         # The prefix_lengths tensor must be on the same device as the model/data.
    #         out = attn(Q, K, V, prefix_lengths=test_prefix_lengths)
    #         self.assertEqual(out.shape, (B, H, N, D))
    #         self.assertTrue(torch.is_tensor(out))
    #         self.assertFalse(torch.isnan(out).any())
    #         # TODO: More specific checks for PrefixLM behavior

    # def test_document_mask_llm_shapes(self):
    #     for B_orig, H, N_total, D in LLM_SHAPES:
    #         if N_total == 0: continue
    #         # For document masking, typically B=1 and N_total is the sum of sequence lengths
    #         B = 1 
    #         Q = torch.randn(B, H, N_total, D, device=self.device)
    #         K = torch.randn(B, H, N_total, D, device=self.device)
    #         V = torch.randn(B, H, N_total, D, device=self.device)

    #         # Example document_id for N_total tokens
    #         # e.g., [0, 0, 0, 1, 1, 2, 2, 2, 2] for N_total = 9, 3 docs
    #         if N_total < 3:
    #             doc_ids_list = [0] * N_total
    #         else:
    #             doc_ids_list = [0] * (N_total // 3) + [1] * (N_total // 3) + [2] * (N_total - 2 * (N_total // 3))
    #         document_id = torch.tensor(doc_ids_list[:N_total], device=self.device, dtype=torch.long)
    #         if document_id.numel() == 0 and N_total > 0 : #handle N_total = 1,2 where N_total//3 = 0
    #             document_id = torch.zeros(N_total, device=self.device, dtype=torch.long)
    #         elif document_id.numel() != N_total and N_total > 0:
    #              # Fallback if logic above is tricky for small N_total
    #             document_id = torch.zeros(N_total, device=self.device, dtype=torch.long)
    #             if N_total > 1: document_id[N_total//2:] = 1 # Simple two docs for testing if possible

    #         attn = DocumentMaskAttention().to(self.device)
    #         out = attn(Q, K, V, document_id=document_id)
    #         self.assertEqual(out.shape, (B, H, N_total, D))
    #         self.assertTrue(torch.is_tensor(out))
    #         self.assertFalse(torch.isnan(out).any())
    #         # TODO: Test generate_doc_mask_mod functionality with a combined mask

    # Example of how one might test generate_doc_mask_mod (more involved)
    # def test_generated_doc_causal_mask(self):
    #     B, H, N_total, D = 1, 2, 10, 4 # Example shape
    #     Q = torch.randn(B, H, N_total, D, device=self.device)
    #     K = torch.randn(B, H, N_total, D, device=self.device)
    #     V = torch.randn(B, H, N_total, D, device=self.device)
    #     document_id = torch.tensor([0,0,0,0,1,1,1,2,2,2], device=self.device, dtype=torch.long)

    #     def base_causal_mask_mod(b, h, q_idx, kv_idx):
    #         return q_idx >= kv_idx
        
    #     # This uses the generate_doc_mask_mod from attention_ops.document_mask
    #     combined_mask_mod = generate_doc_mask_mod(base_causal_mask_mod, document_id)
        
    #     # Need an nn.Module to use it, or call flex_attention directly
    #     block_mask = create_block_mask(combined_mask_mod, B=B, H=H, Q_LEN=N_total, KV_LEN=N_total, device=self.device)
    #     out = flex_attention(Q,K,V, block_mask=block_mask.to(self.device))
    #     self.assertEqual(out.shape, (B, H, N_total, D))
    #     self.assertFalse(torch.isnan(out).any())

if __name__ == '__main__':
    unittest.main()
