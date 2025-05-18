import torch
import unittest
from attention_ops.linformer import LinformerAttention
from attention_ops.performer import PerformerAttention
from attention_ops.longformer import LongformerAttention

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

if __name__ == '__main__':
    unittest.main()
