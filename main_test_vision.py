import torch
import unittest
from attention_ops.swin import SwinAttention
from attention_ops.neighborhood import NeighborhoodAttention

class TestVisionAttentionOps(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_swin_attention(self):
        B, H, C = 1, 8, 96
        H_img, W_img = 56, 56
        M = 7
        N = H_img * W_img
        x = torch.randn(B, N, C, device=self.device)
        qkv = torch.nn.Linear(C, C*3, bias=False).to(self.device)(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, N, H, C//H).transpose(1,2) for t in qkv]  # [B, H, N, D]
        # 普通窗口
        attn = SwinAttention(dim=C, n_heads=H, H_img=H_img, W_img=W_img, window_size=M, shift_size=0).to(self.device)
        out = attn(Q, K, V)
        self.assertEqual(out.shape, (B, H, N, C//H))
        # Shifted-Window
        attn_shift = SwinAttention(dim=C, n_heads=H, H_img=H_img, W_img=W_img, window_size=M, shift_size=M//2).to(self.device)
        out_shift = attn_shift(Q, K, V)
        self.assertEqual(out_shift.shape, (B, H, N, C//H))

    def test_neighborhood_attention(self):
        B, H, C = 1, 8, 96
        H_img, W_img = 56, 56
        N = H_img * W_img
        r = 2
        x = torch.randn(B, N, C, device=self.device)
        qkv = torch.nn.Linear(C, C*3, bias=False).to(self.device)(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, N, H, C//H).transpose(1,2) for t in qkv]  # [B, H, N, D]
        attn = NeighborhoodAttention(dim=C, n_heads=H, H_img=H_img, W_img=W_img, radius=r).to(self.device)
        out = attn(Q, K, V)
        self.assertEqual(out.shape, (B, H, N, C//H))

if __name__ == '__main__':
    unittest.main()
