import torch
from torch.nn.attention.flex_attention import create_mask
import functools

# --- 配置参数 ---
SEQ_LEN = 12  # 序列长度，与图片一致
B = 1        # Batch size
H = 1        # Number of heads
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 辅助函数 ---
def print_mask(name: str, mask_tensor: torch.Tensor):
    """打印掩码张量"""
    print(f"--- {name} (尺寸: {mask_tensor.shape}) ---")
    # 移除 Batch 和 Head 维度进行打印 (如果它们是1)
    if mask_tensor.size(0) == 1 and mask_tensor.size(1) == 1:
        # 将布尔张量转换为整数张量 (True -> 1, False -> 0)
        print(mask_tensor.squeeze(0).squeeze(0).int())
    else:
        print(mask_tensor.int())
    print("\n")

# --- 1. Vanilla Attention (Causal Mask) ---
def causal_attention_mask_fn(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
    """因果注意力掩码：查询只能注意到键值对中索引小于或等于自身的位置。"""
    return q_idx >= kv_idx

print("正在生成 Vanilla Attention (Causal Mask)...")
vanilla_mask = create_mask(
    mod_fn=causal_attention_mask_fn,
    B=B,
    H=H,
    Q_LEN=SEQ_LEN,
    KV_LEN=SEQ_LEN,
    device=DEVICE
)
print_mask("Vanilla Attention (Causal Mask)", vanilla_mask)

# --- 2. Sliding Window Attention (Causal) ---
WINDOW_SIZE = 3 # 根据图片推断的窗口大小

def sliding_window_causal_attention_mask_fn(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """滑动窗口因果注意力掩码：
    1. 查询只能注意到键值对中索引小于或等于自身的位置 (causal)。
    2. 查询和键值对之间的距离必须在窗口大小之内。
    """
    causal_condition = q_idx >= kv_idx
    window_condition = (q_idx - kv_idx) < window_size
    return causal_condition & window_condition

print(f"正在生成 Sliding Window Attention (Window Size = {WINDOW_SIZE})...")
# functools.partial 用于将 window_size 参数固定到掩码函数中
# 因为 create_mask 期望的 mod_fn 只有 (b, h, q_idx, kv_idx) 四个参数
partial_sliding_window_fn = functools.partial(
    sliding_window_causal_attention_mask_fn,
    window_size=WINDOW_SIZE
)

sliding_window_mask = create_mask(
    mod_fn=partial_sliding_window_fn,
    B=B,
    H=H,
    Q_LEN=SEQ_LEN,
    KV_LEN=SEQ_LEN,
    device=DEVICE
)
print_mask(f"Sliding Window Attention (Window Size = {WINDOW_SIZE})", sliding_window_mask)

# --- 3. Neighborhood Attention --- 
H_IMG_NEIGHBOR = 8
W_IMG_NEIGHBOR = 8
RADIUS_NEIGHBOR = 3
SEQ_LEN_NEIGHBOR = H_IMG_NEIGHBOR * W_IMG_NEIGHBOR

def neighborhood_attention_mask_fn(
    b: torch.Tensor, 
    h: torch.Tensor, 
    q_idx: torch.Tensor, 
    kv_idx: torch.Tensor, 
    W_img: int, 
    radius: int
) -> torch.Tensor:
    """Neighborhood attention mask:
    A query pixel (q_row, q_col) attends to a key pixel (k_row, k_col) if 
    abs(q_row - k_row) <= radius AND abs(q_col - k_col) <= radius.
    q_idx and kv_idx are 1D indices that need to be converted to 2D.
    """
    q_row = q_idx // W_img
    q_col = q_idx % W_img
    k_row = kv_idx // W_img
    k_col = kv_idx % W_img
    
    row_condition = torch.abs(q_row - k_row) <= radius
    col_condition = torch.abs(q_col - k_col) <= radius
    return row_condition & col_condition

print(f"\n正在生成 Neighborhood Attention (H={H_IMG_NEIGHBOR}, W={W_IMG_NEIGHBOR}, Radius={RADIUS_NEIGHBOR})...")

partial_neighborhood_fn = functools.partial(
    neighborhood_attention_mask_fn,
    W_img=W_IMG_NEIGHBOR,
    radius=RADIUS_NEIGHBOR
)

neighborhood_mask = create_mask(
    mod_fn=partial_neighborhood_fn,
    B=B, # Using global B and H for simplicity, can be adjusted if needed
    H=H,
    Q_LEN=SEQ_LEN_NEIGHBOR,
    KV_LEN=SEQ_LEN_NEIGHBOR,
    device=DEVICE
)
print_mask(f"Neighborhood Attention (H={H_IMG_NEIGHBOR}, W={W_IMG_NEIGHBOR}, Radius={RADIUS_NEIGHBOR})", neighborhood_mask)

print("可视化脚本执行完毕。")
