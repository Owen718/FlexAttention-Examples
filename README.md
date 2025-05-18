# FlexAttention-Examples

This repository provides classic Attention mechanism variants implemented using the PyTorch [FlexAttention API](https://pytorch.org/blog/flexattention/).

## What is FlexAttention?

FlexAttention is a new API in PyTorch that allows users to flexibly implement various Attention variants in a few lines of Python code. It leverages `torch.compile` to compile these implementations into highly efficient fused kernels, comparable to manually optimized ones. This means researchers and developers can experiment with new Attention mechanisms more easily without needing to write complex CUDA kernels.

The core idea behind FlexAttention is to provide a `score_mod` function, where users can modify the "score" during the Attention matrix calculation, and a `mask_mod` function used озера with `create_block_mask` to leverage sparsity in the Attention matrix for performance improvements.

For more details and examples, refer to:
- [PyTorch Official Blog: FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/)
- [pytorch-labs/attention-gym: Helpful tools and examples for working with flex-attention](https://github.com/pytorch-labs/attention-gym)

## Implemented Attention Variants

This repository implements the following Attention variants in the `attention_ops/` directory:

- **LinformerAttention**: (`attention_ops/linformer.py`)
    - Reduces the complexity of the self-attention mechanism from O(N^2) to O(N) by projecting the Key and Value tensors into a lower-dimensional space.
    - Suitable for processing long sequences.
- **PerformerAttention**: (`attention_ops/performer.py`)
    - Uses Orthogonal Random Features to approximate the standard Softmax Attention, achieving linear time and space complexity.
    - Also suitable for long sequence tasks.
- **LongformerAttention**: (`attention_ops/longformer.py`)
    - Combines local windowed Attention with global Attention, enabling the model to handle very long sequences while focusing on important global information.
    - `window_size` controls the size of the local window, and `global_positions` specifies which token positions have global attention.
- **SwinAttention**: (`attention_ops/swin.py`)
    - The windowed self-attention mechanism used in Swin Transformer. It computes self-attention within non-overlapping local windows and enables cross-window information flow through a shifted window mechanism.
    - Includes modes for both regular and shifted windows.
- **NeighborhoodAttention**: (`attention_ops/neighborhood.py`)
    - A local attention mechanism where each token attends only to other tokens within a fixed-size neighborhood in the input sequence.
    - The `radius` parameter controls the size of the neighborhood.

## How to Run Tests

The project includes unit tests for the different Attention mechanisms.

1.  **For Attention mechanisms targeting Language Models (Linformer, Performer, Longformer):**
    ```bash
    python main_test_lm.py
    ```
2.  **For Attention mechanisms targeting Vision Models (Swin, Neighborhood):**
    ```bash
    python main_test_vision.py
    ```

Ensure you have installed all necessary dependencies.

## Dependencies

- PyTorch (ensure your PyTorch version supports FlexAttention, typically the latest stable or nightly build. For example, `torch>=2.4` or newer.)
- `unittest` (Python standard library)

Running in a CUDA environment is recommended for optimal performance and full API support.

## How to Use in Your Project

You can directly import the required Attention modules from the `attention_ops` directory. Here is a simple example demonstrating how to use `LinformerAttention`:

```python
import torch
from attention_ops.linformer import LinformerAttention

# Example parameters
B, H, N, D = 2, 8, 1024, 64  # Batch, Heads, SeqLen, Dim_per_head
M = 128  # Projected length for Linformer

# Create input tensors
Q = torch.randn(B, H, N, D)
K = torch.randn(B, H, N, D)
V = torch.randn(B, H, N, D)

# Initialize LinformerAttention
# Note: the dim parameter should be H * D
attn_model = LinformerAttention(dim=H*D, n_heads=H, seq_len=N, proj_len=M)

# Perform Attention operation
output = attn_model(Q, K, V)

print(output.shape) # Expected output: torch.Size([2, 8, 1024, 64])
```

For other Attention modules, please refer to their respective implementations and parameter settings in the test files.

## Contributing

Contributions to this project are welcome! If you have implemented new Attention variants based on FlexAttention or have suggestions for improving existing implementations, please feel free to submit a Pull Request or create an Issue.

