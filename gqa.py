from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        # Q projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # K/V projection
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)

        # Output Projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split the heads
        q = q.reshape(B, T, self.num_q_heads, self.head_dim)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        # 3. Transpose to (B, H, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Share KV heads across groups of query heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = (
            q.float() @ k.float().transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        mask = torch.ones(
            T, T, dtype=torch.bool, device=x.device
        ).tril()

        scores = scores.masked_fill(~mask, float("-inf"))

        # 7. Stable softmax
        out = scores @ v.float()

        # 9. Merge heads
        out = out.to(x.dtype)
        out = out.transpose(1, 2).contigous()
        out = out.reshape(B, T, D)

        # 10. Output projection
        return self.out_proj(out)