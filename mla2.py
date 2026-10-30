from typing import Any

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    d = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angle = positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos = angle.cos()[None, None, :, :]
    sin = angle.sin()[None, None, :, :]
    pairs = x.float().reshape(*x.shape[:-1], d // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class MLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, kv_rank: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_rank = kv_rank

        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        self.kv_down = nn.Linear(d_model, kv_rank, bias=False)

        self.k_up = nn.Linear(kv_rank, d_model, bias=False)
        self.v_up = nn.Linear(kv_rank, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        # 1. queries
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, d)

        # 2, compressed KV: (B, T, R)
        c = self.kv_down(x)

        # 3. get up-projection weight per head
        Wk = self.k_up.weight.reshape(self.num_heads, self.head_dim, self.kv_rank) # (H, d, R)
        Wv = self.v_up.weight.reshape(self.num_heads, self.head_dim, self.kv_rank) # (H, d, R)

        # Absorb K projection into queries
        q_abs = q @ Wk   # (B, H, T, R)

        # 4. attention direcly over compressed KV
        scores = q_abs @ c.transpose(1, 2).unsqueeze(1)
        scores = scores * (self.head_dim ** -0.5) # (B, H, T, T)

        # 6. Casual masking + softmax
        mask  = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)

        # 7. Aggregate compressed latents, then expand V.
        z = attention @ c.unsqueeze(1)
        out = z @ Wv.transpose(-2, -1)

        out = out.transpose(1, 2).contingous().reshape(B, T, D)
        return self.out_proj(out)


