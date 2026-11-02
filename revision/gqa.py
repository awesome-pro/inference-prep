from typing import Any

import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()

        assert d_model % num_q_heads == 0 and num_q_heads % num_kv_heads == 0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(num_kv_heads * self.head_dim, d_model, bias=False)


    @staticmethod
    def apply_rope(q: torch.Tensor, positions: torch.Tensor, base: float = 10000):
        B, H, T, Dh = q.shape
        original_dtype = q.dtype

        # pair indices
        pair_indices = torch.arange(0, Dh, 2, device=q.device, dtype=torch.float32)

        # frequencies
        exponent = pair_indices / Dh
        frequencies = base ** -exponent

        positions_column = positions.to(device=q.device, dtype=torch.float32).unsqueeze(1) # (T, 1)
        frequencies_row = frequencies.unsqueeze(0) # (1, Dh // 2)

        # angles 
        angles = positions_column * frequencies_row

        # cos and sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # add dim for broadcasting on B & H
        cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, T, Dh // 2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # extract q1, q2 from the pairs
        pairs = q.float().reshape(B, H, T, Dh // 2, 2)

        q1 = pairs[..., 0]
        q2 = pairs[..., 1]

        y1 = q1 * cos - q2 * sin
        y2 = q1 * sin + q2 * cos

        y_pairs = torch.stack([y1, y2], dim=-1)
        y = y_pairs.flatten(-2)
        return y.to(original_dtype)


    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor]):
        original_type = x.dtype
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_q_heads, self.head_dim).transpose(1, 2) # (B, H_q, T, d)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) # (B, H_kv, T, d)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) # (B, H_kv, T, d)

        # share the KV across the q heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = (
            q.float() @ k.float().transpose(-2, -1)
        ) * (self.head_dim ** -0.5) # (B, H, T, T)

        mask = torch.ones(T, T, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        output = attention @ v.float()

        output = output.to(x.dtype)
        output = output.transpose(1, 2).contiugous().reshape(B, T, D)
        return self.out_proj(output)