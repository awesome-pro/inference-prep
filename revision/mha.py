from typing import Any

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        orginial_dtype = x.dtype
        B, T, D = x.shape

        # TODO: apply rms_norm
        
        # apply projections
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(-2, -1).float() # (B, H, T, d)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(-2, -1).float() # (B, H, T, d)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(-2, -1).float() # (B, H, T, d)

        # TODO: apply RoPE to the Q, K

        # calculate scores
        scores = (
            q @ k.transpose(-2, -1)
        ) * (self.head_dim ** -0.5) # (B, H, T, T)

        # casual masking
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

        # calc softmax
        attention_weight = torch.softmax(scores, dim=-1)

        # output
        output = attention_weight @ v # (B, H, T, d)

        # convert the dim back to the (B, T, D) for next layer
        output = output.transpose(-2, -1).contiguous().reshape(B, T, D)

        return self.out_proj(output.to(orginial_dtype))
