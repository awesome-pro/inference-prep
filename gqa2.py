from typing import Any

import torch
import torch.nn as nn

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

        q = self.q_proj(x) #[d_model, d_model]
        k = self.k_proj(x) # [d_model, head_dim * num_kv_heads]
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_q_heads, self.head_dim).transpose(1, 2) #[B, num_q_heads, T, head_dim]
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2) #[B, num_kv_heads, T, head_dim]
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # share the KV heads across the groups of the query heads. 
        # Query heads h reads the  h // group_size, which is repear_interleave(group_size)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # calculate attention scores
        score = (
            q.float() @ k.float().transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        # casual masking
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = score.masked_fill(~mask, float("-inf"))

        # calc softmax
        weights = torch.softmax(scores, dim=-1)

        # weighted sum of values
        out = weights @ v.float()

        out = out.to(x.dtype)

        out = out.transpose(1, 2).contigous() # [B, T, num_q_heads, head_dim]
        out = out.reshape(B, T, D)

        return self.out_proj(out)

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D
        