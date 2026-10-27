from typing import Any

import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # learned Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # final output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        H = self.num_heads
        d = self.head_dim

        # convert form [B,T,D] -> [B,T,H,d] -> [B,H,T,d]
        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)

        # calculate the attention score = QK^T
        score = (
            q.float() @ k.float().transpose(-2, -1)
        ) * (d ** -0.5) # [B,H,T,T]

        # casual masking
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()

        scores = score.masked_fill(~mask, float("-inf"))

        # calculate weighted softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # weighted sum of the values
        out = attention_weights @ v.float() # [B,H,T,d]

        # back to original dtype
        out.to(x.dtype)

        # convert back to [B,T,D]
        out = out.transpose(1,2).contigous()
        out = out.reshape(B, T, D)

        return self.out_proj(out)