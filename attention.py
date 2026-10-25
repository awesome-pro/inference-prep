from typing import Any

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

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

        q = self.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, d).transpose(1, 2)

        # shape (B, H, T, d)

        score = (
            q.float() @ k.float().transpose(-2, -1)
        ) * (d ** -0.5)

        # shape - (B, H, T, T)

        # Casual masking
        mask = torch.ones(
            T, T, device=x.device, dtype=torch.bool
        )

        scores = score.masked_fill(~mask, float("-inf"))

        # stable softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # weighted sum of values
        out = attention_weights @ v.float()

        # shape - (B, H, T, d)

        # Merge hards back togethar
        out = out.to(x.dtype)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, D)

        return self.out_proj(out)


model = MultiHeadAttention(
    d_model=4096,
    num_heads=32
)

x = torch.randn(2, 4, 4096)

output = model(x)

print(output.shape)
# torch.Size([2, 4, 4096])