from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        mean_square = (x * x).mean(dim=-1, keepdim=True)

        y = xf * torch.rsqrt(mean_square + self.eps)
        if self.weight:
            y = y * self.weight
        return  y.to(x.dtype)

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base=100000) -> torch.Tensor:
    # x: (B,H,T,Dh)
    # positions: (T,)

    B, H, T, Dh = x.shape
    original_type = x.dtype

    assert Dh % 2 == 0
    assert positions.numel() == x.shape[-2]

    # 1. Create the indices for each feature pair.
    pair_indices = torch.arange(
        0, Dh, 2, device=x.device, dtype=x.dtype
    )
    # 2. Calculate one inverse frequency per pair
    exponent = pair_indices / Dh
    frequencies = base ** exponent
    inv_freq = 1.0 / frequencies

    # 3. Put positions into a column: (T, 1)
    positions = positions.to(device=x.device, dtype=torch.float32)
    positions_column = positions.unsqueeze(1)

    # 4. Put frequencies into a row: (1, Dh // 2)
    frequencies_row = frequencies.unsqueeze(0)

    # 5. Broadcasting calculates every angle.
    # Result: (T, Dh // 2)
    angles = positions_column * frequencies_row

    # 6: Calculate sine and cosine.
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # 7. Add dimensions for batch and head broadcasting.
    # Both shapes: (1, 1, T, Dh // 2)
    cos_angles = cos_angles.unsqueeze(0).unsqueeze(0)
    sin_angles = sin_angles.unsqueeze(0).unsqueeze(0)

    # 8. Convert to FP32 and group adjacent features into pairs.
    # Shape: (B, H, T, Dh // 2, 2)
    x_pairs = x.float().reshape(B, H, T, Dh // 2, 2)

    # 9. Extract the first and second value of each pair.
    # Both shapes: (B, H, T, Dh // 2)
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    # 10. Rotate every pair
    y1 = (
        x1 * cos_angles - x2 * sin_angles
    )
    y2 = (
        x1 * sin_angles + x2 * cos_angles
    )

    y_pairs = torch.stack(
        [y1, y2],
        dim=-1
    )

    return y_pairs.flatten(-2).to(original_type)

class CachedGQA(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None=None):
        B, T_new, D = x.shape
        past_len = 0 if past_kv is None else past_kv[0].size(-2)

        q = self.q_proj(x).reshape(B, T_new, self.num_q_heads, self.head_dim).transpose(1, 2)

        k_new = self.k_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(past_len, past_len + T_new, device=x.device)
        q = apply_rope(q, positions)
        k_new = apply_rope(k_new, positions)

        if past_kv is None:
            k, v  = k_new, v_new
        else:
            k = torch.cat((past_kv[0], k_new), dim=-2)
            v = torch.cat((past_kv[1], v_new), dim=-2)

        k_attn = k.repeat_interleave(self.group_size, dim=1)
        v_attn = v.repeat_interleave(self.group_size, dim=1)

        scores = (q.float() @ k.float().transpose(-2, -1)) * (self.head_dim ** -0.5)

        q_pos = torch.arange(past_len, past_len + T_new, device=x.device)
        k_pos = torch.arange(k.size(-2), device=x.device)

        allowed = k_pos[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weigths = torch.softmax(scores, dim=-1)
        out = (weigths @ v.float()).to(x.dtype)
        out = out.transpose(1, 2).contiguous().reshape(B, T_new, D)
        return self.out_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, mlp_hidden_dim: int, eps: float = 1e-6):
        super().__init__()

        self.attention_norm = RMSNorm(d_model, eps)
        self.attention = CachedGQA(d_model, num_q_heads, num_kv_heads)
        self.mlp_norm = RMSNorm(d_model, eps)
        self.mlp = SwiGLU(d_model, mlp_hidden_dim)

    def forward(self, x: torch.Tensor, past_kv=None):
        # RMSNorm for attention
        attention_norm_input = self.attention_norm.forward(x)

        # GQA Casual Attention
        attention_output, new_kv = self.attention.forward(attention_norm_input, past_kv)

        # Residual
        h = x + attention_output

        # RMSNorm for MLP
        mlp_norm_input=self.mlp_norm.forward(h)

        # MLP
        y = self.mlp.forward(mlp_norm_input)

        # reidual
        z = h + y
        return x, new_kv
    