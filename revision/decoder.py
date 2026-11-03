from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-10) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        if self.weight:
            y = y * self.weight
        return y.to(x.dtype)

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


    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T_new, D = x.shape
        past_len = 0 if past_kv is None else past_kv[0].size(-2)

        q = self.q_proj(x).reshape(B, T_new, self.num_q_heads, self.head_dim).transpose(1, 2) # (B, H_q, T_new, d)

        k_new = self.k_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2) # (B, H_kv, T_new, d)
        v_new = self.v_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(past_len, past_len + T_new, device=x.device)
        q = apply_rope(q=q, positions=positions)
        k_new = apply_rope(q=k_new, positions=positions)

        # build the complete KV
        if past_kv is None:
            k, v = k_new, v_new
        else:
            k = torch.cat((past_kv[0], k_new), dim=-2) # across T - tokens
            v = torch.cat((past_kv[1], v_new), dim=-2)

        # repeat interleave
        k_attn = k.repeat_interleave(self.group_size, dim=1) # across H - heads
        v_attn = v.repeat_interleave(self.group_size, dim=1)

        # calculate the scores
        scores = (q.float() @ k_attn.float().transpose(-2, -1)) * (self.head_dim ** -0.5)

        # casual mask
        q_pos = torch.arange(past_len, past_len + T_new, device=x.device)
        k_pos = torch.arange(k.size(-2), device=x.device)

        allowed = k_pos[None, :] <= q_pos[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1) # across d, - hidden dimension
        out = (attention_weights @ v_attn.float()).to(x.dtype)
        out = out.transpose(1, 2).contiguous().reshape(B, T_new, D)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_hidden_dim: int) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(d_model, mlp_hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, mlp_hidden_dim: int, eps: float=1e-8) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.attention = GroupedQueryAttention(d_model, num_q_heads, num_kv_heads)
        self.mlp_norm = RMSNorm(d_model, 1e-12)
        self.mlp = MLP(d_model=d_model, mlp_hidden_dim=mlp_hidden_dim)


    def forward(self, x: torch.Tensor, past_kv=None):
        # RMSNorm for attention
        attention_normalized_input = self.attention_norm.normalize(x)

        # GQA casual attention
        attention_output, new_kv = self.attention.forward(attention_normalized_input, past_kv)

        # residual add
        h = x + attention_output

        # RMSNorm for MLP
        mlp_norm_input = self.mlp_norm.normalize(h)

        #MLP
        mlp_output = self.mlp.forward(mlp_norm_input)

        # residual add
        z = h + mlp_output
        return z, new_kv
