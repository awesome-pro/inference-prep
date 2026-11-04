from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-9) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        if self.weight:
            y = y * self.weight
        return y.to(x.dtype)

def apply_rope(q: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    B, H, T, Dh = q.shape
    original_dtype = q.dtype

    # pair indices
    pair_indices = torch.arange(0, Dh, 2, device=q.device, dtype=torch.float32)

    # exponent
    exponent = pair_indices / Dh
    frequencies = base ** -exponent

    positions_column = positions.to(device=q.device, dtype=torch.float32).unsqueeze(1)
    frequencies_row = frequencies.unsqueeze(0)

    angles = positions_column * frequencies_row

    # cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # add dim for broadcasting on B & H
    cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, T, Dh // 2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # extract q1, q2 from pairs
    pairs = q.float().reshape(B, H, T, Dh // 2, 2)

    q1 = pairs[..., 0]
    q2 = pairs[..., 1]

    y1 = q1 * cos - q2 * sin
    y2 = q1 * sin + q2 * cos

    y_pairs = torch.stack([y1, y2], dim=-1)
    y = y_pairs.flatten(-2)
    return y.to(original_dtype)


class FlashAttentionGQA(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, tile_size: int = 32) -> None:
        super().__init__()

        assert d_model % num_q_heads == 0 and num_q_heads % num_kv_heads == 0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.tile_size = tile_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(num_kv_heads * self.head_dim, d_model, bias=False)

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v:torch.Tensor, past_len: int) -> torch.Tensor:
        B, H, Tq, d = q.shape
        Tkv = k.size(-2)
        scale = d ** -0.5

        output_blocks = []
        

        for q_start in range(0, Tq, self.tile_size):
            q_end = min(q_start + self.tile_size, Tq)
            q_block = q[:, :, q_start: q_end, :]

            # tile length
            bq = q_end - q_start

            # online softmax state
            m = torch.full(
                (B, H, bq),
                float("-inf"),
                device=q.device,
                dtype=q.dtype
            )
            l = torch.zeros_like(m)
            acc = torch.zeros(
                B, H, bq, d,
                device=q.device,
                dtype=q.dtype
            )

            # absolute query positions
            q_pos = past_len + torch.arange(q_start, q_end, device=q.device)

            for k_start in range(0, min(Tkv, past_len + q_end), self.tile_size):
                k_end = min(k_start + self.tile_size, Tkv)

                k_block = k[:, :, k_start:k_end, :]
                v_block = v[:, :, k_start:k_end, :]

                # (B, H, bq, bk)
                scores = (
                    q_block @ k_block.transpose(1, 2)
                ) * scale

                # casual masking
                k_pos = torch.arange(k_start, k_end, device=q.device)

                mask = k_pos[None, :] > q_pos[:, None]

                scores = scores.masked_fill(
                    mask[None, None, :, :],
                    float("-inf")
                )

                # online softmax
                m_new = torch.maximum(m, scores.max(dim=-1).values)

                safe_m = torch.where(
                    torch.isfinite(m_new),
                    m_new,
                    torch.zeros_like(m_new)
                )

                alpha = torch.exp(m - safe_m)
                p = torch.exp(scores - safe_m[..., None])

                l = alpha * l + p.sum(dim=-1)

                acc = (
                    alpha[..., None] * acc + p @ v_block
                )

                m = m_new

            # normalize current current query block
            output_block = acc / l.clamp_min(1e-20)[..., None]
            output_blocks.append(output_block.to(q.dtype))

        return torch.cat(output_blocks, dim=2)
                
    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T_new, D = x.shape

        past_len = 0 if past_kv is None else past_kv[0].size(-2)

        q = self.q_proj(x).reshape(B, T_new, self.num_q_heads, self.head_dim).transpose(1, 2)

        k_new = self.k_proj(x).reshape(B, past_len + T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(past_len, past_len + T_new, device=x.device)
        q = apply_rope(q, positions)
        k_new = apply_rope(k_new, positions)

        if past_kv is None:
            k, v = k_new, v_new
        else:
            k = torch.cat((past_kv[0], k_new), dim=-2)
            v = torch.cat((past_kv[1], v_new), dim=-2)

        # repeat kv heads for GQA
        k_attn = k.repeat_interleave(self.group_size, dim=1)
        v_attn = v.repeat_interleave(self.group_size, dim=1)

        output = self._flash_attention(q, k_attn, v_attn, past_len)
        output = output.transpose(1, 2).contiguous().reshape(B, T_new, D)
        return self.out_proj(output), (k, v)

    