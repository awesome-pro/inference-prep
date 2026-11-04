from typing import Any

import torch
import torch.nn as nn


class FlashAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 32, tile_size: int = 32) -> None:
        super().__init__()
        self.tile_size = tile_size

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)


    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Q, K, V: (B, H, T, d)
        B, H, T, d = q.shape
        scale = d ** -0.5

        output_blocks = []

        for q_start in range(0, T, self.tile_size):
            q_end = min(q_start + self.tile_size, T)
            q_block = q[:, :, q_start: q_end, :]

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

            # Process KV blocks
            for k_start in range(0, q_end, self.tile_size):
                k_end = min(k_start + self.tile_size, T)

                k_block = k[:, :, k_start: k_end, :]
                v_block = v[:, :, k_start: k_end, :]

                # (B, H, bq, bk)
                scores = (
                    q_block @ k_block.transpose(-2, -1)
                ) * scale

                # casual masking
                q_pos = torch.arange(q_start, q_end, device=q.device)
                k_pos = torch.arange(k_start, k_end, device=k.device)

                mask = k_pos[None, :] <= q_pos[:, None]

                scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

                # online softmax
                m_new = torch.maximum(m, scores.max(dim=-1).values)

                alpha = torch.exp(m - m_new)
                p = torch.exp(scores - m_new[..., None])

                l = alpha * l + p.sum(dim=-1)

                acc = (
                    alpha[..., None] * alpha + p @ v_block
                )

                m = m_new

            # normalize current query block
            out_block = acc / l[..., None]
            output_blocks.append(out_block)
    
        return torch.cat(output_blocks, dim=2)



    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor]):
        B, T, D = x.shape

        # TODO: apply rms_norm
                
        # apply projections
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).float() # (B, H, T, d)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).float() # (B, H, T, d)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).float() # (B, H, T, d)

        # TODO: apply RoPE to the Q, K

        # FlashAttention
        output = self._flash_attention(q, k, v)

        # (B, H, T, d) -> (B, T, D)
        output = output.transpose(1, 2).contiguous().reshape(B, T, D)

        return self.out_proj(output)
        



