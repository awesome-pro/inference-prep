
import torch
from torch import nn

class CachedGQA(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, self.head_dim * num_q_heads, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T_new, D = x.shape

        # 1. Project only the new tokens
        q = self.q_proj(x).reshape(B, T_new, self.num_q_heads, self.head_dim).transpose(1, 2) # [B, num_q_heads, T_new, head_dim]

        k_new = self.k_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2) # [B, num_kv_heads, T_new, head_dim]
        v_new = self.v_proj(x).reshape(B, T_new, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is None:
            k, v = k_new, v_new
        else:
            k_past, v_past = past_kv
            past_len = k_past.size(-2)

            k = torch.cat((k_past, k_new), dim=-2)
            v = torch.cat((v_past, v_new), dim=-2)

        # 3. Expand KV heads for reference GQA attention
        k_attn = k.repeat_interleave(self.group_size, dim=1)
        v_attn = v.repeat_interleave(self.group_size, dim=1)

        # 4. compute scaled attention scores
        scores = (
            q.float() @ k_attn.float().transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        # Cache aware casual mask.
        query_positions = past_len + torch.arange(
            T_new, device=x.device
        )
        key_positions = torch.arange(
            k.size(-2), device=x.device
        )

        allowed = (
            key_positions[None, :] <= query_positions[:, None]
        )
        scores = scores.masked_fill(~allowed, float("-inf"))

        # 6. softmax and weighted sum
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v_attn.float()

        # 7. Merge query heads
        output = output.to(x.dtype)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(B, T_new, D)

        # 8. Return the attention output and updated KV cache
        return self.out_proj(output), (k, v)

