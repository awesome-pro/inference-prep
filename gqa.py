import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int) -> None:
        super().__init__()

        assert d_model % num_q_heads == 0, "d_model must split into heads"
        assert num_q_heads % num_kv_heads == 0, "q heads must group evenly"

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

        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split the heads
        q = q.reshape(B, T, self.num_q_heads, self.head_dim)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        # 3. Transpose to (B, H, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Share KV heads across groups of query heads.
        #    Query head h reads KV head h // group_size, which is exactly
        #    what repeat_interleave(group_size) produces.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = (
            q.float() @ k.float().transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        mask = torch.ones(
            T, T, dtype=torch.bool, device=x.device
        ).tril()

        scores = scores.masked_fill(~mask, float("-inf"))

        # 7. Softmax over the keys
        weights = torch.softmax(scores, dim=-1)

        # 8. Weighted sum of values
        out = weights @ v.float()

        # 9. Merge heads
        out = out.to(x.dtype)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, D)

        # 10. Output projection
        return self.out_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, Hq, Hkv = 2, 6, 32, 8, 2

    m = GQA(D, Hq, Hkv)
    x = torch.randn(B, T, D)
    out = m(x)

    # Reference: repeat the KV heads ourselves, then use torch's kernel.
    q = m.q_proj(x).reshape(B, T, Hq, m.head_dim).transpose(1, 2)
    k = m.k_proj(x).reshape(B, T, Hkv, m.head_dim).transpose(1, 2)
    v = m.v_proj(x).reshape(B, T, Hkv, m.head_dim).transpose(1, 2)
    ref = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(m.group_size, 1),
        v.repeat_interleave(m.group_size, 1),
        is_causal=True,
    )
    ref = m.out_proj(ref.transpose(1, 2).contiguous().reshape(B, T, D))

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    # The thing GQA is actually about: q head h -> kv head h // group_size.
    ids = torch.arange(Hkv).view(1, Hkv, 1, 1).float()
    expanded = ids.repeat_interleave(m.group_size, dim=1)
    for h in range(Hq):
        assert int(expanded[0, h, 0, 0]) == h // m.group_size

    print("gqa ok")
