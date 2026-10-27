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

        # Causal masking: token i may only attend to j <= i.
        # NOTE: this must be .tril(). A plain torch.ones() masks nothing.
        mask = torch.ones(
            T, T, device=x.device, dtype=torch.bool
        ).tril()

        scores = score.masked_fill(~mask, float("-inf"))

        # stable softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # weighted sum of values
        out = attention_weights @ v.float()

        # shape - (B, H, T, d)

        # Merge heads back together
        out = out.to(x.dtype)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, D)

        return self.out_proj(out)


if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(42)

    model = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    x = torch.randn(2, 4, 8)

    output = model(x)

    print(output.shape)

    B, T, D = x.shape
    H = model.num_heads
    d = model.head_dim

    q = model.q_proj(x).reshape(B, T, H, d).transpose(1, 2)
    k = model.k_proj(x).reshape(B, T, H, d).transpose(1, 2)
    v = model.v_proj(x).reshape(B, T, H, d).transpose(1, 2)

    ref = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, dropout_p=0.0
    )

    ref = ref.transpose(1, 2).contiguous().reshape(B, T, D)
    ref = model.out_proj(ref)

    torch.testing.assert_close(
        output, ref, rtol=1e-4, atol=1e-5
    )

    # Stronger than comparing to SDPA: change the FUTURE and check that the
    # past does not move. This is what actually proves causality.
    y = x.clone()
    y[0, -1] += 100.0
    moved = model(y)
    assert torch.allclose(output[0, :-1], moved[0, :-1], atol=1e-5)
    assert not torch.allclose(output[0, -1], moved[0, -1], atol=1e-5)

    print("Correct!")
