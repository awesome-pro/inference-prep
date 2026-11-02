import torch
import torch.nn as nn


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    # dim, assuming MHA
    B, T, D = q.shape
    assert q.shape == k.shape and q.shape == v.shape # (all of these has the same shape in case of the MHA)
    H = D // head_dim # number of the heads

    # [B, T, D] -> convert to the (B, H, T, head_dim)
    q = q.reshape(B, T, H, head_dim).transpose(1, 2) # ()
    k = k.reshape(B, T, H, head_dim).transpose(1, 2)
    v = v.reshape(B, T, H, head_dim).transpose(1, 2)

    # [B, H, T, T]
    scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)

    mask = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()

    scores = scores.masked_fill(~mask, float("-inf"))

    output = torch.softmax(scores, dim=-1) @ v

    return output.transpose(1, 2).contiguous().view(B, T, D)


def casual_attention(q, k, v):
    # q, k, v: (B, H, T, d)

    _, _, T, d = q.shape

    scores = (
        q.float() @ k.float().transpose(-2. -1)
    ) * (d ** -0.5)

    mask = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()

    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    output = weights @ v.float()

    return output.to(q.dtype)