from typing import Any

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-8):
    original_dtype = x.dtype
    x = x.float()
    mean_square = torch.mean(x * x, dim=-1, keepdim=True)
    y = x * torch.rsqrt(mean_square + eps)
    if weight is not None:
        y = y * weight
    return y.to(original_dtype)

def layer_norm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float=1e-6):
    o_type = x.dtype
    x = x.float()

    mean = x.mean(dim=-1, keepdim=True)
    variance = torch.sum((x - mean) ** 2, dim=-1, keepdim=True)
    y = (x  - mean) * torch.rsqrt(variance + eps)

    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y.to(o_type)