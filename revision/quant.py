import torch


def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.float()

    amax = x.abs().max()
    scale = torch.clamp(amax / 127.0, min=1e-8)

    x_int8 = torch.round(x_float / scale)
    x_int8 = torch.clamp(x_int8, min=-127.0, max=127)
    x_int8 =x_int8.to(torch.int8)

    return x_int8, scale

def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q_float = q.float()
    x = q_float * scale
    return x.to(torch.bfloat16)