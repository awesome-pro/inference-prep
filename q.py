import torch

def quantize_int(x: torch.Tensor):
    # Convert to fp32 for more precision.
    x_fp32 = x.float()

    amax = x.abs().max()

    scale = amax / 127.0

    scale = torch.clamp(amax, min=1e-8)

    x_int8 = torch.round(x_fp32 / scale)

    x_int8 = torch.clamp(x_fp32, min=-127, max=127)

    # conver to int8
    x_int8 = x_int8.to(torch.int8)

    return x_int8, scale

def dequantize(x_int8: torch.Tensor, scale: torch.Tensor):
    x_fp32 = x_int8.float()

    x_fp32 = x_fp32 * scale

    return x_fp32.to(torch.bfloat16)
