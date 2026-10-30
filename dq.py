import torch

def quantize_int8(x: torch.Tensor):
    # Convert to FP32 for more accurate scale calculation
    x_fp32 = x.float()

    # Find maximum absolute value
    amax = x_fp32.abs().max()

    # Calculate scale
    scale = amax / 127.0

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    # Quantize
    x_int8 = torch.round(x_fp32 / scale)

    # Keep values within INT8 range
    x_int8 = torch.clamp(x_int8, -127, 127)

    # Convert to INT8
    x_int8 = x_int8.to(torch.int8)

    return x_int8, scale


def dequantize_int8(x_int8: torch.Tensor, scale: torch.Tensor):
    # Convert INT8 to FP32
    x_fp32 = x_int8.float()

    # Recover approximate original values
    x_fp32 = x_fp32 * scale

    # Convert back to BF16
    return x_fp32.to(torch.bfloat16)