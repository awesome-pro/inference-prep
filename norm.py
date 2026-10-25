import torch

def rms_norm(x: torch.Tensor, weight=None, eps=1e-6):
    original_dtype = x.dtype
    x = x.float()

    mean_square = (x * x).mean(dim=-1, keepdim=True)

    y = x * torch.rsqrt(mean_square + eps)

    if weight is not None:
        y = y * weight

    return y.to(original_dtype)


def layer_norm(x: torch.Tensor, weight=None, bias=None, eps:float=1e-5):
    original_dtype = x.dtype

    x = x.float()

    mean = x.mean(dim=-1, keepdim=True)

    var = ((x - mean) ** 2).mean(
        dim=-1, keepdim=True
    )

    y = (x - mean) * torch.rsqrt(var + eps)

    if weight is not None:
        y = y * weight

    if bias is not None:
        y = y + bias

    return y.to(original_dtype)


torch.manual_seed(42)

x = torch.randn(2, 4, 128, dtype=torch.float32)

weight = torch.randn(128)
bias = torch.randn(128)

# LayerNorm
my_ln = layer_norm(x, weight, bias, eps=1e-5)

ref_ln = torch.nn.functional.layer_norm(
    x, (128,), weight, bias, eps=1e-5
)

torch.testing.assert_close(my_ln, ref_ln)

# RMSNorm
my_rms = rms_norm(x, weight, eps=1e-6)

ref_rms = torch.nn.functional.rms_norm(
    x, (128,), weight, eps=1e-6
)

torch.testing.assert_close(my_rms, ref_rms)

print("Both implementations passed!")
