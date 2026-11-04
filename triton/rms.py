import torch
import triton
import triton.language as tl # type: ignore

@triton.jit
def rms_norm_kernel(
    X, 
    W, 
    O, 
    N: tl.constexpr, 
    eps: tl.constexpr,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr
):
    # each program processes one row
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # input offsets
    input_offets = row * stride_row + cols * stride_col

    # load the input x, weight and eps
    x = tl.load(X + input_offets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    mean_square  = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square  + eps)
    y = x * inv_rms * weight

    output_offsets = row * N + cols

    tl.store(
        O + output_offsets, 
        y, 
        mask=mask
    )



def rms_norm(x, weights, eps):
    assert x.ndim == 2
    assert weights.ndim == 1
    assert x.size(1) == weights.size(0)
    assert x.device == w.device

    M, N = x.shape

    output = torch.zeros(
        (M, N),
        device=x.device,
        dtype=x.dtype
    )

    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (M,)

    rms_norm_kernel[grid](
        x, weights, eps, output, N, x.stride(0), x.stride(1), BLOCK_SIZE
    )

    return output
