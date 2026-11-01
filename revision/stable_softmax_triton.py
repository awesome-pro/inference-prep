import torch
import triton
import triton.language as tl


@triton.jit
def stable_softmax_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    amax = tl.max(x, axis=0)
    shifted = x - amax
    exp_values = tl.exp(shifted)
    d = tl.sum(exp_values)
    output = exp_values / d
    tl.store(out_ptr + offsets, output, mask=mask)

def stable_softmax(x):
    N = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    stable_softmax_kernel[(1,)](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    return out


# For 2D

import torch
import triton
import triton.language as tl


@triton.jit
def stable_softmax_kernel(x_ptr, out_ptr, N: tl.constexpr, stride_row, stride_col, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    input_offsets = row * stride_row + cols * stride_col
    x = tl.load(x_ptr + input_offsets, mask=mask, other=float("-inf")).to(tl.float32)

    amax = tl.max(x, axis=0)
    shifted = x - amax
    exp_values = tl.exp(shifted)

    output = exp_values / exp_values.sum()

    output_offsets = row * N + cols

    tl.store(out_ptr + output_offsets, output, mask=mask)


def stable_softmax(x):
    M, N = x.shape
    out = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype
    )

    BLOCK_SIZE = triton.next_power_of_2(N)

    stable_softmax_kernel[(M,)](x, out, N, x.stride(0), x.stride(1), BLOCK_SIZE=BLOCK_SIZE)

    return out