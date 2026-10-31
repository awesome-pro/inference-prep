
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    N,
    stride_row,
    stride_col,
    eps,
    BLOCK_N: tl.constexpr,
):
    # 1. Identify row
    row = tl.program_id(0)

    # 2. Generate columns and mask
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 3. Calculate input offsets using strides
    input_offsets = row * stride_row + cols * stride_col

    # 4. Load x and convert to FP32
    x = tl.load(x_ptr + input_offsets, mask=mask, other=0)
    x = x.to(tl.float32)

    # 5. Calculate mean of squared values
    # IMPORTANT: divide by N, not BLOCK_N
    mean_x_sqr = tl.sum(x * x, axis=0) / N

    # 6. Calculate inverse RMS using tl.rsqrt
    normalized = x * tl.rsqrt(mean_x_sqr + eps)

    # 7. Load weight and convert to FP32
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 8. Normalize and apply weight
    rms_norm = normalized * weight

    # 9. Store to contiguous output
    output_offsets = row * N + cols

    tl.store(out_ptr + output_offsets, rms_norm, mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    M, N = x.shape

    out = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype
    )

    # TODO: Calculate BLOCK_N
    block_n = triton.next_power_of_2(N)

    # TODO: Construct launch grid
    grid = (M,)

    # TODO: Launch kernel
    rmsnorm_kernel[grid](x, weight, out, N, x.stride(0), x.stride(1), eps, BLOCK_N=block_n)

    return out



def rmsnorm_reference(x, weight, eps=1e-6):
    xf = x.float()
    wf = weight.float()

    inv_rms = torch.rsqrt(
        xf.square().mean(dim=-1, keepdim=True) + eps
    )

    return (xf * inv_rms * wf).to(x.dtype)


def check(x, weight):
    actual = rmsnorm(x, weight)
    expected = rmsnorm_reference(x, weight)

    if x.dtype == torch.float16:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 1e-5, 1e-4

    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol
    )


# 1. Standard input
x = torch.randn(32, 768, device="cuda")
w = torch.randn(768, device="cuda")
check(x, w)

# 2. Irregular hidden dimension
x = torch.randn(16, 777, device="cuda")
w = torch.randn(777, device="cuda")
check(x, w)

# 3. All-zero input
x = torch.zeros(8, 128, device="cuda")
w = torch.randn(128, device="cuda")
check(x, w)

# 4. Noncontiguous input
x = torch.randn(16, 1536, device="cuda")[:, ::2]
w = torch.randn(768, device="cuda")
check(x, w)

# 5. Transposed input
x = torch.randn(768, 16, device="cuda").T
w = torch.randn(768, device="cuda")
check(x, w)

# 6. FP16 input
x = torch.randn(
    32, 768, device="cuda", dtype=torch.float16
)
w = torch.randn(
    768, device="cuda", dtype=torch.float16
)
check(x, w)

print("All RMSNorm tests passed!")