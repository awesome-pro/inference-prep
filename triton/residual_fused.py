
import torch
import triton
import triton.language as tl


@triton.jit
def fused_residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    N,
    x_stride_row,
    x_stride_col,
    r_stride_row,
    r_stride_col,
    eps,
    BLOCK_N: tl.constexpr,
):
    # calculate the row
    row = tl.program_id(0)

    # calc cols
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # x input offsets
    x_offsets = row * x_stride_row + cols * x_stride_col

    # load the x
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    # calc offset resildual
    residual_offsets = row * r_stride_row + cols * r_stride_col

    # load residual
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)

    # merge
    z = x + residual

    # execute the rmsnorms on the merged
    mean_sqr = tl.sum(z * z, axis=0) / N
    normalized = z * tl.rsqrt(mean_sqr + eps) 

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    result = normalized * weight

    output_offsets = row * N + cols

    tl.store(out_ptr + output_offsets, result, mask=mask)   


def fused_residual_rmsnorm(x, residual, weight, eps=1e-6):
    M, N = x.shape

    out = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype,
    )

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)

    fused_residual_rmsnorm_kernel[grid](x, residual, weight, out, N, x.stride(0), x.stride(1), residual.stride(0), residual.stride(1), eps, BLOCK_N=BLOCK_N)

    return out

def reference_fused_rmsnorm(x, residual, weight, eps=1e-6):

    # Convert to FP32 for accumulation
    merged = x.float() + residual.float()

    # Mean of squares
    mean_sqr = (merged * merged).mean(
        dim=-1,
        keepdim=True
    )

    # Inverse RMS
    inv_rms = torch.rsqrt(mean_sqr + eps)

    # Normalize and apply weight
    result = merged * inv_rms * weight.float()

    return result.to(x.dtype)

def check(x, residual, weight):

    expected = reference_fused_rmsnorm(
        x, residual, weight
    )

    actual = fused_residual_rmsnorm(
        x, residual, weight
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5
    )


# Test 1: Normal contiguous inputs
M, N = 5, 13

x = torch.randn(M, N, device="cuda")
r = torch.randn(M, N, device="cuda")
w = torch.randn(N, device="cuda")

check(x, r, w)


# Test 2: Irregular dimensions
M, N = 7, 37

x = torch.randn(M, N, device="cuda")
r = torch.randn(M, N, device="cuda")
w = torch.randn(N, device="cuda")

check(x, r, w)


# Test 3: Noncontiguous X
M, N = 5, 13

x = torch.randn(M, N * 2, device="cuda")[:, ::2]
r = torch.randn(M, N, device="cuda")
w = torch.randn(N, device="cuda")

check(x, r, w)


# Test 4: Noncontiguous residual
x = torch.randn(M, N, device="cuda")
r = torch.randn(M, N * 2, device="cuda")[:, ::2]
w = torch.randn(N, device="cuda")

check(x, r, w)


# Test 5: Both inputs transposed
x = torch.randn(N, M, device="cuda").T
r = torch.randn(N, M, device="cuda").T
w = torch.randn(N, device="cuda")

check(x, r, w)


# Test 6: Single-column input
x = torch.randn(7, 1, device="cuda")
r = torch.randn(7, 1, device="cuda")
w = torch.randn(1, device="cuda")

check(x, r, w)


print("All tests passed!")



import triton.testing

M, N = 4096, 1024

x = torch.randn(M, N, device="cuda")
r = torch.randn(M, N, device="cuda")
w = torch.randn(N, device="cuda")

# Verify correctness first
torch.testing.assert_close(
    fused_residual_rmsnorm(x, r, w),
    reference(x, r, w),
    rtol=1e-4,
    atol=1e-5,
)

# Warm up
fused_residual_rmsnorm(x, r, w)
reference(x, r, w)

fused_ms = triton.testing.do_bench(
    lambda: fused_residual_rmsnorm(x, r, w)
)

unfused_ms = triton.testing.do_bench(
    lambda: reference(x, r, w)
)

print(f"Fused Triton: {fused_ms * 1000:.2f} us")
print(f"PyTorch reference: {unfused_ms * 1000:.2f} us")
print(f"Speedup: {unfused_ms / fused_ms:.2f}x")
