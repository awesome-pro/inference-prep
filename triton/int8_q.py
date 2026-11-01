
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def quantize_int8_kernel(
    x_ptr,
    q_ptr,
    scales_ptr,
    N,
    stride_row,
    stride_col,
    BLOCK_N: tl.constexpr,
):
   row = tl.program_id(0)

   cols = tl.arange(0, BLOCK_N)
   mask = cols < N

   x_offsets = row * stride_row + cols * stride_col
   x = tl.load(x_ptr + x_offsets, mask=mask, other=0).to(tl.float32)

   # calc amax
   amax = tl.max(tl.abs(x), axis=0)

   # scale
   scale = max(amax / 127.0, 1e-8)

   # quantize
   q_float = libdevice.round(x / scale)
   q = tl.clamp(q_float, min=-127.0, max=127.0).to(tl.int8)

   # store the output & scale
   # offsets
   out_offsets = row * N + cols
   tl.store(q_ptr + out_offsets, q, mask=mask)
   tl.store(scales_ptr + row, scale)


def quantize_int8(x):
    assert x.ndim == 2
    assert x.is_cuda
    assert x.dtype in (torch.float16, torch.float32)

    M, N = x.shape
    assert M > 0 and N > 0

    q = torch.empty((M, N), device=x.device, dtype=torch.int8)

    scales = torch.empty(
        (M,),
        device=x.device,
        dtype=torch.float32
    )

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)

    quantize_int8_kernel[grid](
        x, q, scales, N, x.stride(0), x.stride(1), BLOCK_N
    )

    return q, scales


def quantize_reference(x):
    xf = x.float()

    amax = xf.abs().amax(dim=1)
    scales = (amax / 127.0).clamp_min(1e-8)

    z = xf / scales[:, None]

    # Round to nearest, ties away from zero
    rounded = torch.sign(z) * torch.floor(z.abs() + 0.5)

    q = rounded.clamp(-127, 127).to(torch.int8)

    return q, scales


def check(x):
    q, scales = quantize_int8(x)
    q_ref, s_ref = quantize_reference(x)

    assert q.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert q.shape == x.shape
    assert scales.shape == (x.shape[0],)

    torch.testing.assert_close(
        scales, s_ref, rtol=1e-5, atol=1e-7
    )

    # Reconstruct original values
    restored = q.float() * scales[:, None]

    # Each value's ideal rounding error is <= scale/2.
    # Allow a small numerical margin.
    error = (restored - x.float()).abs()
    allowed = scales[:, None] * 0.501 + 1e-5

    assert torch.all(error <= allowed).item()

    print("Passed:", x.shape, x.dtype, x.stride())


# Test 1: Small example
x = torch.tensor(
    [[-2.0, -0.8, 0.0, 0.8, 2.0]],
    device="cuda",
    dtype=torch.float32,
)
q, s = quantize_int8(x)
assert q.tolist() == [[-127, -51, 0, 51, 127]]
check(x)

# Test 2: FP32 regular shape
check(torch.randn(32, 768, device="cuda"))

# Test 3: Irregular N
check(torch.randn(16, 777, device="cuda"))

# Test 4: All zero rows
check(torch.zeros(8, 128, device="cuda"))

# Test 5: Noncontiguous input
x = torch.randn(16, 1536, device="cuda")[:, ::2]
check(x)

# Test 6: FP16 input
check(torch.randn(
    32, 1024, device="cuda", dtype=torch.float16
))

print("All INT8 quantization tests passed!")