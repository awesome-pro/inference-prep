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
   scale = tl.maximum(amax / 127.0, 1e-8)

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


@triton.jit
def dequantize_int8_kernel(
    q_ptr,
    scale_ptr,
    out_ptr,
    N,
    stride_row,
    stride_col,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # calc the q input offsets
    q_offsets = row * stride_row + cols * stride_col

    # load q
    q = tl.load(q_ptr + q_offsets, mask=mask, other=0)

    # convert to fp32
    q = q.to(tl.float32)

    # load the fp32 scales
    scales = tl.load(scale_ptr + row)

    x = q * scales

    # store continougs fp32 output
    out_offsets = row * N + cols
    tl.store(out_ptr + out_offsets, x, mask=mask)

def dequantize_int8(q, scales):
  assert q.ndim == 2
  assert q.dtype == torch.int8

  M, N = q.shape
  assert M > 0 and N > 0

  out = torch.empty(
      (M, N),
      device=q.device,
      dtype=torch.float32
  )

  BLOCK_N = triton.next_power_of_2(N)

  grid = (M,)

  dequantize_int8_kernel[grid](
      q, scales, out, N, x.stride(0), x.stride(1), BLOCK_N
  )

  return out


def check_dequant(q, scales):
    result = dequantize_int8(q, scales)
    expected = q.float() * scales[:, None]

    torch.testing.assert_close(
        result, expected,
        rtol=1e-6,
        atol=1e-6,
    )


# 1. Normal
x = torch.randn(32, 768, device="cuda")
q, scales = quantize_int8(x)
check_dequant(q, scales)

# 2. Irregular width
x = torch.randn(16, 777, device="cuda")
q, scales = quantize_int8(x)
check_dequant(q, scales)

# 3. All-zero rows
x = torch.zeros(8, 128, device="cuda")
q, scales = quantize_int8(x)
check_dequant(q, scales)

# 4. Noncontiguous INT8 tensor
q = torch.randint(
    -127, 128, (777, 16),
    device="cuda", dtype=torch.int8
).T

scales = torch.rand(16, device="cuda") + 0.01
check_dequant(q, scales)

print("All dequantization tests passed!")

