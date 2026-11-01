import torch
import triton
import triton.language as tl

@triton.jit
def fused_swiglu_residual_kernel(
    x_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    N,
    stride_x_row,
    stride_x_col,
    stride_r_row,
    stride_r_col,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    gate_offsets = row * stride_x_row + cols * stride_x_col
    up_offsets = row * stride_x_row + (N + cols) * stride_x_col

    gate = tl.load(x_ptr + gate_offsets, mask=mask, other=0.0)
    up = tl.load(x_ptr + up_offsets, mask=mask, other=0.0)


    bias_gate = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    bias_up = tl.load(bias_ptr + N + cols, mask=mask, other=0.0)

    gate = gate.to(tl.float32) + bias_gate.to(tl.float32)
    up = up.to(tl.float32) + bias_up.to(tl.float32)

    # silu for the x_gate
    activated_gate = gate * tl.sigmoid(gate)

    # load the residual
    r_offsets = row * stride_r_row + cols * stride_r_col
    residual = tl.load(residual_ptr + r_offsets, mask=mask, other=0.0).to(tl.float32)

    final = (activated_gate * up) + residual

    out_offsets = row * N + cols
    tl.store(out_ptr + out_offsets, final, mask=mask)


def fused_swiglu_residual(x, bias, residual):
    # Your Triton kernel and wrapper
    assert x.ndim == 2 and residual.ndim == 2
    assert x.shape[1] == bias.shape[0] and x.shape[1] == 2 * residual.shape[1]
    assert x.dtype == bias.dtype and x.dtype == residual.dtype
    assert x.shape[1] % 2 == 0

    M, N_2 = x.shape
    N = N_2 // 2

    out = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype
    )

    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)

    fused_swiglu_residual_kernel[grid](
        x, bias, residual, out, N, x.stride(0), x.stride(1), residual.stride(0), residual.stride(1), BLOCK_N
    )

    return out



def reference(x, bias, residual):
    M, twice_n = x.shape
    N = twice_n // 2

    xf = x.float()
    bf = bias.float()
    rf = residual.float()

    gate = xf[:, :N] + bf[:N]
    up = xf[:, N:] + bf[N:]

    silu = gate * torch.sigmoid(gate)

    return (silu * up + rf).to(x.dtype)


def check(x, bias, residual):
    expected = reference(x, bias, residual)
    actual = fused_swiglu_residual(x, bias, residual)

    assert actual.shape == expected.shape
    assert actual.dtype == x.dtype
    assert actual.is_contiguous()

    if x.dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-4

    torch.testing.assert_close(
        actual, expected,
        atol=atol, rtol=rtol
    )


# Case 1: Regular FP32
M, N = 32, 256
x = torch.randn(M, 2*N, device="cuda")
bias = torch.randn(2*N, device="cuda")
residual = torch.randn(M, N, device="cuda")
check(x, bias, residual)

# Case 2: Irregular N
M, N = 17, 777
x = torch.randn(M, 2*N, device="cuda")
bias = torch.randn(2*N, device="cuda")
residual = torch.randn(M, N, device="cuda")
check(x, bias, residual)

# Case 3: FP16
M, N = 64, 1024
x = torch.randn(
    M, 2*N, device="cuda", dtype=torch.float16
)
bias = torch.randn(
    2*N, device="cuda", dtype=torch.float16
)
residual = torch.randn(
    M, N, device="cuda", dtype=torch.float16
)
check(x, bias, residual)

# Case 4: Noncontiguous inputs
M, N = 16, 127
x = torch.randn(2*N, M, device="cuda").T
bias = torch.randn(2*N, device="cuda")
residual = torch.randn(N, M, device="cuda").T
check(x, bias, residual)

# Case 5: Zero gate and up values
M, N = 8, 33
x = torch.zeros(M, 2*N, device="cuda")
bias = torch.zeros(2*N, device="cuda")
residual = torch.randn(M, N, device="cuda")
check(x, bias, residual)

print("All tests passed!")
