
import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_row,
    stride_col,
    BLOCK_N: tl.constexpr,
):
    # Identify the row
    row = tl.program_id(0)

    # Generate column indices
    cols = tl.arange(0, BLOCK_N)

    # Create mask
    mask = cols < N

    # Input offsets (respect input strides)
    offsets = row * stride_row + cols * stride_col

    # Load input
    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=float("-inf")
    )

    # Maximum reduction
    max_r = tl.max(x, axis=0)

    # Subtract maximum and exponentiate
    exp_values = tl.exp(x - max_r)

    # Sum reduction
    sum_r = tl.sum(exp_values, axis=0)

    # Normalize
    result = exp_values / sum_r

    # Output offsets (contiguous output)
    out_offsets = row * N + cols

    # Store using OUTPUT offsets
    tl.store(
        out_ptr + out_offsets,
        result,
        mask=mask
    )


def softmax(x):
    M, N = x.shape

    out = torch.empty(
        (M, N),
        device=x.device,
        dtype=x.dtype
    )

    # TODO: Choose block size
    block_size = triton.next_power_of_2(N)

    # TODO: Construct grid
    grid = (M,)

    softmax_kernel[grid](x, out, N, x.stride(0), x.stride(1), BLOCK_N=block_size)

    return out


def check(x):
    expected = torch.softmax(x, dim=-1)
    actual = softmax(x)

    torch.testing.assert_close(
        actual, expected,
        rtol=1e-4, atol=1e-5
    )

    assert torch.isfinite(actual).all()

    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(
            x.shape[0],
            device=x.device,
            dtype=x.dtype
        ),
        rtol=1e-4, atol=1e-5
    )


# 1. Normal + irregular dimensions
check(torch.randn(5, 13, device="cuda"))

# 2. Negative-only values
check(-torch.rand(7, 19, device="cuda") * 100)

# 3. Large logits (numerical stability)
check(torch.randn(8, 37, device="cuda") * 1000)

# 4. Noncontiguous input
check(torch.randn(5, 26, device="cuda")[:, ::2])

# 5. Transposed input
check(torch.randn(13, 5, device="cuda").T)

# 6. Single-column input
check(torch.randn(7, 1, device="cuda"))

print("All tests passed!")




def benchmark(M, N):
    x = torch.randn(M, N, device="cuda")

    # Compile and warm up both paths
    softmax(x)
    torch.softmax(x, dim=-1)

    triton_ms = triton.testing.do_bench(
        lambda: softmax(x)
    )

    torch_ms = triton.testing.do_bench(
        lambda: torch.softmax(x, dim=-1)
    )

    print(f"Shape: ({M}, {N})")
    print(f"Triton:  {triton_ms * 1000:.2f} us")
    print(f"PyTorch: {torch_ms * 1000:.2f} us")
    print(f"Speedup: {torch_ms / triton_ms:.2f}x")
    print()


for N in [128, 512, 1024, 4096]:
    benchmark(M=4096, N=N)
