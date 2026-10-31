
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


M, N = 4096, 1024
x = torch.randn(M, N, device="cuda")
out = torch.empty_like(x)

block_n = triton.next_power_of_2(N)

def run_kernel():
    softmax_kernel[(M,)](
        x, out, N,
        x.stride(0), x.stride(1),
        BLOCK_N=block_n,
    )

run_kernel()  # Compile and warm up

kernel_ms = triton.testing.do_bench_cudagraph(
    run_kernel
)

print(f"Kernel time: {kernel_ms * 1000:.2f} us")


