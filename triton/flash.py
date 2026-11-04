
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # One program per query tile and batch/head
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Assume contiguous (B, H, N, D)
    base = pid_bh * N * D

    # Query positions and head dimensions
    m_idx = pid_m * BM + tl.arange(0, BM)
    d_idx = tl.arange(0, D)

    # Load query tile: (BM, D)
    q_ptrs = (
        Q + base
        + m_idx[:, None] * D
        + d_idx[None, :]
    )
    q = tl.load(
        q_ptrs,
        mask=m_idx[:, None] < N,
        other=0.0,
    )

    # Online softmax state, per query row
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, D), dtype=tl.float32)

    # Causal: visit only potentially useful KV tiles
    end_n = tl.minimum(N, (pid_m + 1) * BM)

    for start_n in range(0, end_n, BN):
        n_idx = start_n + tl.arange(0, BN)

        # Load K and V tiles: (BN, D)
        k_ptrs = (
            K + base
            + n_idx[:, None] * D
            + d_idx[None, :]
        )
        v_ptrs = (
            V + base
            + n_idx[:, None] * D
            + d_idx[None, :]
        )

        k = tl.load(
            k_ptrs,
            mask=n_idx[:, None] < N,
            other=0.0,
        )
        v = tl.load(
            v_ptrs,
            mask=n_idx[:, None] < N,
            other=0.0,
        )

        # 1. QK^T: (BM, BN)
        scores = tl.dot(q, tl.trans(k)) * SCALE

        # 2. Causal + sequence boundary mask
        valid = (
            (m_idx[:, None] < N)
            & (n_idx[None, :] < N)
            & (n_idx[None, :] <= m_idx[:, None])
        )

        scores = tl.where(
            valid, scores, float("-inf")
        )

        # 3. Online softmax maximum
        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, tile_max)

        # Handle rows with no valid keys
        safe_m = tl.where(
            m_new == float("-inf"),
            0.0,
            m_new,
        )

        # 4. Rescale old contributions
        alpha = tl.exp(m - safe_m)

        # 5. New unnormalized probabilities
        p = tl.exp(scores - safe_m[:, None])

        # 6. Update denominator
        l = alpha * l + tl.sum(p, axis=1)

        # 7. Update weighted output
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        m = m_new

    # 8. Normalize output
    denominator = tl.where(l > 0, l, 1.0)
    out = acc / denominator[:, None]

    # 9. Store final output tile
    o_ptrs = (
        O + base
        + m_idx[:, None] * D
        + d_idx[None, :]
    )

    tl.store(
        o_ptrs,
        out,
        mask=m_idx[:, None] < N,
    )


def flash_attention_triton(q, k, v):
    assert q.shape == k.shape == v.shape
    assert q.ndim == 4
    assert q.device.type == "cuda"
    assert q.dtype == torch.float16
    assert k.dtype == v.dtype == q.dtype

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, N, D = q.shape

    assert N > 0
    assert D in (32, 64, 128)

    out = torch.empty_like(q)

    BM = 32
    BN = 32

    grid = (
        triton.cdiv(N, BM),
        B * H,
    )

    flash_attention_kernel[grid](
        q, k, v, out,
        N=N,
        D=D,
        SCALE=D ** -0.5,
        BM=BM,
        BN=BN,
        num_warps=4,
    )

    return out
