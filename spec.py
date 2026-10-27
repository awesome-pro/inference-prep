
import torch


def verify_speculative(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    generator=None,
):
    """
    draft_tokens: (K,)
    draft_probs:  (K, V)
    target_probs: (K + 1, V)

    Probabilities must be normalized and in FP32.

    Returns:
        emitted: list of output token IDs
        accepted_count: number of accepted draft tokens
        status: "rejected" or "all_accepted"
    """
    K = draft_tokens.numel()

    assert draft_tokens.ndim == 1 and K > 0
    assert draft_probs.ndim == 2

    V = draft_probs.shape[-1]
    assert draft_probs.shape == (K, V)
    assert target_probs.shape == (K + 1, V)

    emitted = []

    # 1. Check proposed tokens left to right
    for i in range(K):
        token = int(draft_tokens[i].item())

        p_token = float(target_probs[i, token].item())
        q_token = float(draft_probs[i, token].item())

        if q_token <= 0:
            raise ValueError("Draft token must have positive probability")

        # 2. Compute acceptance probability
        alpha = min(1.0, p_token / q_token)

        # 3. Uniform random sample
        u = float(torch.rand(
            (),
            device=draft_tokens.device,
            generator=generator,
        ))

        if u < alpha:
            # Accept draft token
            emitted.append(token)
            continue

        # 4. First rejection: correction distribution
        residual = (
            target_probs[i] - draft_probs[i]
        ).clamp_min(0)

        total = residual.sum()

        if total <= 0:
            raise ValueError("Residual distribution has zero mass")

        residual = residual / total

        # 5. Sample replacement token
        replacement = int(
            torch.multinomial(
                residual, 1, generator=generator
            ).item()
        )

        emitted.append(replacement)

        # Stop immediately on first rejection
        return emitted, i, "rejected"

    # 6. All K proposals accepted
    bonus = int(
        torch.multinomial(
            target_probs[K], 1, generator=generator
        ).item()
    )

    emitted.append(bonus)

    return emitted, K, "all_accepted"
