"""
sampler.py

Batched sampling: logits [B, V] in, one token per row out.

The interview version of this question is always the same:

    "each request has its own temperature, top_k and top_p.
     Sample one token per row. No Python loop over rows."

That last sentence is the whole problem. You cannot slice, because row 0 may
want top_k=50 and row 1 may want none. So every filter has to be a mask
computed from each row's own parameter.

The pipeline, in order (same order as vLLM and HuggingFace):

    1. penalties      logits -> logits   (repetition / frequency / presence)
    2. temperature    logits / T
    3. top-k          mask everything below the k-th largest
    4. top-p          mask the nucleus tail
    5. softmax        logits -> probs
    6. multinomial    probs -> one token

Order matters. Penalties before temperature, or a repetition penalty gets
scaled differently per request. top-k before top-p, because top-p is defined
on the already-truncated distribution.

The five things that get people rejected
----------------------------------------
  1. temperature == 0  -> divide by zero. Must branch to argmax.
  2. top-k per row     -> `logits.topk(k)` does not broadcast over rows.
  3. top-p boundary    -> naive `cumsum > p` drops the token that crosses p.
                          The right test is `cumsum - probs > p`.
  4. masking with a big negative number instead of -inf, then softmax
     leaks probability into the masked tokens.
  5. sampling after masking without renormalising (softmax does it for you).
"""

import torch


# ======================================================================
# 1. penalties -- applied to logits, before anything else
# ======================================================================
def apply_penalties(
    logits: torch.Tensor,           # [B, V] one row per request being sampled
    output_ids: torch.Tensor,       # [B, L] tokens each request already generated
    repetition_penalty: torch.Tensor | None = None,  # [B]
    frequency_penalty: torch.Tensor | None = None,   # [B]
    presence_penalty: torch.Tensor | None = None,    # [B]
) -> torch.Tensor:
    """Discourage tokens the request has already produced.

    `output_ids` is what each request has generated SO FAR. Logits carry no
    memory, so this tensor is the only place the sampler learns about history.

    B = batch size (requests sampled together this step).
    L = the LONGEST generated sequence in the batch. Requests stop at
        different lengths, and a tensor must be rectangular, so shorter rows
        are padded with -1. (Same reason block_tables is padded in the KV file.)

    We only need one thing from this ragged data: how often each token id
    appears per request. That is a [B, V] matrix, and `scatter_add_` builds it
    by scattering a 1 into every generated id's column:

        output_ids = [[1, 3, 1],      counts = [[0, 2, 0, 1, 0],
                      [4, -1, -1]]              [0, 0, 0, 0, 1]]

    From there it is plain arithmetic on [B, V].

    repetition_penalty (CTRL): divide positive logits, multiply negative ones.
        > 1 discourages repeats, < 1 encourages them.
    frequency_penalty (OpenAI): subtract penalty * count.
    presence_penalty  (OpenAI): subtract penalty * (count > 0), once per token.
    """
    batch, vocab = logits.shape

    # -1 marks padding, not a token. It must never reach scatter_add_, because
    # a -1 index means "last column" and would corrupt the real count there.
    valid = output_ids >= 0
    counts = torch.zeros(batch, vocab, dtype=logits.dtype)
    counts.scatter_add_(1, output_ids.clamp(min=0), valid.to(logits.dtype))

    if repetition_penalty is not None:
        pen = repetition_penalty.unsqueeze(1)
        seen = counts > 0
        logits = torch.where(seen & (logits > 0), logits / pen, logits)
        logits = torch.where(seen & (logits <= 0), logits * pen, logits)
    if frequency_penalty is not None:
        logits = logits - frequency_penalty.unsqueeze(1) * counts
    if presence_penalty is not None:
        logits = logits - presence_penalty.unsqueeze(1) * (counts > 0).to(logits.dtype)
    return logits


# ======================================================================
# 2. filtering
# ======================================================================
def apply_top_k(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Keep each row's k largest logits. k <= 0 means "keep everything".

    The trick: take the same top-k.max() columns for every row, then keep only
    the first k_i of them by comparing against row i's own k_i-th value.
    """
    vocab = logits.shape[-1]
    k = torch.where(k <= 0, vocab, k).clamp(max=vocab)      # [B]

    kth_values = logits.topk(int(k.max()), dim=-1).values   # [B, k_max]
    threshold = kth_values.gather(1, (k - 1).unsqueeze(1))  # [B, 1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Nucleus sampling: keep the shortest prefix of sorted tokens with mass >= p.

    p <= 0 keeps one token, p >= 1 keeps all of them. No special cases needed.
    """
    sorted_logits, index = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # `cumulative - probs` is the mass *before* this token. Dropping when it
    # already exceeds p keeps the token that crosses the boundary.
    drop = (cumulative - probs) > p.unsqueeze(1)
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    # Put the tokens back in vocabulary order.
    return torch.full_like(logits, float("-inf")).scatter(1, index, sorted_logits)


# ======================================================================
# 3. the sampler
# ======================================================================
def sample(
    logits: torch.Tensor,                 # [B, V]
    temperature: torch.Tensor,            # [B]
    top_k: torch.Tensor | None = None,    # [B], 0 = off
    top_p: torch.Tensor | None = None,    # [B], 1.0 = off
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sampled tokens [B], final probabilities [B, V])."""
    greedy = temperature <= 0
    argmax = logits.argmax(dim=-1)        # computed on the *raw* logits

    # clamp keeps the divide finite; greedy rows are overridden at the end.
    logits = logits / temperature.clamp(min=1e-5).unsqueeze(1)

    if top_k is not None:
        logits = apply_top_k(logits, top_k)
    if top_p is not None:
        logits = apply_top_p(logits, top_p)

    probs = logits.softmax(dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    tokens = torch.where(greedy, argmax, tokens)
    return tokens, probs


# ======================================================================
# 4. a slow per-row reference, used only to check the batched version
# ======================================================================
def reference_probs(logits, temperature, top_k, top_p):
    """The obvious implementation: one row at a time, plain Python."""
    out = torch.empty_like(logits)
    for b in range(logits.shape[0]):
        row = logits[b] / max(float(temperature[b]), 1e-5)

        k = int(top_k[b])
        if 0 < k < row.numel():
            kth = row.topk(k).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))

        p = float(top_p[b])
        values, index = row.sort(descending=True)
        probs = values.softmax(dim=-1)
        keep = (probs.cumsum(dim=-1) - probs) <= p
        values = values.masked_fill(~keep, float("-inf"))

        row = torch.full_like(row, float("-inf")).scatter(0, index, values)
        out[b] = row.softmax(dim=-1)
    return out


# ======================================================================
# demo
# ======================================================================
def main() -> None:
    torch.manual_seed(0)

    batch, vocab = 4, 12
    logits = torch.randn(batch, vocab) * 2.0

    temperature = torch.tensor([1.0, 0.0, 0.7, 1.5])
    top_k = torch.tensor([0, 3, 5, 0])           # 0 = off
    top_p = torch.tensor([1.0, 1.0, 0.0, 0.8])   # 1.0 = off

    tokens, probs = sample(logits, temperature, top_k, top_p)

    print(f"{'row':>3} {'temp':>5} {'top_k':>6} {'top_p':>6} | "
          f"{'kept':>5} {'argmax':>7} {'sampled':>8} {'p(sampled)':>11}")
    for b in range(batch):
        kept = int((probs[b] > 0).sum())
        print(f"{b:>3} {float(temperature[b]):>5.1f} {int(top_k[b]):>6} "
              f"{float(top_p[b]):>6.1f} | {kept:>5} {int(logits[b].argmax()):>7} "
              f"{int(tokens[b]):>8} {float(probs[b, tokens[b]]):>11.3f}")

    # ---- the check that matters: batched == per-row reference -------------
    expected = reference_probs(logits, temperature, top_k, top_p)
    assert torch.allclose(probs, expected, atol=1e-6), "filter mismatch"
    print("  (kept = tokens with nonzero probability. temperature 0 collapses")
    print("   that to 1 by design, which is why row 1 shows 1 and not 3.)")
    print("\nbatched probabilities == per-row reference")

    # ---- we only ever sample inside the kept set --------------------------
    assert (probs.gather(1, tokens.unsqueeze(1)) > 0).all()

    # ---- temperature 0 is greedy ------------------------------------------
    assert tokens[1] == logits[1].argmax()

    # ---- top_k = 1 is greedy, for every row -------------------------------
    ones = torch.ones(batch)
    greedy_tokens, _ = sample(logits, ones, torch.ones(batch, dtype=torch.long), ones)
    assert torch.equal(greedy_tokens, logits.argmax(dim=-1))

    # ---- penalties ---------------------------------------------------------
    row = torch.tensor([[2.0, -2.0, 1.0, 1.0, 1.0]])
    seen = torch.tensor([[0, 1]])

    repeated = apply_penalties(row, seen, repetition_penalty=torch.tensor([2.0]))
    assert repeated[0, 0] == 1.0    # positive: 2.0 / 2
    assert repeated[0, 1] == -4.0   # negative: -2.0 * 2
    assert repeated[0, 2] == 1.0    # unseen: untouched

    present = apply_penalties(row, seen, presence_penalty=torch.tensor([1.0]))
    assert present[0, 0] == 1.0 and present[0, 1] == -3.0

    # token 0 appears twice, so frequency scales with the count
    counted = apply_penalties(row, torch.tensor([[0, 0, 1]]),
                              frequency_penalty=torch.tensor([0.5]))
    assert counted[0, 0] == 1.0    # 2.0 - 0.5*2
    assert counted[0, 1] == -2.5   # -2.0 - 0.5*1

    print("penalties, greedy and top_k=1 all check out")


if __name__ == "__main__":
    main()
