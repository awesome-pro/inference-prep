"""
spec_decode.py

Speculative decoding: the accept/reject loop and the verifier.
Read after sampler.py.

The idea
--------
Decoding one token at a time wastes the GPU: a forward pass costs the same
whether you feed it 1 token or 100. So:

  1. a small DRAFT model proposes K tokens        (K cheap passes)
  2. the big TARGET model scores all of them      (ONE pass, K+1 positions)
  3. we accept the ones the target agrees with, and stop at the first
     disagreement

If the draft is right, one target pass produced K+1 tokens instead of 1.

The promise, and the thing to say out loud
------------------------------------------
The output distribution is IDENTICAL to sampling from the target directly.
Speculative decoding is a pure speed trick -- there is no quality loss, no
approximation, no temperature clawback. The accept/reject rule is carefully
constructed to guarantee that, and this file tests it empirically.

The two halves
--------------
  verify()              the verifier: ONE target pass -> p of shape [K+1, V]
  speculative_step()    the loop: compare p vs q, accept, correct, or bonus

The accept/reject rule
----------------------
For draft token x_i with draft probability q and target probability p:

    accept with probability  min(1, p[x_i] / q[x_i])

If rejected, do NOT resample from p. Sample from the residual

    r = normalize( max(0, p - q) )

That correction is the whole trick. Without it you would either over-sample
the token the draft liked, or lose probability mass entirely.

If all K are accepted, take one bonus token from p[K] -- the target already
computed that distribution, so it is free.

Three ways to get this wrong
----------------------------
  1. Forgetting the bonus token when everything is accepted.
  2. Resampling from p on rejection instead of from max(0, p - q).
     That biases toward tokens the draft already over-weighted.
  3. The verifier's slice. p[i] must be the target's distribution at the
     position that PRODUCED draft token i, so it starts at n-1, not n.
"""

from dataclasses import dataclass

import torch


# ======================================================================
# toy models -- real ones are transformers, these are lookup tables
# ======================================================================
def make_model(vocab: int, seed: int, noise: float = 0.0):
    """Return model(tokens) -> logits [len(tokens), V].

    Row i is the prediction for the token AFTER position i. Adding `noise`
    gives a different-but-similar model, which is exactly the draft/target
    relationship we want to simulate.
    """

    def build():
        g = torch.Generator().manual_seed(seed)
        table = torch.randn(vocab, vocab, generator=g) * 1.5
        if noise:
            table = table + torch.randn(vocab, vocab, generator=g) * noise
        return table

    table = build()

    def model(tokens: list[int]) -> torch.Tensor:
        return table[torch.tensor(tokens, dtype=torch.long)]

    return model


# ======================================================================
# 1. the draft
# ======================================================================
def propose(draft, prefix: list[int], k: int, generator):
    """Ask the draft model for k tokens, one forward pass each.

    Returns the tokens and the q distribution that produced each of them.
    This is the part that costs O(k) passes -- it is only affordable because
    the draft model is small.
    """
    tokens, probs = [], []
    context = list(prefix)
    for _ in range(k):
        q = draft(context)[-1].softmax(dim=-1)
        token = int(torch.multinomial(q, 1, generator=generator))
        tokens.append(token)
        probs.append(q)
        context.append(token)
    return tokens, torch.stack(probs)  # [k], [k, V]


# ======================================================================
# 2. the verifier
# ======================================================================
def verify(target, prefix: list[int], draft_tokens: list[int]) -> torch.Tensor:
    """Score every draft token in ONE target pass.

    The target sees [prefix + draft] and produces a distribution at every
    position. We keep the last K+1 of them:

        p[0]   ... the distribution that SHOULD have produced draft[0]
        p[1]   ... the distribution that SHOULD have produced draft[1]
        ...
        p[K-1] ... the distribution that SHOULD have produced draft[K-1]
        p[K]   ... the bonus distribution, after the last draft token

    Returns [K+1, V].

    The indexing is the subtle part. With prefix length n, draft token 0 sits
    at position n, and the distribution predicting it is at position n-1.
    So the slice starts at n-1, not n. Off by one here and every comparison
    after it is wrong.
    """
    n = len(prefix)
    assert n >= 1, "need at least one prefix token to condition on"
    k = len(draft_tokens)
    logits = target(prefix + draft_tokens)  # [n + k, V] -- ONE pass
    return logits[n - 1 : n + k].softmax(dim=-1)


# ======================================================================
# 3. the accept/reject loop
# ======================================================================
@dataclass
class Step:
    tokens: list[int]  # what this step emits (1 .. K+1 tokens)
    accepted: int      # how many draft tokens survived
    bonus: bool        # True if we accepted all K and took the extra token


def speculative_step(target, draft, prefix: list[int], k: int, generator) -> Step:
    draft_tokens, q = propose(draft, prefix, k, generator)   # [K], [K, V]
    p = verify(target, prefix, draft_tokens)                 # [K+1, V]

    emitted: list[int] = []
    for i, token in enumerate(draft_tokens):
        ratio = float(p[i, token] / q[i, token])
        if float(torch.rand((), generator=generator)) < min(1.0, ratio):
            emitted.append(token)
            continue

        # Rejected. Correct with the residual max(0, p - q), not with p.
        residual = (p[i] - q[i]).clamp(min=0)
        total = float(residual.sum())
        if total <= 0.0:
            # p == q at this position; the residual is empty. p is fine.
            emitted.append(int(torch.multinomial(p[i], 1, generator=generator)))
        else:
            emitted.append(
                int(torch.multinomial(residual / total, 1, generator=generator))
            )
        return Step(emitted, accepted=i, bonus=False)

    # Everything was accepted, so the target's distribution at position K is
    # already computed. Take one more token for free.
    emitted.append(int(torch.multinomial(p[k], 1, generator=generator)))
    return Step(emitted, accepted=k, bonus=True)


# ======================================================================
# checks
# ======================================================================
def main() -> None:
    torch.manual_seed(0)
    vocab, k = 3, 3

    target = make_model(vocab, seed=1)
    # Same base table plus noise -> a realistic draft: right often enough to
    # win, wrong often enough that you can see the correction path.
    draft = make_model(vocab, seed=1, noise=1.4)
    prefix = [0]

    # ---- 1. the verifier's one pass equals K+1 separate passes -----------
    gen = torch.Generator().manual_seed(7)
    draft_tokens, q = propose(draft, prefix, k, gen)
    p = verify(target, prefix, draft_tokens)

    assert p.shape == (k + 1, vocab), p.shape
    for i in range(k + 1):
        separate = target(prefix + draft_tokens[:i])[-1].softmax(dim=-1)
        assert torch.allclose(p[i], separate, atol=1e-6), f"p[{i}] is wrong"
    print(f"verifier: 1 pass -> p{tuple(p.shape)} "
          f"== {k + 1} separate target passes")

    # ---- 2. a perfect draft is accepted every time -----------------------
    perfect = target
    step = speculative_step(perfect, perfect, prefix, k,
                            torch.Generator().manual_seed(3))
    assert step.accepted == k and step.bonus and len(step.tokens) == k + 1
    print(f"perfect draft: accepted {step.accepted}/{k} + bonus "
          f"-> {len(step.tokens)} tokens in one target pass")

    # ---- 3. one real step, printed ---------------------------------------
    step = speculative_step(target, draft, prefix, k,
                            torch.Generator().manual_seed(11))
    print(f"\none step: draft={draft_tokens} accepted={step.accepted}/{k} "
          f"bonus={step.bonus} emitted={step.tokens}")

    # ---- 4. the output distribution equals the target's ------------------
    # This is the claim that matters: no quality loss.
    trials = 20000
    gen = torch.Generator().manual_seed(123)
    counts = torch.zeros(vocab)
    for _ in range(trials):
        out = speculative_step(target, draft, prefix, k, gen)
        counts[out.tokens[0]] += 1

    empirical = counts / trials
    expected = target(prefix)[-1].softmax(dim=-1)
    error = float((empirical - expected).abs().max())
    assert error < 0.02, (empirical.tolist(), expected.tolist())

    print(f"\ndistribution check over {trials} steps")
    print(f"  empirical   {[round(v, 3) for v in empirical.tolist()]}")
    print(f"  target      {[round(v, 3) for v in expected.tolist()]}")
    print(f"  max error   {error:.4f}")

    # ---- 5. how much did we actually win? --------------------------------
    steps = 5000
    gen = torch.Generator().manual_seed(99)
    emitted = 0
    accepted = 0
    histogram = [0] * (k + 1)
    for _ in range(steps):
        out = speculative_step(target, draft, prefix, k, gen)
        emitted += len(out.tokens)
        accepted += out.accepted
        histogram[out.accepted] += 1

    print(f"\nefficiency over {steps} steps")
    print(f"  acceptance rate   {accepted / (steps * k):.3f}  (draft tokens kept)")
    print(f"  tokens per step   {emitted / steps:.2f}  (max {k + 1})")
    print(f"  accepted counts   {histogram}  (index = accepted, last = all K)")

    print("\nThe distribution matched the target. That is the point:")
    print("speculative decoding buys throughput, and costs nothing in quality.")


if __name__ == "__main__":
    main()
