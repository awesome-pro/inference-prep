"""
sampler.py

Logits -> token ids, batched. Three filters and one sort.

    temperature -> top_k -> top_p -> softmax -> multinomial

Every request can have its own settings, so nothing is ever sliced or
looped: everything is a mask over a sorted row.
"""

import torch


def sample(logits, temperature, top_k, top_p):
    """logits [B, V] -> token ids [B].

    temperature: [B], 0 means greedy
    top_k:       [B], 0 means off
    top_p:       [B], 1.0 means off
    """
    greedy = temperature <= 0

    # 1. temperature
    logits = logits / temperature.clamp(min=1e-5).unsqueeze(1)

    # 2. sort once. After this, position == rank, and both filters are masks.
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs

    # 3. top_k: keep ranks 0 .. k-1
    vocab = logits.shape[-1]
    k = torch.where(top_k <= 0, vocab, top_k).clamp(max=vocab)
    keep = torch.arange(vocab) < k.unsqueeze(1)

    # 4. top_p: keep while the mass before this token is still under p
    keep &= mass_before <= top_p.unsqueeze(1)

    # 5. mask, renormalize, draw
    probs = probs.masked_fill(~keep, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    rank = torch.multinomial(probs, num_samples=1)
    tokens = order.gather(1, rank).squeeze(1)
    return torch.where(greedy, logits.argmax(dim=-1), tokens)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, vocab = 3, 10
    logits = torch.randn(batch, vocab)

    temperature = torch.tensor([1.0, 0.5, 0.0])
    top_k = torch.tensor([0, 3, 0])
    top_p = torch.tensor([0.9, 1.0, 1.0])

    print("temp   ", temperature.tolist())
    print("top_k  ", top_k.tolist())
    print("top_p  ", top_p.tolist())
    for _ in range(3):
        print("sample ", sample(logits, temperature, top_k, top_p).tolist())

    ones = torch.ones(batch)
    one = torch.ones(batch, dtype=torch.long)
    zeros = torch.zeros(batch)

    # temperature 0, top_k 1, and top_p 0 all mean "take the argmax"
    assert torch.equal(sample(logits, zeros, one, ones), logits.argmax(-1))
    assert torch.equal(sample(logits, ones, one, ones), logits.argmax(-1))
    assert torch.equal(sample(logits, ones, zeros, zeros), logits.argmax(-1))
    print("argmax cases ok")
