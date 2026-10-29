import torch

def sample(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor):
    greedy = temperature <= 0

    # temperature
    logits = logits / temperature.clamp(min=1e-5).unsqueeze(1)

    # After this position == rank and both filters are masks
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    sorted_probs = sorted_logits.softmax(dim=-1)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs

    # top k
    vocab = logits.shape[-1]
    k = torch.where(top_k <= 0, vocab, top_k).clamp(max=vocab)
    keep = torch.arange(vocab) < k.unsqueeze(1)

    # top p
    keep = keep & (mass_before <= top_p.unsqueeze(1))

    # mask renormalize, draw
    probs = sorted_probs.masked_fill(~keep, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    rank = torch.multinomial(probs, num_samples=1)
    tokens = order.gather(1, rank).squeeze(1)
    return torch.where(greedy, logits.argmax(dim=-1), tokens)
