import torch

def apply_penalities(
        logits: torch.Tensor,       # [B, V]
        output_ids: torch.Tensor,   # [B, L] tokens already generated
        repetition_penality: torch.Tensor | None = None,
        frequency_penalty: torch.Tensor | None = None,
        presence_penalty: torch.Tensor | None = None 
) -> torch.Tensor:
    pass

def apply_top_k(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    k = torch.where(k <= 0, vocab, k).clamp(max=vocab)

    kth_values = logits.topk(int(k.max()), dim=-1).values
    thresold = kth_values.gather(1, (k-1)).unsqueeze(1)
    return logits.masked_fill(logits < thresold, float("-inf"))

def apply_top_p(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    sorted_logits, index = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)

    drop = (cumulative - probs) > p.unsqueeze(1)
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))

    return torch.full_like(logits, float("-inf")).scatter(1, index, sorted_logits)

def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """returns (sampled tokesn [B], final probabilities: [B, V])"""
    greedy = temperature <= 0
    argmax = logits.argmax(dim=-1)

    logits = logits / temperature.clamp(min=1e-5).unsqueeze(1)

    if top_k is not None:
        logits = apply_top_k(logits, top_k)
    if top_p is not None:
        logits = apply_top_p(logits, top_p)

    probs = logits.softmax(dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).unsqueeze(1)
    tokens = torch.where(greedy, argmax, tokens)

    return tokens, probs