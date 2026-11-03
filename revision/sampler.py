import torch

def sample_next_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k=None,
        top_p=1.0
):
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    assert temperature > 0
    assert 0 < top_p <= 1

    scores = logits.float() / temperature

    # top-k
    if top_k is not None:
        assert top_k > 0

        k = min(top_k, scores.size(-1))
        indices = torch.topk(scores, k, dim=-1).indices

        filtered = torch.full_like(scores, float("-inf"))
        scores = filtered.scatter(
            -1, indices, scores.gather(-1, indices)
        )

    if top_p < 1.0:
        assert 0 < top_p <= 1

        sorted_scores, sorted_indices = torch.sort(
            scores, descending=True, dim=-1
        )

        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)

        keep = (cum - sorted_probs) <= top_p

        filtered_scores = sorted_scores.masked_fill(~keep, float("-inf"))

        scores = torch.empty_like(filtered_scores).scatter(-1, sorted_indices, filtered_scores)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)