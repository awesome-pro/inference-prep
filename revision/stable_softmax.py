import torch

def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    amax = x.max(dim=-1, keepdim=True).values
    shifted = x - amax
    exp_values = torch.exp(shifted)
    return exp_values / exp_values.sum(dim=-1, keepdim=True)
