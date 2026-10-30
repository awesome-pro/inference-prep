import torch

def stable_softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values

    shifted = x - x_max

    exp_x = torch.exp(shifted)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

x = torch.Tensor([1000, 1001, 1002])

print(stable_softmax(x))