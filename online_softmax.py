import torch

def stable_softmax(x: torch.Tensor, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    shifted = x - x_max
    exp_x = torch.exp(shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


x = torch.tensor([
    [1000., 1001., 1002.],
    [-1000., -1001., -1002.],
    [1., 2., 3.]
])

actual = stable_softmax(x, dim=-1)
expected = torch.softmax(x, dim=-1, dtype=torch.float32)

torch.testing.assert_close(actual, expected)

# Every row should sum to 1
assert torch.allclose(
    actual.sum(dim=-1),
    torch.ones(x.shape[0])
)

# Adding a constant must not change probabilities
assert torch.allclose(
    stable_softmax(x),
    stable_softmax(x + 10000)
)

print("All tests passed!")
