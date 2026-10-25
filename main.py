import torch

A = torch.randn(2, 3)
B = torch.randn(3, 4)

C = torch.einsum("ih,kj->ij", A, B)

print(C)
print(C.shape)
# torch.Size([2, 4])

