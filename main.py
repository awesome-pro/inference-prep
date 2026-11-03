import torch

x = torch.arange(12).reshape(3, 4)
print(x.shape)     # (3, 4)
print(x.stride())   # (4, 1)

y = x.transpose(0, 1)
print(y.shape)            # (4, 3)
print(y.stride())         # (1, 4)
print(y.is_contiguous())  # False

z = y.view(12)  # RuntimeError