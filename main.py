
import torch

x = torch.tensor([10.0, 20.0, 30.0, 40.0])

pairs = x.reshape(2, 2)
print("Pairs:\n", pairs)

first = pairs[:, 0]
second = pairs[:, 1]

print("First:", first)
print("Second:", second)

cos_angle = 0.5403
sin_angle = 0.8415

new_first = first * cos_angle - second * sin_angle
new_second = first * sin_angle + second * cos_angle

print("Rotated first:", new_first)
print("Rotated second:", new_second)
