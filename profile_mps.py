import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
).eval().to("mps")

x = torch.randn(256, 512, device="mps")

with torch.inference_mode():
    # Warm up outside the capture
    for _ in range(10):
        model(x)
    torch.mps.synchronize()

    # Capture only a small amount of GPU work
    with torch.mps.profiler.metal_capture("mlp_trace.gputrace"):
        for _ in range(3):
            model(x)
        torch.mps.synchronize()

print("Capture complete")