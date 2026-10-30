import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function

torch.set_num_threads(4)

model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
).eval()

x = torch.randn(256, 512)

# Warmup
with torch.inference_mode():
    for _ in range(10):
        model(x)

# Profile CPU tensor operations
with torch.inference_mode():
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(20):
            with record_function("model_forward"):
                model(x)

print(
    prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15,
    )
)