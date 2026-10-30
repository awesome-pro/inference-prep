import copy
import statistics
import time

import torch
from torch import nn

torch.set_num_threads(4)

assert torch.backends.mps.is_available(), "MPS is unavailable"

cpu_model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
).eval()

mps_model = copy.deepcopy(cpu_model).to("mps")


def benchmark(model, x, device, steps=50):
    times = []

    with torch.inference_mode():
        # Warmup
        for _ in range(10):
            model(x)

        if device == "mps":
            torch.mps.synchronize()

        for _ in range(5):
            if device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()

            for _ in range(steps):
                model(x)

            if device == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed / steps * 1000)

    return statistics.median(times)


print("PyTorch:", torch.__version__)

for batch_size in [1, 256, 1024]:
    x_cpu = torch.randn(batch_size, 512)
    x_mps = x_cpu.to("mps")

    cpu_ms = benchmark(cpu_model, x_cpu, "cpu")
    mps_ms = benchmark(mps_model, x_mps, "mps")

    print(f"\nBatch size: {batch_size}")
    print(f"CPU: {cpu_ms:.4f} ms/forward")
    print(f"MPS: {mps_ms:.4f} ms/forward")
    print(f"Speedup: {cpu_ms / mps_ms:.2f}x")